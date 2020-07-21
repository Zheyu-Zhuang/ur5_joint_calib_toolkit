import os
import yaml
import math

# ROS Sys Pkg
import json
import argparse
import numpy as np
import cv2
import cv2.aruco as aruco
import glob
from ur5_kinematics import UR5Kinematics as UR5Kin
from tqdm import tqdm

parser = argparse.ArgumentParser()
#
parser.add_argument('--dataset', type=str, help='directory for saving the data')
#
parser.add_argument('--aruco_bit', type=int, default=4,
                    help='format of aruco dictionary')
parser.add_argument('--board_dim', type=int, hnargs="+", detault=[4, 6],
                    help='width, height of checkerboard (unit: squares)')
parser.add_argument('--square_len', type=float, default=0.029,
                    help='measured in metre')
parser.add_argument('--marker_len', type=float, default=0.022,
                    help='measured in metre')
parser.add_argument('--camera_topic', type=str, default='/camera/image_color')
args = parser.parse_args()

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)  # 4X4 = 4x4 bit markers
charuco_board = aruco.CharucoBoard_create(args.board_dim[0], args.board_dim[1],
                                          args.square_len, args.marker_len,
                                          aruco_dict)
ur5_kin = UR5Kin()

class MVC:
    def __init__(self):
        # Todo: load all images and meta data from the input directory
        self.M_X_E = np.array([[0, -1, 0, 0.23],
                               [-1, 0, 0, 0.19],
                               [0, 0, -1, -0.011],
                               [0, 0, 0, 1]])
        # offset between end-effector and table hen board is on table
        self.board_height_offset = np.eye(4)
        self.board_height_offset[2, 3] = -0.027
        # offset from the ee reference frame to the calibration board
        self.E_X_T = np.matmul(np.linalg.inv(self.M_X_E),
                               self.board_height_offset)
        self.cv2robotics= np.array([[0, -1, 0, 0], [0, 0, -1, 0],
                                    [1, 0, 0, 0], [0, 0, 0, 1]])
        self.dataset_dir = os.path.join('./dataset', args.dataset)
        # if calibration file exist?
        self.K = None
        self.distortion = None

    def start(self):
        intrinsics_path = os.path.join(self.dataset_dir, 'intrinsics.json')
        if os.path.exists(intrinsics_path):
            f = json.load(open(intrinsics_path))
            self.K = np.array(f['K']).reshape(3, 3)
            self.distortion = np.array(f['distortion']).reshape(1, 5)
            print('Camera Intrinsics loaded...')
        else:
            data_dict = self.process_all_images()
            all_corners, all_ids = [], []
            if data_dict:
                all_image_names = data_dict.keys()
                n_images = len(all_image_names)
                for image_name in all_image_names:
                    all_corners.append(data_dict[image_name]["corners"])
                    all_ids.append(data_dict[image_name]["ids"])
                    im_size = data_dict[image_name]["im_size"][:2]
                print("Calibrating Camera Intrinsics...")
                intrinsics_dict = self.get_camera_intrinsics(all_corners,
                                                             all_ids, im_size)
                self.K = intrinsics_dict["K"]
                self.distortion = intrinsics_dict["distortion"]
                json.dump(intrinsics_dict, open(intrinsics_path, 'w'))
                B_X_C_stack = np.zeros((n_images, 4, 4))
                camera_calib_path = os.path.join(self.dataset_dir,
                                                 'camera_pose.json')
                calibration_dict = {}
                table_height = 0
                for i, image_name in enumerate(all_image_names):
                    corners_temp = data_dict[image_name]["corners"]
                    ids_temp = data_dict[image_name]["ids"]
                    if args.mode == "camera":
                        print("Estimating Camera Pose w.r.t Robot Base...")
                        B_X_C_stack[i] = self.get_ee_pose_wrt_cam(corners_temp,
                                                              ids_temp)
                        calibration_dict["B_X_C"] = self.SE3_avg(B_X_C_stack)
                        json.dump(calibration_dict, open(camera_calib_path, 'w'))
                    # Todo: if the table height calibration folder exist,
                    #  calibrate table, else skip
                    elif args.mode == "Table":
                        if os.path.exists(camera_calib_path):
                            data_temp = json.load(open(camera_calib_path, 'r'))
                            B_X_C = data_temp["B_X_C"]
                            table_height_temp =\
                                self.get_table_height_from_sample(corners_temp)
                        print()


    def process_all_images(self):
        image_dir = os.path.join(self.dataset_dir, 'images')
        meta_dir = os.path.join(self.dataset_dir, 'meta')
        all_meta_paths = glob.glob(os.path.join(meta_dir, '*.json'))
        data_dict = {}
        print("Extracting corners from all images")
        for meta_path in tqdm(all_meta_paths):
            meta = json.load(open(meta_path))
            image_name = meta['image_name']
            cv2_img = cv2.imread(os.path.join(image_dir, image_name))
            corners_temp, ids_temp = self.get_corners(cv2_img)
            if corners_temp is not None:
                temp_dict = {"corners": corners_temp, "ids": ids_temp,
                             "joint_config": meta["joint_config"],
                             "im_size": cv2_img.shape}
                data_dict[image_name] = temp_dict
            else:
                print('No enough corners detected, skip %s' % image_name)
        return data_dict

    @staticmethod
    def get_camera_intrinsics(all_corners, all_ids, im_size):
        try:
            _, cam_mat, dist_coeffs, _, _ = aruco.calibrateCameraCharuco(
                charucoCorners=all_corners, charucoIds=all_ids,
                board=charuco_board, imageSize=im_size,
                cameraMatrix=None, distCoeffs=None)
            print("K: {}".format(cam_mat[0:3]))
            print("Dist coeffs: {}".format(dist_coeffs[0]))
            return {"K": cam_mat.tolist(), "d": dist_coeffs.tolist()}
        except:
            print("Intrinsic calibration failed. Recalibrating...")

    def get_ee_pose_wrt_cam(self, corners, ids, ref_frame='robotic'):
        retval, rvecs, tvecs = aruco.estimatePoseCharucoBoard(
            corners, ids, charuco_board, self.K, self.distortion)
        if retval:
            Ccv_X_M = np.eye(4)
            Ccv_X_M[:3, 3] = tvecs[:3]
            Ccv_X_M[0:3, 0:3] = cv2.Rodrigues(rvecs[:])[0]
            Ccv_X_E = Ccv_X_M.dot(self.M_X_E)
            E_X_Ccv = np.linalg.inv(Ccv_X_E)
            if ref_frame == 'robotic':
                return E_X_Ccv.dot(self.cv2robotics)
            else:
                return E_X_Ccv
        else:
            print('Board Pose not detected')
            return None

    def get_camera_pose_from_sample(self, corners, ids, joint_config):
        E_X_C = self.get_ee_pose_wrt_cam(corners, ids)
        if E_X_C is not None:
            B_X_E = ur5_kin.get_ee_pose(joint_config)
            return B_X_E.dot(E_X_C)
        else:
            return None

    def get_table_height_from_sample(self, corners, ids, B_X_C, joint_config):
        E_X_C = self.get_ee_pose_wrt_cam(corners, ids)
        if E_X_C is not None:
            B_X_T_vision = B_X_C.dot(np.linalg.inv(E_X_C)).dot(self.E_X_T)
            height_from_vision = B_X_T_vision[2, 3]
            B_X_E = ur5_kin.get_ee_pose(joint_config)
            B_X_T_kin = B_X_E.dot(self.E_X_T)
            height_from_kin = B_X_T_kin[2, 3]
            err = height_from_kin - height_from_vision
            print("Error: {}".format(err))
            return height_from_kin
        else:
            return None

    @staticmethod
    def get_corners(cv2_img):
        gray_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(image=gray_img,
                                              dictionary=aruco_dict)
        if ids is not None and len(ids) > 5:
            _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, gray_img, charuco_board)
            return charuco_corners, charuco_ids
        else:
            # Todo: print image id
            print("Too few markers detected, skip...")
            return None, None
    @staticmethod
    def SO3_log(_SO3):
        try:
            theta = math.acos((np.matrix.trace(_SO3) - 1) / 2)
        except:
            theta = 0.0
        if abs(theta) < 1e-8:
            coeff = 0.5
        else:
            coeff = theta / (2 * math.sin(theta))
        return coeff * (_SO3 - np.matrix.transpose(_SO3))

    @staticmethod
    def so3_to_vec(_so3):
        return np.array([_so3[2, 1], _so3[0, 2], _so3[1, 0]]).reshape(3, 1)

    def so3_exp(self, _so3):
        w = self.so3_to_vec(_so3).reshape(3)
        th = np.linalg.norm(w)
        A = 1
        B = 0.5
        if abs(th) >= 1e-8:
            A = math.sin(th) / th
            B = (1 - math.cos(th)) / pow(th, 2)
        return np.eye(3) + A * _so3 + B * _so3.dot(_so3)

    def SO3_avg(self, SO3_stack, err_th=0.01):
        # implement Karcher mean / geodesic L2-mean on SO(3)
        n = SO3_stack.shape[0]
        out = SO3_stack[0, :, :]
        err = 100
        while err > err_th:
            _so3_err_sum = np.zeros([3, 3])
            for i in range(n):
                _so3_err_sum += self.SO3_log(
                    np.transpose(out).dot(SO3_stack[i, :, :]))
            out = out.dot(self.so3_exp(_so3_err_sum/n))
            err = np.linalg.norm(_so3_err_sum)
            # print(err)
        return out

    def SE3_avg(self, SE3_stack):
        out = np.eye(4)
        out[:, 3] = np.mean(SE3_stack[:, :, 3], axis=0)
        out[:3, :3] = self.SO3_avg(SE3_stack[:, :3, :3])
        return out

if __name__ == "__main__":
    main()
