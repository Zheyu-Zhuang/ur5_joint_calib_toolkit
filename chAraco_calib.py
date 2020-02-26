#!/usr/bin/env python

import os
# Sys Pkg
import sys
import yaml
from ur5_kin import UR5Kin
import math

# ROS Sys Pkg
import message_filters
import numpy as np
import cv2
import cv2.aruco as aruco
import rospy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import copy
import tf
import csv
cwd = os.getcwd()
fk_ur5 = UR5Kin()

################### Initialisation ################################
camera = 2 # number of cameras
rootDir = "/home/zheyu/ur5_calib_toolkit/ZED/"
robotYAML = "{}robotCalibration.yaml".format(rootDir)
cameraYAML = "cameraCalibration.yaml"
extrinsicDataDir = "{}calibDataset/".format(rootDir)
tableDataDir = "{}tableCalibDataset/".format(rootDir)
fileName = "jointAngles.csv"
aruco_dict =  aruco.Dictionary_get(aruco.DICT_4X4_1000) # 4X4 = 4x4 bit markers
charuco_board =  aruco.CharucoBoard_create( 4, 6, 0.063, 0.048, aruco_dict ) # width, height (full board), square length, marker length
##################################################################

class chArucoCalib:

    def __init__(self):

        images = np.array([extrinsicDataDir + f for f in os.listdir(extrinsicDataDir) if f.endswith(".png") ])
        order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in images])
        images = images[order]
        q = []
        with open("{}{}".format(extrinsicDataDir,fileName)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            line_count = 0
            for row in csv_reader:
                l = row
                l = list(map(float, l))
                l = np.asarray(l)
                l = l.reshape((6, 1))
                q.append(l)
                line_count += 1
        
        imagesT = np.array([tableDataDir + f for f in os.listdir(tableDataDir) if f.endswith(".png") ])
        order = np.argsort([int(p.split(".")[-2].split("_")[-1]) for p in imagesT])
        imagesT = imagesT[order]
        qT = []
        with open("{}{}".format(tableDataDir,fileName)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            line_count = 0
            for row in csv_reader:
                l = row
                l = list(map(float, l))
                l = np.asarray(l)
                l = l.reshape((6, 1))
                qT.append(l)
                line_count += 1

        self.camera_matrix = np.zeros(shape=(3,3))
        self.distCoeffs = np.zeros(shape=(1,5))
        self.rvecs = np.zeros(shape=(3,1))
        self.tvecs = np.zeros(shape=(3,1))
        self.current_markers = dict()
        self.corners_all = dict()
        self.ids_all = dict()
        self.frame_counter = 0
        self.im_size = dict()
        self.calibOutput = []
        self.l_B_C_mean = np.eye(3)
        self.l_B_T_mean = np.eye(3)
        self.R_B_C_mean = np.eye(3)
        self.R_B_T_mean = np.eye(3)
        self.t_B_C_mean = np.zeros((3,1))
        self.t_B_T_mean = np.zeros((3,1))
        self.X_M_T = np.eye(4) 
        self.X_M_T[2,3] = -0.027 # offset between top of checkerboard and table
        self.X_M_E = np.array([[0, -1, 0, 0.23],
                                [-1, 0, 0, 0.19],
                                [0, 0, -1, -0.011],
                                [0, 0, 0, 1]])
        self.X_E_T = np.matmul(np.linalg.inv(self.X_M_E),self.X_M_T) # offset between end-effector and table when board is on table
        self.SE3_mean = np.eye(4)
        self.current_markers = 0
        self.corners_all = []
        self.ids_all = []
        self.im_size = ()
        
        # intrinsic calibration
        for cam in range(camera):
            self.frame_counter=0
            for im in images:
                if "cam{}".format(cam) in im:
                    print(im)
                    frame = cv2.imread(im)
                    if self.frame_counter==0:
                        self.im_size = frame.shape[:2]
                    self.get_corners(frame)
                    self.frame_counter=self.frame_counter+1 
            self.intrinsic_calib()

            # extrinsic calibration
            calibratingCamera = True
            calibratingTable = False
            self.frame_counter=0
            for im in images:
                if "cam{}".format(cam) in im:
                    frame = cv2.imread(im)
                    self.est_board_pose(frame)
                    self.get_ref_frames(calibratingCamera,calibratingTable,q[self.frame_counter])
                    self.frame_counter=self.frame_counter+1
            self.writeCameraYAML(cam)
        
        # table calibration
        calibratingCamera = False
        calibratingTable = True
        self.frame_counter=0
        for im in imagesT:
            if "cam0" in im or camera == 1:
                frame = cv2.imread(im)
                self.est_board_pose(frame)
                self.get_ref_frames(calibratingCamera,calibratingTable,qT[self.frame_counter])
                self.frame_counter=self.frame_counter+1
        self.writeRobotYAML()

    def SO3Log(self,rotation):
        # takes SO(3) to so(3)
        try:
            theta = math.acos((np.matrix.trace(rotation)-1)/2)
        except:
            theta = 0.0
        coeff = 0
        if (abs(theta)<1e-8):
            coeff = 0.5
        else:
            coeff = theta/(2*math.sin(theta))
        return coeff * (rotation - np.matrix.transpose(rotation))

    def vex(self,m):
        v = np.zeros(shape=(3,1))
        v[0,0] = m[2,1]
        v[1,0] = m[2,0]
        v[2,0] = m[1,0]
        return v
    
    def so3Exp(self,vel):
        # takes so(3) to SO(3)
        w = self.vex(vel)
        th = np.linalg.norm(w)
        A=1
        B=0.5
        if abs(th) >= 1e-12:
            A = math.sin(th)/th
            B = (1-math.cos(th))/pow(th,2)
        return np.eye(3) + A.dot(vel) + B.dot(vel).dot(vel)

    def SO3_averaging(self, Rs):
        # implement Karcher mean / geodesic L2-mean on SO(3)
        n = Rs.shape[0]
        R = Rs[0, :, :]
        r = np.zeros([3, 3])
        err = 100
        th = 0.01
        while err > th:
            for i in range(n):
                r += self.SO3Log(np.transpose(R).dot(Rs[i, :, :]))
            r = r/n
            R = R.dot(self.so3Exp(r))
            err = np.linalg.norm(r)
            print(err)
        return R
    
    def writeRobotYAML(self):
        with open(robotYAML, 'w') as f:    
            yaml.dump({"X_B_E": self.X_B_E.tolist(),
            "X_B_T": self.X_B_T.tolist(),
            "Error": self.E.tolist()}, f, default_flow_style=False)

    def writeCameraYAML(self, cam_id):
        with open("{}cam{}_{}".format(rootDir, cam_id, cameraYAML), 'w') as f:    
            yaml.dump({"K": self.camera_matrix.tolist(),
            "distortion": self.distCoeffs.tolist(),
            "X_B_C": self.X_B_C.tolist()}, f, default_flow_style=False)

    def intrinsic_calib(self):
        cameraMatrixInit = np.array([[ 1000.,    0., self.im_size[0]/2.],
                                 [    0., 1000., self.im_size[1]/2.],
                                 [    0.,    0.,           1.]])
        try:
            _, cameramatrix, distcoeffs, _, _ = aruco.calibrateCameraCharuco(
            charucoCorners=self.corners_all,
            charucoIds=self.ids_all,
            board=charuco_board,
            imageSize=self.im_size,
            cameraMatrix=cameraMatrixInit,
            distCoeffs=None)
            print("K: {}".format(cameramatrix[0:3]))
            print("Dist coeffs: {}".format(distcoeffs[0]))
            self.camera_matrix[:,:] = cameramatrix
            self.distCoeffs[:,:] = distcoeffs
            self.calibOutput.append({"K": cameramatrix.tolist(),
                "d": distcoeffs.tolist()})
        except: 
            print("Intrinsic calibration failed. Recalibrating...")
            self.frame_counter = 0 

    def get_corners(self,cv2_img):
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(image=gray,dictionary=aruco_dict)  
        if ids is not None and len(ids) > 5: 
            _, Chcorners, Chids = aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board )  
            self.current_markers = len(Chcorners)
            self.corners_all.append(Chcorners)
            self.ids_all.append(Chids)      

    def est_board_pose(self,cv2_img):
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(image=gray,dictionary=aruco_dict)  
        if ids is not None and len(ids) > 5: 
            _, Chcorners, Chids = aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board )  
            retval, rvecs, tvecs = aruco.estimatePoseCharucoBoard(Chcorners, Chids, charuco_board, self.camera_matrix[:,:], self.distCoeffs[:,:]) 
            #######################
            X_C_M = np.eye(4)
            X_C_M[0, 3] = tvecs[0]
            X_C_M[1, 3] = tvecs[1]
            X_C_M[2, 3] = tvecs[2]
            R = cv2.Rodrigues(rvecs[:])[0]
            X_C_M[0:3, 0:3] = R
            X_C_E = X_C_M.dot(self.X_M_E)
            tvecs_new = X_C_E[:3, 3]
            rvecs_new = cv2.Rodrigues(X_C_E[0:3, 0:3])[0]
            # #####################
            QueryImg = aruco.drawDetectedCornersCharuco(cv2_img, Chcorners, Chids,(0,0,255))
            if( retval ):
                QueryImg = aruco.drawAxis(cv2_img, self.camera_matrix[:,:], self.distCoeffs[:,:], rvecs, tvecs, 0.032)  
                QueryImg = aruco.drawAxis(QueryImg, self.camera_matrix[:,:], self.distCoeffs[:,:], rvecs_new, tvecs_new, 0.032)  

            # Display our image
            # cv2.imshow('QueryImage', QueryImg)
            # cv2.waitKey(0)    
        else:
            rvecs = np.zeros((3, 1))
            tvecs = np.zeros((3, 1))
            cv2.imshow('QueryImage', cv2_img)
            print("board not detected")
            cv2.waitKey(1) 
        #print("{} markers detected".format(len(ids)))      
        self.tvecs = tvecs
        self.rvecs= rvecs
       
    def get_ref_frames(self, calibratingCamera, calibratingTable,q):
        X_C_M = np.zeros((4, 4))
        X_C_M[3, 3] = np.ones(1)
        X_C_M[0, 3] = self.tvecs[0]
        X_C_M[1, 3] = self.tvecs[1]
        X_C_M[2, 3] = self.tvecs[2]
        X_C_M[0:3, 0:3] = cv2.Rodrigues(self.rvecs[:])[0]
        Q = np.array([[0,-1, 0, 0],
                     [0, 0, -1, 0],
                     [1, 0, 0, 0],
                     [0, 0, 0, 1]])# Transform from cv convention to robotic
        X_C_E = X_C_M.dot(self.X_M_E) # cv frame
        X_E_C = np.linalg.inv(X_C_E).dot(Q) # robot frame
        self.X_B_E = fk_ur5.get_ee_pose(q) # current pose of end effector wrt ur5 base
        # print("X_B_E: {}".format(self.X_B_E))
        if calibratingCamera == True:           
            self.X_B_C = self.X_B_E.dot(X_E_C) 
            #print("X_B_C: {}".format(self.X_B_C))

            if self.frame_counter == 0:
                self.R_B_C_0 = self.X_B_C[0:3, 0:3]
            l_B_C = self.SO3_averaging(np.matmul(np.transpose(self.R_B_C_0),self.X_B_C[0:3,0:3]))
            self.l_B_C_mean = (self.l_B_C_mean*self.frame_counter + l_B_C) / (self.frame_counter + 1)
            self.R_B_C_mean = np.matmul(self.R_B_C_0, self.so3Exp(self.l_B_C_mean))
            self.t_B_C_mean = (self.t_B_C_mean*self.frame_counter + self.X_B_C[0:3,3].reshape((3,1))) / (self.frame_counter + 1)
            self.X_B_C_mean = np.zeros((4,4))
            self.X_B_C_mean[3,3] = 1
            self.X_B_C_mean[0:3, 0:3] = self.R_B_C_mean
            self.X_B_C_mean[0,3] = self.t_B_C_mean[0,0]
            self.X_B_C_mean[1,3] = self.t_B_C_mean[1,0]
            self.X_B_C_mean[2,3] = self.t_B_C_mean[2,0]
            #print("X_B_C_mean: {}".format(self.X_B_C_mean))

        elif calibratingTable == True: # calibrate table
            X_C_T = np.linalg.inv(Q).dot(X_C_M.dot(self.X_M_T)) # world frame
            self.X_B_T = np.matmul(self.X_B_C_mean, X_C_T)
            #print("X_B_T: {}".format(self.X_B_T))
            
            if self.frame_counter == 0:
                self.R_B_T_0 = self.X_B_T[0:3, 0:3]
            l_B_T = self.SO3_averaging(np.matmul(np.transpose(self.R_B_T_0),self.X_B_T[0:3,0:3]))
            self.l_B_T_mean = (self.l_B_T_mean*self.frame_counter + l_B_T) / (self.frame_counter + 1)
            self.R_B_T_mean = np.matmul(self.R_B_T_0, self.so3Exp(self.l_B_T_mean))
            self.t_B_T_mean = (self.t_B_T_mean*self.frame_counter + self.X_B_T[0:3,3].reshape((3,1))) / (self.frame_counter + 1)
            X_B_T_mean = np.zeros((4,4))
            X_B_T_mean[3,3] = 1
            X_B_T_mean[0:3, 0:3] = self.R_B_T_mean
            X_B_T_mean[0,3] = self.t_B_T_mean[0,0]
            X_B_T_mean[1,3] = self.t_B_T_mean[1,0]
            X_B_T_mean[2,3] = self.t_B_T_mean[2,0]
            #print("X_B_T_mean: {}".format(X_B_T_mean))           
                        
            #print("Table Height w.r.t Base: %.6f" % X_B_T_mean[2, 3])
            X_B_T_from_fk = self.X_B_E.dot(self.X_E_T)
            X_B_T_from_vision = self.X_B_T
            self.E = np.linalg.norm(X_B_T_from_fk - X_B_T_from_vision) # error function
            print("Error: {}".format(self.E))

def main():
    aruco_calib = chArucoCalib()

if __name__ == "__main__":
    main()