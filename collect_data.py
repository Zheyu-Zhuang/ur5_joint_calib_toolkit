#!/usr/bin/env python

import argparse
import glob
import json
import os

import cv2
import cv2.aruco as aruco
# ROS Sys Pkg
import message_filters
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState

# Sys Pkg

bridge = CvBridge()

parser = argparse.ArgumentParser()
#
parser.add_argument('--dataset', type=str, help='directory for saving the data')
#
parser.add_argument('--aruco_bit', type=int, default=4,
                    help='format of aruco dictionary')
parser.add_argument('--board_dim', type=int, nargs="+", default=[4, 6],
                    help='width, height of checkerboard (unit: squares)')
parser.add_argument('--square_len', type=float, default=0.029,
                    help='measured in metre')
parser.add_argument('--marker_len', type=float, default=0.022,
                    help='measured in metre')
parser.add_argument('--camera_topic', type=str, default='/camera/color/image_raw')
args = parser.parse_args()

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_1000)  # 4X4 = 4x4 bit markers
charuco_board = aruco.CharucoBoard_create(args.board_dim[0], args.board_dim[1],
                                          args.square_len, args.marker_len,
                                          aruco_dict)

# Todo add save images to separate table surface calibration subfolder

class SampleCollector:
    def __init__(self):
        self.joint_sub = message_filters.Subscriber(
            "/joint_states", JointState)
        self.img_sub = message_filters.Subscriber(args.camera_topic, Image)
        self.synced_msgs = message_filters.ApproximateTimeSynchronizer(
            [self.img_sub, self.joint_sub], 10, 0.1)
        self.synced_msgs.registerCallback(self.sync_callback)
        self.dataset_dir = os.path.join('./dataset', args.dataset)
        self.cv2_img = None
        self.joint_config = None

    def sync_callback(self, image_msg, joint_msg):
        self.cv2_img = bridge.imgmsg_to_cv2(image_msg)
        self.joint_config = self.msg_to_joint_config(joint_msg)

    def get_sample(self):
        meta_dir = os.path.join(self.dataset_dir, 'meta')
        image_dir = os.path.join(self.dataset_dir, 'images')
        check_path(meta_dir)
        check_path(image_dir)
        if (self.cv2_img is not None) and (self.joint_config is not None):
            self.check_corners(self.cv2_img)
            n_files = len(glob.glob1(meta_dir, "*.json"))
            sample_id = "%03d" % n_files
            image_file_path = os.path.join(image_dir, "%s.png" % sample_id)
            meta_file_path = os.path.join(meta_dir, "%s.json" % sample_id)
            dict_temp = {'image_name': "%s.png" % sample_id,
                         'joint_config': self.joint_config.reshape(6).tolist()}
            json.dump(dict_temp, open(meta_file_path, 'w'))
            img_rgb = cv2.cvtColor(self.cv2_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_file_path, img_rgb, )
            print("Sample %s is captured" % sample_id)
        else:
            print("Image or joint configuration is None")

    @staticmethod
    def msg_to_joint_config(joint_msg):
        order = [2, 1, 0, 3, 4, 5]
        q = np.asarray([joint_msg.position[i] for i in order])
        q = np.around(q, decimals=4)
        q = q.reshape((6, 1))
        return q

    @staticmethod
    def check_corners(cv2_img):
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(image=gray,
                                              dictionary=aruco_dict)
        if ids is not None and len(ids) > 5:
            print("{} markers detected".format(len(ids)))
        else:
            raise Exception('Need to detect more corners')


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    rospy.init_node("aruco_calib", anonymous=True)
    collector = SampleCollector()
    rospy.sleep(0.5)
    collector.get_sample()
