#!/usr/bin/env python

import os
# Sys Pkg
import sys
import yaml
from ur5_kin import UR5Kin

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
bridge = CvBridge()
cwd = os.getcwd()
fk_ur5 = UR5Kin()
PGREY = 1
ZED = 2

################### Initialisation ################################
camera = ZED
images = 30
aruco_dict =  aruco.Dictionary_get(aruco.DICT_4X4_1000) # 4X4 = 4x4 bit markers
charuco_board =  aruco.CharucoBoard_create( 4,6, 0.029, 0.022, aruco_dict ) # width, height (full board), square length, marker length
##################################################################

if camera == "ZED":
    camera = ZED
if camera == "PGREY":
    camera = PGREY

class chArucoCalib:

    def __init__(self):
        self.q_sub = message_filters.Subscriber(
            "/joint_states", JointState)
        if camera == PGREY:
            self.image_topic = "/camera/image_color"
            self.img_sub = message_filters.Subscriber(self.image_topic, Image)
            self.synced_msgs = message_filters.ApproximateTimeSynchronizer(
            [self.img_sub, self.q_sub], 10, 0.1)
            self.synced_msgs.registerCallback(self.sync_callback)
                        
        elif camera == ZED:
            self.right_image_topic = "/zed/zed_node/right/image_rect_color"
            self.left_image_topic = "/zed/zed_node/rgb/image_rect_color"
            self.right_img_sub = message_filters.Subscriber(self.right_image_topic, Image)
            self.left_img_sub = message_filters.Subscriber(self.left_image_topic, Image)
            self.synced_msgs = message_filters.ApproximateTimeSynchronizer([self.right_img_sub, self.left_img_sub, self.q_sub],10,0.1)
            self.synced_msgs.registerCallback(self.sync_callback)

        self.current_markers = dict()
        self.frame_counter = np.zeros((camera,1))
     
    def img_callback(self, image_msg, joint_msg, cam_id): 
        cv2_img = bridge.imgmsg_to_cv2(image_msg) # Convert your ROS Image message to OpenCV2
        q = self.joint_msg2q(joint_msg)
        if self.frame_counter[cam_id] < images and self.frame_counter[cam_id]: 
            imageName = "cam{}_{}.png".format(cam_id,int(self.frame_counter[cam_id][0]))
            angles = ["{} {} {} {} {} {}".format(q[0][0],q[1][0],q[2][0],q[3][0],q[4][0],q[5][0])]
            line = [angles]
            if cam_id == 0:
                with open('jointAngles.csv', 'a') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerows(line)
                csvFile.close
            cv2.imwrite(imageName,cv2_img) # save images
            self.get_corners(cv2_img, cam_id)
            print("image: {0}/{1} - {2} total markers detected".format(self.frame_counter[cam_id], images, self.current_markers[cam_id]))       
        elif self.frame_counter[cam_id] == images:
            print("Data collection finished for camera {}. [Ctrl-C] to exit.".format(cam_id))
        self.frame_counter[cam_id]+=1 

    def sync_callback(self, *args):
        image_msg = args[0]
        if len(args)==2: # 1 cam, ur5 
            joint_msg = args[1]
            self.img_callback(image_msg,joint_msg,0)
        elif len(args)==3:
            image_msg1 = args[1]
            joint_msg = args[2]
            self.img_callback(image_msg,joint_msg,0)
            self.img_callback(image_msg1,joint_msg,1)          

    def joint_msg2q(self, joint_msg):
        order = [2, 1, 0, 3, 4, 5]
        q = np.asarray([joint_msg.position[i] for i in order])
        q = np.around(q, decimals=4)
        q = q.reshape((6, 1))
        return q
    
    def get_corners(self,cv2_img, cam_id):
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(image=gray,dictionary=aruco_dict)  
        if ids is not None and len(ids) > 5: 
            _, Chcorners, Chids = aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board )  
            self.current_markers[cam_id] = len(Chcorners)  
            corners, ids, _ = aruco.detectMarkers(image=gray,dictionary=aruco_dict)  
            _, Chcorners, Chids = aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board )  
            QueryImg = aruco.drawDetectedCornersCharuco(gray, Chcorners, Chids,(0,0,255))
            # Display our image
            #cv2.imshow('QueryImage', QueryImg)
            print("{} markers detected".format(len(ids)))
            a = raw_input()    

def main():
    aruco_calib = chArucoCalib()
    rospy.init_node("aruco_calib", anonymous=True)
    rospy.spin()

if __name__ == "__main__":
    main()