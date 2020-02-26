#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String


def set_freedrive_mode(enable_teach_mode=False):
    freedrive_publisher = rospy.Publisher('/ur_driver/URScript', String, queue_size=10)
    rate = rospy.Rate(10000) 
    print("enabled")
    while not rospy.is_shutdown():
        if enable_teach_mode:
            # send URScript command that will enable teach mode
            freedrive_publisher.publish('def myProg():\n\twhile (True):\n\t\tfreedrive_mode()\n\t\tsync()\n\tend\nend\n')
            freedrive_publisher.publish("def myProg():\n\tfreedrive_mode()\nsleep({})\nend")
        else:
            print("disabled")
            # send URScript command that will disable teach mode
            freedrive_publisher.publish('def myProg():\n\twhile (True):\n\t\tend_freedrive_mode()\n\t\tsleep(0.5)\n\tend\nend\n')
        rate.sleep()

def main():
    set_freedrive_mode(True)
    #rospy.spin()

if __name__ == "__main__":
    rospy.init_node("test", anonymous=True)
    main()
