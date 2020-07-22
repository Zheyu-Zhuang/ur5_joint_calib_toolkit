# Mutli-Camera - UR5 Robotic Arm Calibration Package

![optional caption text](Documents/robotic_vision_logo.jpg)

## Package Overview
This software package uses OpenCV and Python to implement calibration of a multi-camera system with a UR5 robot arm. Software for data collection (image and joint angles) is included, implemented with ROS and Python. The package is currently implemented to calibrate a UR5 robot arm with either a PointGrey or ZED camera, however the calibration is general and other robotic or camera platforms can be used if the image and robotic end-effector pose are provided.

## Prequisites
In order to run the software with a UR5 robot arm using either a Point Grey or ZED camera we used the following software packages:

* [OpenCV][cv]

ChAruco calibration boards can be generated [here][board].

## Run the program
To run the program first launch the ros nodes. Eg:
```{p}
roslaunch ur_modern_driver ur5_ros_control.launch robot_ip:=XXX.XXX.XXX.X

roslaunch zed_wrapper zed.launch

roslaunch pointgrey_camera_driver camera.launch
```
To initialise the calibration run:
```{p}
python aruco_calib.py
```
To move the UR5 using freedrive while it is communicating with a ros node run:
```{p}
python freeDrive.py
```

## File Structure
The software package is split into data collection (`collect_data.py`), and calibration (`chAruco_calib.py`) stages. The data collection process is specific to a UR5 robot, and requires that the `freeDive.py` file be run concurrently. The calibration process is general, and requires only that the path to the image data be provided at the beginning of the `chAruco_calib.py` file.

Example input and output files are provided in the `Examples` folder.

The CAD model for the calibration board is included as `XX`.


## Implementation
The code is designed to calibrate the setup shown below.

![optional caption text](Documents/ur5calib.png)
where {C_CV} refers to camera pose using the Computer Vision convention, {C_R} refers to camera pose using the Robotics convention, {M} refers to the ChAruco Marker board, {E} refers to end-effector pose, {B} refers to the pose of the UR5 base, and {T} refers to the pose of the table.

The code computes the poses the pose of the camera, table, and end-effector with respect to the base of the UR5. The pose of the table is defined as being the pose of the calibration board when the board is sitting flat on the table. Thus, only the axis orthogonal to the table will remain constant as the checkerboard moves around the plane of the table.

A more detailed overview of the calibration process is provided in `calibrationBackground.pdf`.

[cv]: https://opencv.org/ "OpenCV"
[ros]: http://wiki.ros.org/ur_kin_py "ur_kin_py"
[zed]: https://github.com/stereolabs/zed-ros-wrapper "zed"
[pg]: http://wiki.ros.org/pointgrey_camera_driver "pointgrey"
[board]: https://calib.io/pages/camera-calibration-pattern-generator "calib.io"

