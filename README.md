# Deepstream Demo
The final indeal outcome of this project is to use a Nvidia AGX Orin to run a docker container, which will subscribe to a ROS2 image topics and then run through deepstream accerlated inferenced wrapped in ROS2 node then output the bounding box information.
## Current plan
### stage1: 
I need to firstly set up a camera node then output custom image messages. I need to be familiar with how camera image output format. Be familiar with how camera image messages in ROS2 Structure flow.


### stage2:
Need to prepare two models (Grab it from hugging face or somewhere). First model will recognize it is a traffic sign, second model will identify the specific traffic sign. learn about Hugging face.

### stage3:
try to make a python pipeline using Deepstream. Pay attention to how to integrate deepstream into python pipeline, especially how to solve the environment dependencies.

### stage4:
After it works well without ROS2. I need to firstly set up a perception pipeline without deepstream, just in ROS2, which can take in images and then publish bounding box after going through a object detection model like yolov as Andrei asked. then do comparation for the same model just running in python. 

### final stage:
After I familiar with all of the process above, I need to integrate deepstream pipeline in ROS2 first. Then, furthermore think about docker.

### MISSIONS FOR AT THE SAME TIME
- [ ] Read [RPN](https://arxiv.org/pdf/1506.01497)
- [ ] go through the Docker tutorials
- [ ] Read through all the Documentation files of DeepStream,especially the python examples carefully.
- [ ] 

## Current Progress

## Hardware components
### cameras info
#### Logitech C310
It can do YUYV and MJPG two differnent ways:
MJPG @ 1280×720 @ 30 FPS
YUYV @ 640×480 @ 30 FPS
MJPG format runs faster

Existing ROS2 node: v4l2_camera usb_cam

## Camera images format
### how to check camera image output format
Check what format camera can output:
```
v4l2-ctl --list-formats-ext -d /dev/video0
```
Check ROS2 topic
```
ros2 topic echo /image_raw --once
```
### yolov11 expected


## Command I use
check dir structure
```
tree -L 2
```

## Useful project link
[Yolo model integration](https://github.com/marcoslucianops/DeepStream-Yolo)

[Official Doc for 7.1](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Overview.html)

[Deepstream python example](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)

