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
###