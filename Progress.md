## Current plan
### stage1✅: 
I need to firstly set up a camera node then output custom image messages. I need to be familiar with how camera image output format. Be familiar with how camera image messages in ROS2 Structure flow.

### stage2✅:
try to make a python pipeline using Deepstream. Pay attention to how to integrate deepstream into python pipeline, especially how to solve the environment dependencies.

### stage3✅:
After it works well without ROS2. I need to firstly set up a perception pipeline without deepstream, just in ROS2, which can take in images and then publish bounding box after going through a object detection model like yolov. then do integrations for Deepstream.

### stage4: 
prepare two models. One is detection yolov11n and classify model YOLO26n-cls for a first coarse detection then do detailed classification task. Got model✅. Pipeline is still under progress.

### stage5: 
multi-cameras inputs and do inferences then output multiple topics for different camera streams. 
### final stage:
After I familiar with all of the process above, I need to integrate deepstream pipeline in ROS2 first. Then, furthermore think about docker.
