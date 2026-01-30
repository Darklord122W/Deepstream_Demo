# Deepstream Demo
The final ideal outcome of this project is to use a Nvidia AGX Orin to run a docker container, which will subscribe to a ROS2 image topics and then run through deepstream accerlated inferenced perception pipeline wrapped in ROS2 node then output the bounding box information.

### PARALLEL MISSIONS FOR AT THE SAME TIME
- [x] Read [RPN](https://arxiv.org/pdf/1506.01497)
- [ ] go through the Docker tutorials
- [ ] Read through all the Documentation files of DeepStream,especially the python examples carefully.

## Current Progress
more details in progress:[Progress stage](Progress.md)

Trying to do 
```
PGIE (detector)
   ├── SGIE-1 (classifier A)
   └── SGIE-2 (classifier B)
```
At the moment, there are mismatch of the output annotated image bounding box, also the second classifier has no actual input. 

**Firstly**, I need to be more familar with the code structure, It can definitely be breaken down into more than one python file and use import to work together in ROS2. 

**Secondly**, I saw sample code in Deepstream ROS2 official GitHub code. one of it uses detection model then applies classification model. Definitely have to take a look.

## Hardware components
### cameras info
#### Logitech C310
It can do YUYV and MJPG two differnent ways:
MJPG @ 1280×720 @ 30 FPS
YUYV @ 640×480 @ 30 FPS
MJPG format runs faster

Existing ROS2 node: v4l2_camera usb_cam

## Camera configuration
### usb_cam format parameter
```
This driver supports the following formats:
	rgb8
	yuyv
	yuyv2rgb
	uyvy
	uyvy2rgb
	mono8
	mono16
	y102mono8
	raw_mjpeg
	mjpeg2rgb
	m4202rgb
```
### File location
```
cd /home/autodrive/pythonDeep/deepstream_python_apps/apps/deepstream-test1-usbcam

```
### how to check camera image output format
Check what format camera can output:
```
v4l2-ctl --list-formats-ext -d /dev/video0
```
Check ROS2 topic
```
ros2 topic echo /image_raw --once
```
check topic frequency example
```
ros2 topic hz /image_raw
```


## Command I use frequently
check dir structure
```
tree -L 2
```
for my ROS2 nodes running
```
ros2 run usb_cam usb_cam_node_exe
```

for see ROS2 node connection graph
```
rqt_graph
```
## Useful project link
[Yolo model integration](https://github.com/marcoslucianops/DeepStream-Yolo)

[Official Doc for 7.1](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_Overview.html)

[Deepstream python example](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/tree/v1.2.0)

[Deepstream ROS2 example](https://github.com/NVIDIA-AI-IOT/ros2_deepstream/tree/main)


## Terminology 
**GStreamer**:GStreamer is a library for constructing graphs of media-handling components. The applications it supports range from simple Ogg/Vorbis playback, audio/video streaming to complex audio (mixing) and video (non-linear editing) processing.

**nvstreammux**: It takes multiple input video streams (or even just one), synchronizes them, optionally scales/converts them to a common size, and outputs one batched buffer that downstream GPU plugins (especially nvinfer) can process efficiently.

**nvinfer**:nvinfer runs TensorRT-optimized deep-learning inference on batched GPU frames and attaches results as metadata — it does NOT draw, publish, or display anything by itself.
```
It does:
1. Pre-processing
2. TensorRT inference
3. Post-processing
4. Attach metadata
```

From upstream (almost always nvstreammux):
GPU buffers (NVMM memory)
Batched frames
Consistent resolution / format
Frame metadata (source_id, timestamps)