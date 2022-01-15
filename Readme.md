# 7-Scenes Color/Depth Calibration

This repository contains code to estimate the calibration of the Kinect camera used in the [7-Scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) dataset.


## RGB intrinsics
The intrinsics parameters of the RGB camera are estimated from vanishing points detected using rectangles in the scene (see [calibrate_rgb_camera_from_rectangle.py](calibrate_rgb_camera_from_rectangle.py))


<img src="doc/K_rgb.png " width="324" height="106">

- fx = 532.57
- fy = 531.54
- cx = 320
- cy = 240



<img src="doc/rgb_calibration.png " width="530" height="400">


## Depth intrinsics and extrinsics
The intrinsics of the depth camera and extrinsics parameters to align depth and rgb are estimated using points correspondences (see [calibrate_depth.py](calibrate_depth.py)).

The two cameras are assumed to have the same orientation. Only a translation is estimated.


<img src="doc/K_depth.png " width="331" height="99">

- fx = 598.84 
- fy = 587.62
- cx = 320
- cy = 240



<img src="doc/T_depth_rgb.png " width="344" height="99">

- tx = 0.023449
- ty = 0.006177
- tz = 0.010525


The calibration can be checked by on the colored point cloud generated in [check_calibration.py](check_calibration.py)

<img src="doc/colored_point_cloud.png " width="887" height="551">


