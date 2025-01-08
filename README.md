# LiDAR Point Cloud Object Tracking

## Overview

The purpose was to perform and explore methods of object tracking in extreme weather conditions for the ROADVIEW project, this was done by further developing the existing tracking algorithm created by Gayan, for example by using an extended Kalman filter instead of a regular Kalman filter. 

The object tracking has been performed on data from the CADC dataset. This project has not used the objects from Idriss’ object detection algorithm. Instead, the true boxes from the CADC dataset have been used, this data has an ID for objects in all 100 frames which has been used as a ground truth for evaluating the tracking quantitatively. Kalman Filtering and Extended Kalman filtering have been used for tracking. Results were visually assessed by plotting the object bounding boxes and quantitatively by calculating HOTA and IDF1 scores. 

### Input Format
The input to the tracking system is a list of boundary boxes, each defined by 7 parameters:
- **3 values** for the center of the box (`x`, `y`, `z`)
- **3 values** for the extent of the box (`x`, `y`, `z`)
- **1 value** for the yaw angle (rotation)

### Output
1. **Object Tracking**: The tracking system outputs a `3d_ann.json` file that contains the boundary boxes and their associated tracking IDs.
2. **Bird's Eye View (BEV) Visualization**: This tool generates a BEV video or GIF of the object tracking results using the LiDAR data.

## Key Features
### 1. Object Tracking Kalman Filter:
  File name: object_traking_modified.py
- **Tracking Algorithm**: An algorithm that track objects between frames of LiDAR data based on a weighted score that takes closest distance, velocity, and yaw into account. 
- **Kalman Filter**: A Kalman filter is used to predict the next position and then compliment that prediction with measurements, making the algorithm more robust.
- **Result**: HOTA score: 95.4, IDF1 score: 95.3. Result generated at 91fps with all true objects within 50m. Visual result in output_KF.gif 

### 2. Object Tracking, EKF:
  File name: obj_tracking_EKF.py
- **Tracking Algorithm**: Uses the same tracking algorithm as in object_traking_modified.py
- **Extended Kalman Filter**: Uses an extended Kalman Filter CTRV model to update positions.
- **Result**: HOTA score: 97.4, IDF1 score 97.3. Result generated at 89fps with all true objects within 50m. Visual result in output_EKF.gif 
  

### 3. Bird's Eye View Visualization:
  File name: cadc_devkit/run_demo_lidar_dev2.py
- **BEV Video/GIF**: Generates a birds-eye view representation of the object tracking results, allowing for visual inspection.

### 4. Quantitative Score:
  File name: evaluation.py
- **HOTA and IDF1 scores**: Gerneates HOTA and IDF1 scores to evaluate the tracking result from the algorithms.

## Installation

Install the required dependencies by running the following commands:

```bash
pip install -r requirements.txt
pip install json
pip install re
```

### Sample Out:
![GIF Visualization](output3.gif)
