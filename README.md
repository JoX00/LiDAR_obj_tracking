# LiDAR Point Cloud Object Tracking

## Overview

This project focuses on tracking objects in LiDAR point cloud data using boundary boxes. The tracking is performed by measuring distances between frames, predicting future positions with a Kalman filter, and comparing velocities to improve tracking accuracy. Additionally, it provides a visualization tool to generate a bird's eye view (BEV) of the tracked objects as a video or GIF.

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
