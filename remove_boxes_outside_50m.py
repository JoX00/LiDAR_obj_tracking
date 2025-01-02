import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import matplotlib.patches as patches
import cv2
import os
import re
from PIL import Image
import random

annotations_path = "C:/Skola/Year5/AFU_research/LiDAR_Tracking/cadc_devkit/dataset/dataset/cadcd/2019_02_27/0031/3d_ann.json"

def filter_objects_within_50m_xy(data, max_distance=50.0):
    """
    Removes objects outside the specified distance in either x or y directions from the ego vehicle.

    Parameters:
    - data (dict): The JSON data containing the objects.
    - max_distance (float): The maximum allowed distance in meters for x or y.

    Returns:
    - dict: The filtered data with objects within the specified distance.
    """
    for frame in data:
        filtered_cuboids = []
        for cuboid in frame['cuboids']:
            position = cuboid['position']
            if abs(position['x']) <= max_distance and abs(position['y']) <= max_distance:
                filtered_cuboids.append(cuboid)
        frame['cuboids'] = filtered_cuboids
    return data

#example usage
with open(annotations_path, "r") as file:
    data = json.load(file)

filtered_data = filter_objects_within_50m_xy(data, max_distance=50.0)

#save the filtered data in another json
with open("3d_ann_objects_within_50m.json", "w") as file:
    json.dump(filtered_data, file, indent=4)