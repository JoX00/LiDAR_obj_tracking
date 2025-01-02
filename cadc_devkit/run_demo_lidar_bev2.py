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

# Parameters
frame = 12
cam = '0'
seq = '0031'
DISTORTED = False
DISPLAY_LIDAR = False
DISPLAY_CUBOID_CENTER = False
MIN_CUBOID_DIST = 40.0
BASE = 'C:/Skola/Year5/AFU_research/LiDAR_Tracking/cadc_devkit/dataset/dataset/cadcd/2019_02_27/'

if DISTORTED:
    path_type = 'raw'
else:
    path_type = 'labeled'

#annotations_path = "C:/Skola/Year5/AFU_research/LiDAR_Tracking/cadc_devkit/dataset/dataset/cadcd/2019_02_27/0031/3d_ann.json"
annotations_path = "C:/Skola\Year5/AFU_research/LiDAR_Tracking/3d_ann.json"
image_folder = 'C:/Skola/Year5/AFU_research/LiDAR_Tracking/cadc_devkit/images/3'
output_gif_file = 'output3.gif'
duration = 150

# Dictionary to store last 40 absolute positions for each cuboid
cuboid_history = {}
color_map = {}

def generate_random_color():
    """Generate a random color as an RGB tuple."""
    while True:
        r, g, b = random.random(), random.random(), random.random()
        if r + g + b > 1.5:  # Check if the sum of the components is strong enough
            return (r, g, b)

def bev(s1, s2, f1, f2, frame, lidar_path, annotations_path):
    # Limit the viewing range
    side_range = [-s1, s2]
    fwd_range = [-f1, f2]

    # Load LiDAR data
    scan_data = np.fromfile(lidar_path, dtype=np.float32)
    lidar = scan_data.reshape((-1, 4))
    lidar_x, lidar_y, lidar_z = lidar[:, 0], lidar[:, 1], lidar[:, 2]

    # Truncate LiDAR data based on viewing range
    lidar_x_trunc = []
    lidar_y_trunc = []
    lidar_z_trunc = []
    for i in range(len(lidar_x)):
        if fwd_range[0] < lidar_x[i] < fwd_range[1] and side_range[0] < lidar_y[i] < side_range[1]:
            lidar_x_trunc.append(lidar_x[i])
            lidar_y_trunc.append(lidar_y[i])
            lidar_z_trunc.append(lidar_z[i])

    # Convert LiDAR data to image coordinates
    x_img = [-y for y in lidar_y_trunc]
    y_img = lidar_x_trunc
    pixel_values = lidar_z_trunc

    x_img = [x - side_range[0] for x in x_img]
    y_img = [y - fwd_range[0] for y in y_img]

    # Load 3D annotations
    with open(annotations_path) as f:
        annotations_data = json.load(f)

    # Plot setup
    cmap = "jet"
    dpi = 100
    x_max = side_range[1] - side_range[0]
    y_max = fwd_range[1] - fwd_range[0]
    fig, ax = plt.subplots(figsize=(2000 / dpi, 2000 / dpi), dpi=dpi)

    # Track each cuboid
    for cuboid in annotations_data[frame]['cuboids']:
        #if cuboid['uuid'] == '17ba5dad-2e63-4c10-bc6a-8f65dfb0a6a3': #choose if you want to plot specific uuid
            uuid = cuboid['uuid']
            position = (cuboid['position']['x'], cuboid['position']['y'])
            
            # Assign a unique color if the uuid is new, otherwise use the existing color
            if uuid not in color_map:
                color_map[uuid] = generate_random_color()
            color = color_map[uuid]

            # Initialize history if not yet initialized for this uuid
            if uuid not in cuboid_history:
                cuboid_history[uuid] = []

            # Update the position history (only keep the last 40 absolute positions)
            cuboid_history[uuid].append(position)
            if len(cuboid_history[uuid]) > 12:
                cuboid_history[uuid].pop(0)

            # Plot last 40 absolute positions without any adjustments
            if len(cuboid_history[uuid]) > 1:
                for i in range(1, len(cuboid_history[uuid])):
                    # Get the current and previous position
                    pos = cuboid_history[uuid][i]
                    prev_pos = cuboid_history[uuid][i - 1]
            
                    # Convert both positions to image coordinates
                    x_hist, y_hist = -pos[1] - side_range[0], pos[0] - fwd_range[0]
                    x_prev_hist, y_prev_hist = -prev_pos[1] - side_range[0], prev_pos[0] - fwd_range[0]

                    #if the object is outside the plot, dont plot the past positions 
                    if (side_range[0] <= position[1] <= side_range[1] and fwd_range[0] <= position[0] <= fwd_range[1] 
                    and cuboid['stationary'] == 8):
                        
                        # Plot a dot for the current position
                        ax.plot(x_hist, y_hist, 'o', color=color, markersize=2)  # Plot history dots

                        # Plot a line between the previous and current position
                        ax.plot([x_prev_hist, x_hist], [y_prev_hist, y_hist], color=color, linewidth=2.5)  # Plot connecting lines

            # Project cuboid to image coordinates
            T_Lidar_Cuboid = np.eye(4)
            T_Lidar_Cuboid[0:3, 0:3] = R.from_euler('z', cuboid['yaw'], degrees=False).as_matrix()
            T_Lidar_Cuboid[0, 3] = position[0]
            T_Lidar_Cuboid[1, 3] = position[1]

            width, length, height = cuboid['dimensions']['x'], cuboid['dimensions']['y'], cuboid['dimensions']['z']

            # Define cuboid corners in local frame
            corners = [
                [length / 2, width / 2, height / 2],
                [length / 2, -width / 2, height / 2],
                [-length / 2, width / 2, height / 2],
                [-length / 2, -width / 2, height / 2]
            ]
            x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = [], [], [], [], [], [], [], []

            # Transform cuboid corners to world frame and plot
            for corner in corners:
                corner_3d = np.array([*corner, 1])
                transformed_corner = np.matmul(T_Lidar_Cuboid, corner_3d)
                x = -transformed_corner[1] - side_range[0]
                y = transformed_corner[0] - fwd_range[0]
                if corner == corners[0]: x_1, y_1 = x, y
                elif corner == corners[1]: x_2, y_2 = x, y
                elif corner == corners[2]: x_3, y_3 = x, y
                elif corner == corners[3]: x_4, y_4 = x, y

            poly = np.array([[x_1, y_1], [x_2, y_2], [x_4, y_4], [x_3, y_3]])
            ax.add_patch(patches.Polygon(poly, closed=True, fill=False, edgecolor=color, linewidth=1))
            #Check if the object is within the side and forward range
            if side_range[0] <= position[1] <= side_range[1] and fwd_range[0] <= position[0] <= fwd_range[1]:
                # Inside the loop where you process each cuboid
                difficulty = cuboid['difficulty']  # Get the difficulty
                if difficulty >= 15:
                    text = 'hard'
                elif 7 <= difficulty <= 15:
                    text = 'medium'
                else: text = 'easy'
                # Ensure 'difficulty' can be converted to a float
                try:
                    formatted_difficulty = f'{float(difficulty):.1f}'  # Format to 1 decimal
                except (ValueError, TypeError):
                    formatted_difficulty = 'N/A'  # Fallback if difficulty is not a number

                # Calculate the position for the top-right corner of the bounding box
                x_top_right = max(x_1, x_2, x_3, x_4)
                y_top_right = max(y_1, y_2, y_3, y_4)

                # Offset to position the text slightly inside the bounding box
                offset = 2  # Adjust as needed
                x_text = x_top_right - offset
                y_text = y_top_right - offset

                # Add difficulty text to the plot
                ax.text(
                    x_text,
                    y_text,
                    text,
                    color=color,  # Use the same color as the bounding box for clarity
                    fontsize=8,
                    ha='right',  # Align the text to the right
                    va='top',    # Align the text to the top
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor=color)  # Optional background for text
                )

    # Plot LiDAR points
    ax.scatter(x_img, y_img, s=1, c=pixel_values, alpha=1.0, cmap=cmap)
    ax.set_facecolor((0, 0, 0))
    ax.axis('scaled')
    plt.xlim([0, x_max])
    plt.ylim([0, y_max])
    fig.savefig(os.path.join(image_folder, f"bev_{frame:03d}.png"), dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)

# Helper functions for GIF creation
def natural_sort_key(string):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', string)]

def make_gif_from_images(image_folder, output_gif_file, duration=100):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=natural_sort_key)
    frames = [Image.open(os.path.join(image_folder, img)) for img in images]
    frames[0].save(output_gif_file, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)

# Main execution
for frame in range(0, 100): #change here for plot of different amount of frames
    lidar_path = os.path.join(BASE, seq, path_type, "lidar_points", "data", f"{frame:010}.bin")
    bev(50, 50, 50, 50, frame, lidar_path, annotations_path)

make_gif_from_images(image_folder, output_gif_file, duration)
