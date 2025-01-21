import pickle
from filterpy.kalman import KalmanFilter
from EKF_modified import ExtendedKalmanFilter
#from filterpy.kalman import ExtendedKalmanFilter 
import numpy as np
import math
import json
import yaml
import uuid
import time
from collections import Counter

start_time = time.time()

class BoxTracker:
    def __init__(self, bbox_3d):
        #initialize Extended Kalman Filter for tracking with CTRV model
        self.ekf = ExtendedKalmanFilter(dim_x=5, dim_z=2)
        
        #state vector [x, y, v, theta, omega]
        self.ekf.x = np.array([bbox_3d[0], bbox_3d[1], 0, 0, 0])

        #measurement noise
        self.ekf.R = np.eye(2) * 0.000001

        #process noise
        self.ekf.Q = np.eye(5) * 0.00001

        #initial state covariance
        self.ekf.P *= 10

        #initialize threshold for turning
        self.epsilon = 0.1

    def state_transition(self, x, dt):
        #CTRV state transition model
        x_pos, y_pos, v, theta, omega = x
        if abs(omega) > self.epsilon:  #turning
            x_pos += (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            y_pos += (v / omega) * (np.cos(theta) - np.cos(theta + omega * dt))
            theta += omega * dt
        else:  #straight line (omega close to zero)
            x_pos += v * np.cos(theta) * dt
            y_pos += v * np.sin(theta) * dt
        return np.array([x_pos, y_pos, v, theta, omega])

    def jacobian_F(self, x, dt):
        #Jacobian of the state transition function for CTRV model
        x_pos, y_pos, v, theta, omega = x
        if abs(omega) > self.epsilon: #turning
            j13 = (np.sin(theta + omega * dt) - np.sin(theta)) / omega
            j14 = (v / omega) * (np.cos(theta + omega * dt) - np.cos(theta))
            j15 = (v * dt * np.cos(theta + omega * dt) / omega) - (v / (omega ** 2)) * (np.sin(theta + omega * dt) - np.sin(theta))
            j23 = (np.cos(theta) - np.cos(theta + omega * dt)) / omega
            j24 = (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
            j25 = (v * dt * np.sin(theta + omega * dt) / omega) - (v / (omega ** 2)) * (np.cos(theta) - np.cos(theta + omega * dt))
        else: #not turning
            j13 = np.cos(theta) * dt
            j14 = -v * np.sin(theta) * dt
            j15 = 0
            j23 = np.sin(theta) * dt
            j24 = v * np.cos(theta) * dt
            j25 = 0
        ret = np.array([
            [1, 0, j13, j14, j15],
            [0, 1, j23, j24, j25],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, dt],
            [0, 0, 0, 0, 1]
        ])
        return ret 

    def H_measurement(self, x):
        #measurement function, mapping state to measurement space (only x and y)
        return np.array([x[0], x[1]])

    def jacobian_H(self, x):
        #jacobian of the measurement function (constant since we directly observe x and y)
        return np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ])
    
    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def predict(self, dt=1.0):
        self.ekf.F = self.jacobian_F(self.ekf.x, dt)

        self.ekf.x = self.state_transition(self.ekf.x, dt)

        #this function has been modified, see more info in the ekf.predict function
        self.ekf.predict()

        self.ekf.x[3] = self.normalize_angle(self.ekf.x[3])

    def update(self, bbox_3d):
        self.ekf.update([bbox_3d[0], bbox_3d[1]], HJacobian=self.jacobian_H, Hx=self.H_measurement)
        self.ekf.x[3] = self.normalize_angle(self.ekf.x[3])

    def get_state(self):
        #return the full state [x, y, v, theta, omega]
        return self.ekf.x

    
class bb_traking:
    def __init__(self, config):
        #load configuration parameters for tracking
        self.lidar_frequency = config['lidar_frequency']
        self.threshold = config['threshold'] #distance threshold
        self.countdown = config['countdown']
        self.yaw_threshold = config['yaw_threshold']
        self.velocity_threshold = config['velocity_threshold']
        self.ignore_velocity = config['ignore_velocity']

    def create_rotation_matrix(self, yaw):
        #3x3 rotation matrix for yaw angle
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        return np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
    
    def calculate_distance(self, point1, point2):
        #Euclidean distance between two points (x, y)
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
    
    def calculate_velocity(self, P1, P2):
        #velocity based on previous and current positions
        t = 1 #/ self.lidar_frequency
        return (np.array(P2[:2]) - np.array(P1[:2])) / t
    
    def generate_traking_id(self, vehicle_ids, box, velocity):
        #assign unique tracking ID and UUID for new objects
        new_id = max(vehicle_ids.keys(), default=-1) + 1
        unique_uuid = str(uuid.uuid4())
        
        #new tracker for the detected box
        tracker = BoxTracker(box)
        vehicle_ids[new_id] = {
            'uuid': unique_uuid, 
            'position': [box[0], box[1], box[2]], 
            'prev_position': [box[0], box[1], box[2]],
            'mapped': 0, 
            'distance': float('inf'), 
            'box': box, 
            'tracker': tracker, 
            'countdown': self.countdown, 
            'velocity': velocity, 
            'new_velocity': True,
            'dist_threshold': 20
        }
        return vehicle_ids, new_id
    
    def create_bb_box(self, box, uuid, stationary, velocity):
        #prepare bounding box data for JSON output
        center = np.array(box[:3])
        extent = np.array(box[3:6])
        yaw = box[6]
        rotation = self.create_rotation_matrix(yaw)

        #return dict of bounding box data
        return {
            'uuid': uuid,
            'label': 'Car',
            'position': {'x': center[0], 'y': center[1], 'z': center[2]},
            'dimensions': {'x': extent[0], 'y': extent[1], 'z': extent[2]},
            'yaw': yaw,
            'stationary': stationary,
            'camera_used': None,
            'attributes': {'state': 'Parked'},
            'points_count': 0,
            'difficulty': 0,
            'velocity': velocity
        }
    
    def run_algorithm(self, frames):
        vehicle_ids = {}  #dictionary to keep track of active objects
        frame_results = []

        #iterate over frames
        for f, frame_boxes in enumerate(frames):
            #reset mapping count for all active objects
            for key in vehicle_ids:
                vehicle_ids[key]['mapped'] = 0

            box_results = []
            new_tracks = []
            #iterate over boxes within a frame
            for box in frame_boxes:
                center = np.array(box[:3])  #current box position
                best_match_key = -1
                existing_yaw = box[6]       #current box yaw
                best_match_distance = float('inf')
                best_match_tot_score = float('inf')
                #ignore tracking for the first frame, 
                #this will also ensure that new vehicle_ids are created for all objects in first frame
                if (f>0):
                    # iterate over all tracks to find the closest match for the current box
                    for key, track in vehicle_ids.items():
                        distance_diff = self.calculate_distance(center, track['position'])
                        velocity_diff = np.linalg.norm(np.array(track['velocity']) - self.calculate_velocity(track['prev_position'], center))
                        detected_yaw = track['box'][6]  
                        yaw_diff = abs(math.atan2(math.sin(detected_yaw - existing_yaw), math.cos(detected_yaw - existing_yaw)))
                        
                        #set scores   
                        dist_score = distance_diff
                        vel_score = velocity_diff
                        yaw_score = yaw_diff
                        w_d, w_v, w_y = 2, 1, 10 #adjust weights here

                        #lower score is better
                        tot_score = w_d*dist_score + w_v*vel_score + w_y*yaw_score 

                        #save best match if score is highest and thresholds are ok for distance, velocity, and yaw
                        if (tot_score < best_match_tot_score and distance_diff < track['dist_threshold'] and 
                            velocity_diff < self.velocity_threshold and yaw_diff < self.yaw_threshold):
                            best_match_distance = distance_diff
                            best_match_key = key
                            best_match_tot_score = tot_score
                
                #if no match found, append the box so a new track is created outside the objects loop
                if best_match_key == -1:
                    new_tracks.append(box)

                #found a match
                else:
                    #update the existing tracked object
                    vehicle_ids[best_match_key]['mapped'] += 1
                    vehicle_ids[best_match_key]['box'] = box
                    previous_position = vehicle_ids[best_match_key]['position']
                    tracker = vehicle_ids[best_match_key]['tracker']

                    #update and predict with Kalman filter
                    tracker.update(box)
                    tracker.predict()

                    #update position and calculate new velocity based on change in position
                    predicted_position = tracker.get_state().tolist()
                    new_velocity = self.calculate_velocity(previous_position, predicted_position)

                    #save dist_threshold for each track based on difference between current and predicted position
                    #the center of the cars should not be closer than 2m, therefore we set 2 as the minimum threshold
                    #also, cars that are moving the same speed as us in the same direction will have a very small 
                    #difference between current and predicted position, which would make dist_thresh very small 
                    #and the matching would be prone to measurement errors, thats also why we set a minimumum threshold to 2
                    dist_thresh = self.calculate_distance(previous_position, predicted_position)
                    #these threshold are chosen based on the fact that still cars has dist_thresh ~3.5, meeting cars
                    #has dist_thresh < 1.5, same direction moving cars has dist_thres ~7.5
                    if dist_thresh < 2:
                        vehicle_ids[best_match_key]['dist_threshold'] = 4
                    elif dist_thresh < 5:
                        vehicle_ids[best_match_key]['dist_threshold'] = 2*dist_thresh
                    else: dist_thresh = 20

                    #update the prev_position of track before updating 'position' with predicted position
                    vehicle_ids[best_match_key]['prev_position'] = vehicle_ids[best_match_key]['position']

                    #update tracking data with new position and velocity
                    vehicle_ids[best_match_key]['position'] = predicted_position
                    vehicle_ids[best_match_key]['velocity'] = new_velocity
                    new_vel_scalar = np.linalg.norm(np.array(new_velocity)) 
                    if 1 < (new_vel_scalar) < 4:
                        stationary = True
                    else:
                        stationary = False

                    #create bounding box result
                    bb_res = self.create_bb_box(box, vehicle_ids[best_match_key]['uuid'], stationary, new_vel_scalar)
                    box_results.append(bb_res)
                    
            #count down and remove untracked objects
            if f > 0:
                for key in list(vehicle_ids.keys()):
                    if vehicle_ids[key]['mapped'] == 0:
                        vehicle_ids[key]['countdown'] -= 1
                        if vehicle_ids[key]['countdown'] <= 0:
                            del vehicle_ids[key]
            
            #create new tracks for all the new objects, it is done after the object loop because 
            #if this is done when looping through objects, 
            #two new objects that are close would match with eachothers tracks
            for track in new_tracks:
                initial_velocity = [0, 0]  #starting velocity for a new object
                vehicle_ids, new_id = self.generate_traking_id(vehicle_ids, track, initial_velocity)
                stationary = True
                bb_res = self.create_bb_box(track, vehicle_ids[new_id]['uuid'], stationary, velocity = 0)
                box_results.append(bb_res)

            #append results for current frame
            frame_results.append({'cuboids': box_results})

        return frame_results



def load_bounding_boxes(file_path):
    # Load bounding box data from a pickle file
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def convert_to_python_types(data):
    # Convert data to Python types for JSON serialization
    if isinstance(data, dict):
        return {key: convert_to_python_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_python_types(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    else:
        return data
    
def print_fps(start_time, end_time):
    elapsed_time = end_time - start_time  #calculate elapsed time
    fps = 100/elapsed_time
    print(f'fps = {fps}')

def get_true_boxes(num_frames):
    #path for all true boxes
    #path = 'C:/Skola/Year5/AFU_research/LiDAR_Tracking/cadc_devkit/dataset/dataset/cadcd/2019_02_27/0031/3d_ann.json'
    #path for true boxes inside 50m
    path = 'C:/Skola/Year5/AFU_research/LiDAR_Tracking/3d_ann_objects_within_50m.json'
    with open(path) as f:
        annotations_data = json.load(f)
    
    frame_list = []
    uuid_list = []

    for frame in annotations_data[:num_frames]: #change here for loading different amount of frames
        obj_list = []
        for obj in frame['cuboids']:
            #create a row with position and dimensions
            one_obj_list = [
                obj['position']['x'], 
                obj['position']['y'], 
                obj['position']['z'],
                obj['dimensions']['x'],  # length
                obj['dimensions']['y'],  # width
                obj['dimensions']['z'],   # height
                obj['yaw']
            ]
            obj_list.append(one_obj_list)
            uuid_list.append(obj['uuid'])
        frame_list.append(obj_list)
    return(frame_list)

def calculate_distance(self, point1, point2):
        #Euclidean distance between two points (x, y)
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def add_difficulty_score(frame_results):
    """
    args: frame_results is the class with cuboids for all frames

    if you want to print velocity only, set occupancy score to 0 in if statments, and vice versa

    returns: the same class but with the difficulty score added for each object
    """
    #difficulty score is based on velocity and occupancy
    vel_score = []
    occ_score = []
    for frame in frame_results:
        for box in frame['cuboids']:
            occupancy_score = 0
            #for loop for calculating occupancy score
            for neighbour_box in frame['cuboids']:
                if box['uuid'] != neighbour_box['uuid']:
                    box_pos = [box['position']['x'], box['position']['y']]
                    neighbour_box_pos = [neighbour_box['position']['x'], neighbour_box['position']['y']]
                    distance = math.sqrt((box_pos[0] - neighbour_box_pos[0]) ** 2 + (box_pos[1] - neighbour_box_pos[1]) ** 2)
                    
                    #to avoid to large number of ocupancy score
                    if distance < 1:
                        distance = 1
                    #the smaller the distance between the object and neighbour object is, add more to occupancy score
                    occupancy_score += (2000/distance**3)
            
            #velocity score
            velocity_score = box['velocity']

            if velocity_score == 0:
                box['difficulty'] = int(occupancy_score)
            else: box['difficulty'] = int(velocity_score + occupancy_score)

    return frame_results

###### Main script to run tracking visualization

# #load predicted boxe from Idriss
#boxes = load_bounding_boxes("box_list.pkl")

#load true boxes from dataset, specify number of frames here, specify in get_true_boxes if you want all boxes or boxes inside 50m
boxes = get_true_boxes(num_frames=100) 

start_time = time.time()
config = yaml.safe_load(open("settings.yaml", 'r'))
tracker = bb_traking(config)
frame_results = tracker.run_algorithm(boxes)
add_difficulty_score(frame_results)

# Convert results and save as JSON
converted_results = convert_to_python_types(frame_results)
with open('3d_ann.json', 'w') as json_file:
    json.dump(converted_results, json_file, indent=4)

end_time = time.time()  #capture end time
print_fps(start_time, end_time)
