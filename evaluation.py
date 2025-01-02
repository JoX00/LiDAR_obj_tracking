import json
import numpy as np
from scipy.spatial.distance import cdist

# Load ground truth for objcets inside 50m and detection data
with open("C:/Skola/Year5/AFU_research/LiDAR_Tracking/3d_ann_objects_within_50m.json") as f:
    ground_truth = json.load(f)

with open('C:/Skola/Year5/AFU_research/LiDAR_Tracking/3d_ann.json') as f:
    detections = json.load(f)

def calculate_hota_and_idf1(gt_data, det_data, thresholds):
    hota_scores = []
    idtp, idfp, idfn = 0, 0, 0
    previous_mapping = {}  # Tracks the UUID mappings from ground truth to detections in the previous frame

    for threshold in thresholds:
        tp, fp, fn, association_scores = 0, 0, 0, []
        
        for frame_idx in range(len(gt_data)):
            gt_cuboids = gt_data[frame_idx]["cuboids"]
            det_cuboids = det_data[frame_idx]["cuboids"]

            if not gt_cuboids or not det_cuboids:
                fn += len(gt_cuboids)
                fp += len(det_cuboids)
                idfn += len(gt_cuboids)
                idfp += len(det_cuboids)
                continue

            # Extract positions and UUIDs
            gt_positions = np.array([[c["position"]["x"], c["position"]["y"]] for c in gt_cuboids])
            det_positions = np.array([[c["position"]["x"], c["position"]["y"]] for c in det_cuboids])
            gt_uuids = [c["uuid"] for c in gt_cuboids]
            det_uuids = [c["uuid"] for c in det_cuboids]

            # Compute distance matrix
            dist_matrix = cdist(gt_positions, det_positions)

            # Match ground truth to detections
            current_mapping = {}  # Map ground truth UUIDs to detection UUIDs for this frame
            matched_gt = set()
            matched_det = set()
            for i, row in enumerate(dist_matrix):
                for j, dist in enumerate(row):
                    if dist <= threshold and i not in matched_gt and j not in matched_det:
                        tp += 1
                        matched_gt.add(i)
                        matched_det.add(j)
                        current_mapping[gt_uuids[i]] = det_uuids[j]

                        # Check association across frames
                        if frame_idx > 0 and gt_uuids[i] in previous_mapping:
                            if previous_mapping[gt_uuids[i]] == det_uuids[j]:
                                idtp += 1  # Identity correctly maintained across frames
                            else:
                                idfp += 1  # Identity mismatch
                        elif frame_idx > 0:
                            idfp += 1  #the uuid has matched with another uuid in the previous frame (false positive)

            # Count false negatives and false positives
            fn += len(gt_positions) - len(matched_gt)
            fp += len(det_positions) - len(matched_det)
            idfn += len(gt_uuids) - len(current_mapping)  # Ground truth objects not matched
            idfp += len(det_uuids) - len(current_mapping)  # Detected objects not matched

            # Update previous mapping for the next frame
            previous_mapping = current_mapping

        # Detection accuracy
        det_a = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0

        # Association accuracy
        ass_a = idtp / (idtp + idfp) if idtp + idfp > 0 else 0

        # HOTA score for this threshold
        hota_scores.append(np.sqrt(det_a * ass_a))

    # Average HOTA across all thresholds
    hota_score = np.mean(hota_scores)
    
    # IDF1 calculation
    idf1 = (2 * idtp) / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0

    return hota_score, idf1

# Define thresholds (e.g., distance in meters)
iou_thresholds = np.linspace(0.1, 1.0, 10)

# Calculate HOTA and IDF1
hota_score, idf1_score = calculate_hota_and_idf1(ground_truth, detections, iou_thresholds)
print(f"HOTA Score: {hota_score}")
print(f"IDF1 Score: {idf1_score}")