import cv2
import torch
import torchvision
import mmcv
import mmengine
import matplotlib.pyplot as plt
import numpy as np


# Function definitions
def calculate_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0


def calculate_iou_box(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of both areas minus the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def Image_depth(image : np.array, depth_model, intrinsic):
  h_original, w_original = image.shape[:2]

  # Scale image
  input_size = (616, 1064) # for vit model
  scale = min(input_size[0] / h_original, input_size[1] / w_original)
  rgb = cv2.resize(image, (int(w_original * scale), int(h_original * scale)), interpolation=cv2.INTER_LINEAR)
  new_intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]

  # Padding
  padding = [123.675, 116.28, 103.53]
  h, w = rgb.shape[:2]
  pad_h = input_size[0] - h
  pad_w = input_size[1] - w
  pad_h_half = pad_h // 2
  pad_w_half = pad_w // 2
  rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=padding)
  pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

  # Normalize
  mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
  std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
  rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
  rgb = torch.div((rgb - mean), std)

  # Predict
  rgb = rgb[None, :, :, :].cuda()
  pred_depth, confidence, output_dict = depth_model.inference({'input': rgb})
  pred_depth = pred_depth.squeeze()
  pred_depth = pred_depth[pad_info[0] : pred_depth.shape[0] - pad_info[1], pad_info[2] : pred_depth.shape[1] - pad_info[3]]

  # upsample to original size
  pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], (h_original,w_original), mode='bilinear').squeeze()

  # de-canonical transform
  canonical_to_real_scale = new_intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
  pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
  pred_depth = torch.clamp(pred_depth, 0, 300)

  # Transform to np.array
  depth_map = pred_depth.squeeze().cpu().numpy()

  return depth_map


def Track_image_sequence(image_sequence: np.ndarray, YOLO_model, tracker, sequence_length=0, iou_criterion=0.9):
    # Initialize list to hold detections and tracking results
    box_list_sequence = [] # Output, array of frame_boxes

    # Determine the sequence length
    num_frames = sequence_length if sequence_length > 0 else len(image_sequence)

    # Reassign tracking IDs sequentially
    track_id_mapping = {}
    next_track_id = 1

    for i in range(num_frames):
        img = image_sequence[i]

        # Perform object detection using YOLO model
        results = YOLO_model(img, verbose=False)
        detections = results[0].boxes  # Extract bounding boxes from YOLO

        # List to hold detection boxes for the current frame
        boxes = []
        for det in detections:
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            conf = det.conf.cpu().item()
            cls = int(det.cls.cpu().item())
            w = x2 - x1
            h = y2 - y1
            boxes.append([[x1, y1, w, h], conf, cls])

        # Initialize for frame 0
        if i == 0:
            tracker.update_tracks(boxes, frame=img)

        # Update tracker only if there are detections in the frame
        frame_boxes = [] # bboxes for each frame
        if boxes:
            track_objects = tracker.update_tracks(boxes, frame=img)

            # Store confirmed tracks with box data in the specified format
            for track in track_objects:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                original_track_id = track.track_id
                # Assign Seqeuntial ID
                if original_track_id not in track_id_mapping:
                    track_id_mapping[original_track_id] = next_track_id
                    next_track_id += 1
                track_id = track_id_mapping[original_track_id]

                bbox = track.to_ltrb()
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

                cls = track.det_class

                duplicate_flag = False
                for prev_box in frame_boxes:
                    _, _, prev_x1, prev_y1, prev_x2, prev_y2 = prev_box
                    iou = calculate_iou_box([prev_x1, prev_y1, prev_x2, prev_y2], [x1, y1, x2, y2])
                    if iou > iou_criterion:
                        duplicate_flag = True
                        break
                if duplicate_flag:
                    continue

                # Append [Class_ID, Instance_ID, left_x, up_y, right_x, down_y]
                if cls in [1, 2, 3, 5, 7]: # Assume as car if COCO label is 'car' or 'motorcycle' or 'bus' or 'truck'
                    frame_boxes.append([1, track_id, x1, y1, x2, y2])
                elif cls in [0]: # Assume as pedestiran if COCO label is 'person'
                    frame_boxes.append([2, track_id, x1, y1, x2, y2])
                else:
                    frame_boxes.append([0, track_id, x1, y1, x2, y2])

        # Append detections for the current frame to the sequence list
        box_list_sequence.append(frame_boxes)

    return box_list_sequence


def Construct_initial_guess(image: np.array, box_list, depth_map, C2bboxID_seq, tracked_objects=10):
    """
    Constructs an initial guess for model input using image, depth map, and bounding box information.

    Args:
        image (np.array): Input RGB image of shape (h, w, 3).
        box_list (list): List of bounding boxes in the format
                         [Class_ID, Instance_ID, left_x, up_y, right_x, down_y].
        depth_map (np.array): 2D array of depth values (h, w).
        C2bboxID_seq (list): List of tracked object mappings over time.
        tracked_objects (int): Maximum number of objects to track.

    Returns:
        np.array: Initial guess with added channels for tracking with Channel 0 : gray image / 1 : depth map / 2 : background mask / 3 ~ : object mask
    """
    h, w = depth_map.shape

    # Initialize tracking state
    if not C2bboxID_seq:
        current_C2bboxID = {}
        for i in range(1, tracked_objects+1): current_C2bboxID[i] = None
    else:
        current_C2bboxID = C2bboxID_seq[-1]
    new_C2bboxID = {i: current_C2bboxID[i] for i in range(1, tracked_objects+1)} # Assign for channel 1 ~ tracked_objects

    # Expand depth map and image
    depth_map_expanded = depth_map[:, :, np.newaxis]
    grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    initial_guess = np.concatenate((grayscale_image[:, :, np.newaxis], depth_map_expanded), axis=2)
    initial_guess = np.concatenate((initial_guess, np.zeros((h, w, tracked_objects+1))), axis=2) # Background channel + traked_objects channel

    # Compute instance depth and bounding box sanitization
    instance_depth = {}
    for class_id, instance_id, left_x, up_y, right_x, down_y in box_list:
        left_x, right_x = max(0, left_x), min(w - 1, right_x)
        up_y, down_y = max(0, up_y), min(h - 1, down_y)
        avg_depth = np.mean(depth_map[up_y:down_y+1, left_x:right_x+1])
        instance_depth[instance_id] = (avg_depth, class_id, left_x, up_y, right_x, down_y) # Depth and other information for each instance id

    # Sort instances by depth and assign IDs
    sorted_keys = sorted(instance_depth, key=lambda k: instance_depth[k][0]) # Sort instance id based on depth
    for i in range(1, tracked_objects+1): # Run for each channel
        # Check if ID in current_C2bboxID is not tracked
        if current_C2bboxID[i] is not None:
            instance_id = current_C2bboxID[i][0] # preserve channel information if tracked
            if instance_id not in instance_depth:
                new_C2bboxID[i] = None # instance id is not tracked
        # Assign instance_ids to channels with depth priority
        if new_C2bboxID[i] is None:
            for instance_id in sorted_keys:
                if instance_id not in [value[0] for value in new_C2bboxID.values() if value is not None]:
                    new_C2bboxID[i] = (instance_id, instance_depth[instance_id][1]) # Add (intance_id, class_id)
                    break
    C2bboxID_seq.append(new_C2bboxID)

    # Create masks for tracked instances
    comp_mask = np.zeros((h, w))
    background_mask = np.ones((h, w))
    for instance_id in sorted_keys:
        _, class_id, left_x, up_y, right_x, down_y = instance_depth[instance_id]
        instance_mask = np.zeros((h, w))
        instance_mask[up_y:down_y+1, left_x:right_x+1] = 1
        instance_mask = np.maximum(instance_mask - comp_mask, 0)
        comp_mask = np.logical_or(comp_mask, instance_mask)

        if (instance_id, class_id) in new_C2bboxID.values():
            C_id = list(new_C2bboxID.keys())[list(new_C2bboxID.values()).index((instance_id, class_id))] # Channel number for instance id
            initial_guess[:, :, 2 + C_id] = instance_mask
            background_mask = background_mask - instance_mask
    initial_guess[:, :, 2] = np.maximum(background_mask, 0)

    return initial_guess


def GT2DetectID(Initial_guess_seq, Instance_gt_list, iou_threshold = 0.2):
    """
    Outputs modified instance gt

    Args:
        Initial_guess_seq (np.array): [Initial guess np.array, ... for each frame].
        Instance_gt_list (list): [{instance_id: Segmentation mask}, ... for each frame].
        iou_threshold (float): Minimum IoU for a valid match.

    Returns:
        np.array: Array of modified instance GT
    """
    if not Instance_gt_list:
        zero_output = np.zeros((11, len(Initial_guess_seq[0][0]), len(Initial_guess_seq[0][0][0])))
        zero_output[0, :, :] = 1
        return zero_output

    modified_instance_gt_seq = []

    for frame in range(len(Instance_gt_list)):
        modified_instance_gt = np.zeros((11, len(Initial_guess_seq[0][0]), len(Initial_guess_seq[0][0][0])))
        modified_instance_gt[0, :, :] = 1
        initial_guess = Initial_guess_seq[frame, :, :, :]
        gt_masks = Instance_gt_list[frame] # masks for gt instance id at a frame


        # Make connections
        for channel in range(1, 11):
            max_iou = 0
            max_gt = None
            for gt_mask in gt_masks.values():
                iou = calculate_iou(initial_guess[2+channel, :, :], gt_mask)
                if iou > max_iou:
                    max_gt = gt_mask
                    max_iou = iou

            if max_iou > iou_threshold:
                modified_instance_gt[channel, :, :] = max_gt
                modified_instance_gt[0, :, :] = np.maximum(modified_instance_gt[0, :, :] - max_gt, 0)
            else:
                # discard initial guess if no gt_mask is matched
                Initial_guess_seq[frame, 2+channel, :, :] = np.zeros_like(Initial_guess_seq[frame, 2+channel, :, :])

        modified_instance_gt_seq.append(modified_instance_gt)

    modified_instance_gt_seq = np.array(modified_instance_gt_seq)

    return modified_instance_gt_seq


def Instance_eval(gt_instances, pred_instances, last_matches, iou_threshold=0.5):
    total_tp = 0
    total_fp = 0
    total_ids = 0
    total_gt_instances = len(gt_instances)

    if total_gt_instances == 0:
        return 1, 0, 0  # Avoid division by zero if no ground truth instances are present

    current_matches = {} # GT/pred instance id for current frame

    # Matching ground truth instances to predicted instances using IoU
    for pred_id, pred_mask in pred_instances.items():
        if pred_id == 0: continue # background
        if np.sum(pred_mask) == 0: continue # not tracked
        best_iou = 0
        best_gt_id = None
        for gt_id, gt_mask in gt_instances.items():
            iou = calculate_iou(gt_mask, pred_mask)
            if iou > iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id

        if best_gt_id is not None:
            current_matches[best_gt_id] = pred_id
            total_tp += 1
        else:
            total_fp += 1

    # Count ID switches
    for gt_id in current_matches:
        for frame_idx in range(len(last_matches) - 1, -1, -1):
            if gt_id in last_matches[frame_idx].keys():
                if current_matches[gt_id] != last_matches[frame_idx][gt_id]:
                    total_ids += 1
                    break

    last_matches.append(current_matches)

    if total_tp + total_fp == 0:
        return total_tp/total_gt_instances, 1, total_ids
    return total_tp/total_gt_instances, total_tp/(total_tp+total_fp), total_ids# True positive rate, Positive predictive value, Id switch