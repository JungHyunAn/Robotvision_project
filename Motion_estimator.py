# Import modules
import cv2
import torch
import torchvision
import mmcv
import mmengine
import matplotlib.pyplot as plt
import numpy as np

# Function definitions
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

def Track_image_sequence(image_sequence: np.ndarray, YOLO_model, tracker, sequence_length=0):
    # Initialize list to hold detections and tracking results
    box_list_sequence = []

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
        frame_boxes = []
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

                # Append [Class_ID, Instance_ID, left_x, up_y, right_x, down_y]
                if cls in [2, 3, 5, 7]: # Assume as car if COCO label is 'car' or 'motorcycle' or 'bus' or 'truck'
                    frame_boxes.append([1, track_id, x1, y1, x2, y2])
                elif cls in [0]: # Assume as pedestiran if COCO label is 'person'
                    frame_boxes.append([2, track_id, x1, y1, x2, y2])
                else:
                    frame_boxes.append([0, track_id, x1, y1, x2, y2])

        # Append detections for the current frame to the sequence list
        box_list_sequence.append(frame_boxes)

    return box_list_sequence


def Construct_initial_guess(image : np.array, box_list, depth_map):
    # box_list format: [[Class_ID, Instance_ID, left_x, up_y, right_x, down_y], ... for frame]

    # Expand depth_map to 3 dimensions to match image dimensions
    depth_map_expanded = depth_map[:, :, np.newaxis]  # Add a new axis for depth

    # Expand image to have extra channels for class_ID and instance_ID
    grayscale_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    initial_guess = np.concatenate((grayscale_image[:, :, np.newaxis] , np.zeros_like(image[:, :, :2])), axis=2)
    initial_guess = np.concatenate((initial_guess, depth_map_expanded), axis=2)

    # Dictionary to store each instance's depth and coordinates
    instance_depth = {}
    for class_id, instance_id, left_x, up_y, right_x, down_y in box_list:
        left_x = max(0, left_x)
        right_x = min(depth_map.shape[1] - 1, right_x)
        up_y = max(0, up_y)
        down_y = min(depth_map.shape[0] - 1, down_y)

        avg_depth = np.mean(depth_map[up_y:down_y+1, left_x:right_x+1])
        instance_depth[instance_id] = (avg_depth, class_id, left_x, up_y, right_x, down_y)
  
    # Sort instances by depth and assign class and instance IDs
    sorted_keys = sorted(instance_depth, key=lambda k: instance_depth[k][0], reverse=True)
    for instance_id in sorted_keys:
        _, class_id, left_x, up_y, right_x, down_y = instance_depth[instance_id]
        initial_guess[up_y:down_y+1, left_x:right_x+1, 1] = class_id
        initial_guess[up_y:down_y+1, left_x:right_x+1, 2] = instance_id
  
    return initial_guess # (Gray, class_id, instance_id, depth)

def calculate_iou(box1, box2):
    # Unpack the coordinates of each box
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the coordinates of the intersection rectangle
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    # Calculate the intersection area
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Calculate the area of each box
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate the IoU
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

def Preprocess_gt(box_list, MOTS_gt_list):
    for frame in range(len(MOTS_gt_list)):
        mots_objects = MOTS_gt_list[frame]
        boxes = box_list[frame]

        for mots_obj in mots_objects:
            mots_track_id = mots_obj['track_id']
            mots_bbox = mots_obj['bbox']
            
            # box_list에서 ID가 매칭되지 않는 경우
            if not any(box['track_id'] == mots_track_id for box in boxes):
                max_iou = 0
                replacement_id = None
                
                # IoU 기준으로 가장 큰 box_list 객체의 ID 찾기
                for box in boxes:
                    box_bbox = box['bbox']
                    iou = calculate_iou(mots_bbox, box_bbox)

                    # IoU가 threshold 이상이고 현재 최대 iou보다 큰 경우 ID 업데이트
                    if iou > 0.5 and iou > max_iou:
                        max_iou = iou
                        replacement_id = box['track_id']
                
                # 새로운 ID를 대체
                if replacement_id is not None:
                    mots_obj['track_id'] = replacement_id
                    mots_obj['iou'] = max_iou  # IoU 정보 저장 (옵션)
                    
    return MOTS_gt_list


def GT2DetectID(Detect_list, Instance_gt_list):
    # Instance_gt_list format : [{instance_id : Segmentation mask} for every frame]
    # Detect_list format: [[[Class_ID, Instance_ID, left_x, up_y, right_x, down_y] for every bounding box] for every frame]
    output_dict = {}
    iou_threshold = 0.2
    picked_target = []
    
    for frame in range(len(Instance_gt_list)):
        gt_masks = Instance_gt_list[frame]
        detect_boxes = Detect_list[frame]

        # Track all mots_gt ids
        for g_id in gt_masks.keys():
            if g_id in output_dict.keys():
                if output_dict[g_id] is not None:
                    continue

            max_iou = 0
            target_id = None

            for d_box in detect_boxes:
                g_mask = gt_masks[g_id]
                d_mask = np.zeros_like(g_mask)
                d_mask[d_box[3]:d_box[5], d_box[2]:d_box[4]] = 1
                intersection = np.sum(np.multiply(g_mask, d_mask))
                iou =  intersection / float(np.sum(g_mask) + (d_box[5]-d_box[3])*(d_box[4]-d_box[2]) - intersection)

                if iou > iou_threshold and iou > max_iou:
                    max_iou = iou
                    target_id = d_box[1]

            output_dict[g_id] = target_id

            if target_id is not None:
                picked_target.append(target_id)
    
    # Assign GT ids with None
    target_candiate = list(set(output_dict.keys()) - set(picked_target))
    i = 0
    for g_id in output_dict.keys():
        if output_dict[g_id] is None:
            output_dict[g_id] = target_candiate[i]
            i += 1

    # Output format : {GT_instance_id : Target id from DeepSORT}
    return output_dict