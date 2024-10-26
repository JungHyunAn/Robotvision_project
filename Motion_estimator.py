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

def Sequence_bounding_box(image_sequence : np.array):
  # box_list_sequence format: [[[Class_ID, Instance_ID, left_x, up_y, right_x, down_y], ... for frame1], [~~ for frame2], ...]
  box_list_sequence = [] 

  return box_list_sequence

def Construct_initial_guess(image, box_list, depth_map):
  # box_list format: [[Class_ID, Instance_ID, left_x, up_y, right_x, down_y], ... for frame]

  # initial_guess format: [[[r, g, b, class_ID, instance_ID, depth] for each pixel]]
  initial_guess = np.concatenate((image, np.zeros_like(image[:, :, 0:2])), axis=2)
  initial_guess = np.concatenate((initial_guess, depth_map), axis=2)

  # First calculate approximate depth for each Class_ID
  instance_depth = {}
  for class_id, instance_id, left_x, up_y, right_x, down_y in box_list:
    left_x = max(0, left_x)
    right_x = min(depth_map.shape[1] - 1, right_x)
    up_y = max(0, up_y)
    down_y = min(depth_map.shape[0] - 1, down_y)

    avg_depth = np.mean(depth_map[up_y:down_y+1, left_x:right_x+1])
    instance_depth[instance_id] = (avg_depth, class_id, left_x, up_y, right_x, down_y)
  
  # Give pixels ID from furthest
  sorted_keys = sorted(instance_depth, key=lambda k: instance_depth[k][0], reverse=True)
  for instance_id in sorted_keys:
    _, class_id, left_x, up_y, right_x, down_y = instance_depth[instance_id]
    initial_guess[up_y:down_y+1, left_x:right_x+1, 3] = class_id
    initial_guess[up_y:down_y+1, left_x:right_x+1, 4] = instance_id
  
  return initial_guess