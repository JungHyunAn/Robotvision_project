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
  h, w = image.shape[:2]

  # Scale image
  input_size = (616, 1064) # for vit model
  scale = min(input_size[0] / h, input_size[1] / w)
  rgb = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
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
  pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], (h,w), mode='bilinear').squeeze()
  
  # de-canonical transform
  canonical_to_real_scale = new_intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
  pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
  pred_depth = torch.clamp(pred_depth, 0, 300)

  # Transform to np.array
  depth_map = pred_depth.squeeze().cpu().numpy()
  
  assert type(depth_map) is np.array, 'Output type error in Image_depth'
  
  return depth_map
