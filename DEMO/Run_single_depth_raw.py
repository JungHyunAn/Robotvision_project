import cv2
import torch
import torchvision
import mmcv
import mmengine
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Motion_estimator import Image_depth

model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
model.cuda().eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define device
model.to(device) # Move model to device

image_file = 'DEMO\\MOTS_sequence_demo\\0001_rgb\\000000.png'
image = cv2.imread(image_file)[:, :, ::-1]

depth_file = 'DEMO/Single_depth_demo/0000000006_gt.png'
gt_depth_scale  = 256.0
gt_depth = cv2.imread(depth_file, -1)
gt_depth = gt_depth/gt_depth_scale
mask = (gt_depth > 1e-8)

# [721.5, 721.5, 609.6, 172.9] for 0000000005_rgb.png
pred_depth = Image_depth(image, model, [721.5, 721.5, 609.6, 172.9])

output_file = 'pred_depth.csv'
np.savetxt(output_file, pred_depth, delimiter=',')

print(f"Predicted depth array saved as {output_file}")
