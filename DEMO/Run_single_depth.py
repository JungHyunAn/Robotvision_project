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

image_file = 'DEMO/MOTS_sequence_demo/0001_rgb/000006.png'
image = cv2.imread(image_file)[:, :, ::-1]

depth_file = 'DEMO/Single_depth_demo/0000000006_gt.png'
gt_depth_scale  = 256.0
gt_depth = cv2.imread(depth_file, -1)
gt_depth = gt_depth/gt_depth_scale
mask = (gt_depth > 1e-8)

# [721.5, 721.5, 609.6, 172.9] for 0000000005_rgb.png
pred_depth = Image_depth(image, model, [721.5, 721.5, 609.6, 172.9])

plt.figure(figsize=(8, 6))
pred_img = plt.imshow(pred_depth, cmap='viridis')  # Fixed typo here
cbar = plt.colorbar(pred_img, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Depth Value')
plt.axis('off')
plt.savefig('DEMO/Single_depth_demo/0001_rgb_000006_pred_depth.png')
plt.show()


pred_depth = np.where(mask, pred_depth, 0)

# Define color limits for consistent color mapping
vmin = min(pred_depth.min(), gt_depth.min())
vmax = max(pred_depth.max(), gt_depth.max())

'''
plt.figure(figsize=(10, 15))

# Plot the original image
plt.subplot(3, 1, 1)
plt.imshow(image)
plt.axis('off')

# Plot the predicted depth
plt.subplot(3, 1, 2)
pred_img = plt.imshow(pred_depth, cmap='viridis', vmin=vmin, vmax=vmax)
plt.axis('off')

# Plot the ground truth depth
plt.subplot(3, 1, 3)
gt_img = plt.imshow(gt_depth, cmap='viridis', vmin=vmin, vmax=vmax)
plt.axis('off')

# Create a single colorbar that applies to both depth images
cbar = plt.colorbar(pred_img, ax=plt.gcf().axes[1:], orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Depth Value')  # Set the colorbar label

# Save and show the plot
plt.savefig('DEMO/Single_depth_demo/comparison.png')
plt.show()

gt_depth = gt_depth / gt_depth_scale
gt_depth = torch.from_numpy(gt_depth).float().cuda()
assert gt_depth.shape == pred_depth.shape
    
mask = (gt_depth > 1e-8)
abs_rel_err = (torch.abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask]).mean()
gt_depth_vis = gt_depth.cpu().numpy()
gt_depth_vis = (gt_depth_vis / gt_depth_vis.max()) * 255  # normalize to [0, 255]
cv2.imwrite('Single_depth_demo/gt_depth_vis.png', gt_depth_vis.astype(np.uint8))

print('abs_rel_err:', abs_rel_err.item())
'''