import cv2
import torch
import torchvision
import mmcv
import mmengine
import matplotlib.pyplot as plt
import numpy as np
from Motion_estimator import Image_depth

model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
model.cuda().eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define device
model.to(device) # Move model to device

image_file = 'Single_depth_demo/0000000005_rgb.png'
image = cv2.imread(image_file)[:, :, ::-1]

depth_file = 'Single_depth_demo/0000000005_gt.png'
gt_depth_scale  = 256.0
gt_depth = cv2.imread(depth_file, -1)
gt_depth = gt_depth/gt_depth_scale
mask = (gt_depth > 1e-8)

pred_depth = Image_depth(image, model, [707.0493, 707.0493, 604.0814, 180.5066])

pred_depth = np.where(mask, pred_depth, 0)

plt.subplot(3, 1, 1)
plt.imshow(image)
plt.axis('off')

plt.subplot(3, 1, 2)
plt.imshow(pred_depth, cmap='viridis')  # You can try 'plasma', 'inferno', etc.
plt.colorbar(label='Predicted Depth')  # Show original depth values on colorbar
plt.axis('off')  # Optional: Hide axis

plt.subplot(3, 1, 3)
plt.imshow(gt_depth, cmap='viridis')  # You can try 'plasma', 'inferno', etc.
plt.colorbar(label='Estimated Depth')  # Show original depth values on colorbar
plt.axis('off')  # Optional: Hide axis

plt.savefig('Single_depth_demo/comparison.png')
plt.show()

'''

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