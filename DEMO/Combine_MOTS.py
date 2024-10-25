import PIL.Image as Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Combine_MOTS_sequence import Combine_MOTS_image

rgb_image = cv2.imread('MOTS_single_demo/000115.png')[:, :, ::-1]
labeled_image = Combine_MOTS_image('MOTS_single_demo/000115.png', 'MOTS_single_demo/000115_gt.png')

# Original image
plt.subplot(2, 1, 1)
plt.imshow(rgb_image)
plt.title("Original Image")
plt.axis('off')

# Overlay of image and mask
plt.subplot(2, 1, 2)
plt.imshow(labeled_image)
plt.title("Image with MOTS Mask and Annotations")
plt.axis('off')

plt.tight_layout()
plt.savefig('MOTS_single_demo/MOTS_combined.png')
plt.show()