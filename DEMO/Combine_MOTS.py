import PIL.Image as Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys, os
import wget
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Motion_estimator import Track_image_sequence, Construct_initial_guess, Image_depth

def Combine_MOTS_image(rgb_path, MOTS_mask_path):
    rgb_image = cv2.imread(rgb_path)[:, :, ::-1]
    MOTS_mask = np.array(Image.open(MOTS_mask_path))
    obj_ids = np.unique(MOTS_mask)

    color_mask = np.zeros_like(rgb_image)

    # Assign a color to each object based on class_id and object_instance_id
    for obj_id in obj_ids:
        if obj_id == 0:
            continue  # Skip background

        # Compute class_id and obj_instance_id
        class_id = obj_id // 1000
        obj_instance_id = obj_id % 1000

        if class_id not in [1, 2]:
            continue # Skip 10000?

        np.random.seed(obj_instance_id)  # Ensure consistent color for the same instance
        # Apply color to the object in the mask
        color_mask[MOTS_mask == obj_id] = np.random.randint(0, 255, size=3)

    # Blend the color mask with the original image
    alpha = 0.5  # Transparency factor
    overlay = cv2.addWeighted(rgb_image, 1 - alpha, color_mask, alpha, 0)

    # Create a copy to add text labels
    labeled_image = overlay.copy()

    # Define class names for class_id
    class_names = {1: 'Car', 2: 'Pedestrian'}

    # Annotate the image with class_id and object_instance_id
    for obj_id in obj_ids:
        if obj_id == 0:
            continue  # Skip background
        
        class_id = obj_id // 1000
        obj_instance_id = obj_id % 1000

        if class_id not in [1, 2]:
            continue # Skip 10000?

        class_name = class_names.get(class_id, 'Unknown')

        # Find the position to place the label (centroid of the object mask)
        mask = (MOTS_mask == obj_id)
        y, x = np.where(mask)  # Get object coordinates
        if len(x) > 0 and len(y) > 0:
            centroid_x = int(np.mean(x))
            centroid_y = int(np.mean(y))

            # Create the label text
            label = f"{class_name} (ID: {obj_instance_id})"
            
            # Put the text on the image
            cv2.putText(labeled_image, label, (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return labeled_image

def Combine_Tracker_image(image, box_list):
    # Iterate through each bounding box in the box list
    for box in box_list:
        class_id, track_id, x1, y1, x2, y2 = box

        # Draw the bounding box on the image
        color = (0, 255, 0)  # Green color for bounding box
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Display the class ID and track ID near the bounding box
        text = f"Class: {class_id}, ID: {track_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (0, 0, 255)  # Red color for text

        # Determine position for the text above the bounding box
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10

        # Draw text on the image
        cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    return image


rgb_image = cv2.imread('DEMO/MOTS_single_demo/000115.png')
labeled_image = Combine_MOTS_image('DEMO/MOTS_single_demo/000115.png', 'DEMO/MOTS_single_demo/000115_gt.png')

model_url = 'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt'
model_path = 'yolov10x.pt'

# Download the model only if it doesn't already exist
if not os.path.isfile(model_path):
    print("Downloading YOLO model...")
    model_path = wget.download(model_url)
else:
    print("YOLO model already exists, skipping download.")
model = YOLO(model_path)
# 최적 가중치 로드
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 모델을 장치에 할당
model.to(device)
# DeepSORT 초기화 (트래킹)
tracker = DeepSort(max_age=30, n_init=0, nn_budget=200)
image_sequence = [rgb_image]
bounding_box_sequence = Track_image_sequence(image_sequence, model, tracker, 1)

# Original image
plt.subplot(3, 1, 1)
plt.imshow(rgb_image)
plt.title("Original Image")
plt.axis('off')

# Overlay of image and mask
plt.subplot(3, 1, 2)
plt.imshow(labeled_image)
plt.title("Image with Ground Truth MOTS Mask")
plt.axis('off')

plt.subplot(3, 1, 3)
plt.imshow(Combine_Tracker_image(rgb_image, bounding_box_sequence[0]))
plt.title("Image with DeepSORT Bounding Boxes")
plt.axis('off')


plt.tight_layout()
plt.savefig('DEMO/MOTS_single_demo/MOTS_combined.png')
plt.show()