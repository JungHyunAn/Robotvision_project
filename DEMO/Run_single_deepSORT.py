import wget
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import pandas as pd
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import plasma

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Motion_estimator import Track_image_sequence, Construct_initial_guess, Image_depth

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

def Visualize_guess(initial_guess):
    # Split initial_guess into separate components
    rgb_image = initial_guess[:, :, :3].astype(np.uint8)  # RGB channels
    class_id_image = initial_guess[:, :, 3]  # Class IDs
    instance_id_image = initial_guess[:, :, 4]  # Instance IDs
    depth_image = initial_guess[:, :, 5]  # Depth channel

    # Create Class ID visualization with random colors
    unique_classes = np.unique(class_id_image)
    class_id_colormap = np.zeros_like(rgb_image)
    for class_id in unique_classes:
        if class_id == 0: continue  # Skip background
        mask = (class_id_image == class_id)
        color = np.random.randint(0, 255, size=3)  # Random color for each class
        class_id_colormap[mask] = color

    # Create Instance ID visualization with random colors
    unique_instances = np.unique(instance_id_image)
    instance_id_colormap = np.zeros_like(rgb_image)
    for instance_id in unique_instances:
        np.random.seed(int(instance_id))  # Ensure consistent color for the same instance
        if instance_id == 0: continue  # Skip background
        mask = (instance_id_image == instance_id)
        color = np.random.randint(0, 255, size=3)  # Random color for each instance
        instance_id_colormap[mask] = color

    # Normalize depth map for visualization
    norm = Normalize(vmin=np.min(depth_image), vmax=np.max(depth_image))
    depth_colormap = plasma(norm(depth_image))[:, :, :3] * 255  # Convert to RGB colormap and scale to 255
    depth_colormap = depth_colormap.astype(np.uint8)

    # Composite View: Overlay instance_id_colormap on top of RGB image
    overlay_image = cv2.addWeighted(rgb_image, 0.6, instance_id_colormap.astype(np.uint8), 0.4, 0)

    # Stack all components along a new axis for a unified output
    output_array = np.stack((rgb_image, class_id_colormap, instance_id_colormap, depth_colormap, overlay_image), axis=0)
    return output_array  # Shape: (5, height, width, 3)

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
tracker = DeepSort(max_age=30, n_init=3, nn_budget=200)

depth_model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
depth_model.cuda().eval()
depth_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")) 

image_sequence = []
for i in range(446):
    if i < 10:
        img_path = 'DEMO/MOTS_sequence_demo/0001_rgb/00000' + str(i) + '.png'
    elif i < 100:
        img_path = 'DEMO/MOTS_sequence_demo/0001_rgb/0000' + str(i) + '.png'
    else:
        img_path = 'DEMO/MOTS_sequence_demo/0001_rgb/000' + str(i) + '.png'
    image_sequence.append(cv2.imread(img_path))

box_list_sequence = Track_image_sequence(image_sequence, model, tracker, 446)

'''
for i in range(446):
    if i < 10:
        combined_path = 'DEMO/Sequence_demo/0001_rgb_combined/00000' + str(i) + '.png'
    elif i < 100:
        combined_path = 'DEMO/Sequence_demo/0001_rgb_combined/0000' + str(i) + '.png'
    else:
        combined_path = 'DEMO/Sequence_demo/0001_rgb_combined/000' + str(i) + '.png'
    cv2.imwrite(combined_path, Combine_Tracker_image(image_sequence[i], box_list_sequence[i]))
'''

for i in range(446):
    if i < 10:
        guessed_path = 'DEMO/Sequence_demo/0001_rgb_initial_guess/00000' + str(i) + '.png'
    elif i < 100:
        guessed_path = 'DEMO/Sequence_demo/0001_rgb_initial_guess/0000' + str(i) + '.png'
    else:
        guessed_path = 'DEMO/Sequence_demo/0001_rgb_initial_guess/000' + str(i) + '.png'
    pred_depth = Image_depth(image_sequence[i], depth_model, [721.5, 721.5, 609.6, 172.9])
    initial_guess = Visualize_guess(Construct_initial_guess(image_sequence[i], box_list_sequence[i], pred_depth))[4]
    cv2.imwrite(guessed_path, initial_guess)
