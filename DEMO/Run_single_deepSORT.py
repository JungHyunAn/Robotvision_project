import wget
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Motion_estimator import Track_image_sequence

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

for i in range(446):
    if i < 10:
        combined_path = 'DEMO/Sequence_demo/0001_rgb_combined/00000' + str(i) + '.png'
    elif i < 100:
        combined_path = 'DEMO/Sequence_demo/0001_rgb_combined/0000' + str(i) + '.png'
    else:
        combined_path = 'DEMO/Sequence_demo/0001_rgb_combined/000' + str(i) + '.png'
    cv2.imwrite(combined_path, Combine_Tracker_image(image_sequence[i], box_list_sequence[i]))