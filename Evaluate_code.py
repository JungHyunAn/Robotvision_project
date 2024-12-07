import torch
import os
import wget
from ultralytics import YOLO
from Evaluate_model import eval_model
from Model import Pred_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define device


# Load YOLO model
YOLO_model_url = 'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10x.pt'
YOLO_model_path = 'yolov10x.pt'
# Download the model only if it doesn't already exist
if not os.path.isfile(YOLO_model_path):
    print("Downloading YOLO model...")
    model_path = wget.download(YOLO_model_url)
else:
    print("YOLO model already exists, skipping download.")
YOLO_model = YOLO(YOLO_model_path)
YOLO_model.to(device)

# Load depth model
depth_model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
depth_model.cuda().eval()
depth_model.to(device) # Move model to device


# Construct training/validiation root list
train_root_list = []
val_root_list = []
with open('matched_train.csv', mode='r') as file:
    next(file) #skip header
    for line in file:
        raw_path, MOTS_path, depth_path, s_idx, l_idx, fx, fy, cx, cy = line.strip().split(',')
        train_root_list.append([raw_path.replace("\\", "/"), MOTS_path.replace("\\", "/"), depth_path.replace("\\", "/"), int(s_idx), int(l_idx), [float(fx), float(fy), float(cx), float(cy)]])

with open('matched_val.csv', mode='r') as file:
    next(file) #skip header
    for line in file:
        raw_path, MOTS_path, depth_path, s_idx, l_idx, fx, fy, cx, cy = line.strip().split(',')
        val_root_list.append([raw_path.replace("\\", "/"), MOTS_path.replace("\\", "/"), depth_path.replace("\\", "/"), int(s_idx), int(l_idx), [float(fx), float(fy), float(cx), float(cy)]])



# Define prediction model
model = Pred_model()

# Train model
eval_model(model, YOLO_model, depth_model, val_root_list, device, pre_trained_path='models\\revised13_model_epoch_7.pth')