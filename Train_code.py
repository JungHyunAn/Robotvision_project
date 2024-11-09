import torch
import torch.optim as optim
import os
import wget
from ultralytics import YOLO
from Train_model import train_model
from Model import Pred_model, CombinedLoss


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
        raw_path, MOTS_path, depth_path, s_idx, l_idx, fx, fy, cx, cy = line.strip().split()
        train_root_list.append([raw_path, MOTS_path, depth_path, s_idx, l_idx, [fx, fy, cx, cy]])

with open('matched_val.csv', mode='r') as file:
    next(file) #skip header
    for line in file:
        raw_path, MOTS_path, depth_path, s_idx, l_idx, fx, fy, cx, cy = line.strip().split()
        val_root_list.append([raw_path, MOTS_path, depth_path, s_idx, l_idx, [fx, fy, cx, cy]])



# Define prediction model
model = Pred_model()
criterion = CombinedLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)


# Train model
train_model(model, YOLO_model, depth_model, criterion, optimizer, train_root_list, val_root_list, 100, device)