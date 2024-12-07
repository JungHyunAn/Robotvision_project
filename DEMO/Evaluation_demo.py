import torch
import os
import wget
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import PIL.Image as Image
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
import sys, os, glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Motion_estimator import Track_image_sequence, Construct_initial_guess, Image_depth, GT2DetectID
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
model.load_state_dict(torch.load('models\\revised13_model_epoch_7.pth'))
model.to(device)
model.eval()

raw_root = 'data_tracking_image_2\\training\\image_02\\0003'
output_dir = 'DEMO\\Prediction_evaluation_demo'
os.makedirs(output_dir, exist_ok=True)

input_seq = []
for i in range(0, 300):
    if i < 10:
        img_path = raw_root + '/00000' + str(i) + '.png'
    elif i < 100:
        img_path = raw_root + '/0000' + str(i) + '.png'
    elif i < 1000:
        img_path = raw_root + '/000' + str(i) + '.png'
    else:
        img_path = raw_root + '/00' + str(i) + '.png'

    if not (os.path.isfile(img_path)):
        continue
    
    input_seq.append(cv2.imread(img_path))

tracker = DeepSort(max_age=30, n_init=0, nn_budget=200)
bbox_seq = Track_image_sequence(input_seq, YOLO_model, tracker, len(input_seq))

initial_guess_seq = [] # Initial guess list for entire batch
C2bboxID_seq = []
for frame in tqdm(range(len(input_seq)), desc="Making initial guesses"):
    depth_map = Image_depth(input_seq[frame], depth_model, [721.5, 721.5, 609.6, 172.9])
    initial_guess_seq.append(Construct_initial_guess(input_seq[frame], bbox_seq[frame], depth_map, C2bboxID_seq))

initial_guess_seq = np.transpose(np.array(initial_guess_seq), (0, 3, 1, 2))

hidden = None
for idx in tqdm(range(0, len(initial_guess_seq), 10), desc="Processing Batches"):
    ldx = min(idx+10, len(initial_guess_seq)) # last index of batch

    initial_guess_mini = torch.from_numpy(initial_guess_seq[idx : ldx]).float().to(device) # slice precalculated initial_guess

    if hidden is not None:
        hidden[0] = hidden[0].detach()

    pred_instance_seq, pred_depth_seq, hidden = model(initial_guess_mini, hidden)

    # Get single map from pred_instance_seq
    pred_instance_seq = torch.argmax(pred_instance_seq, dim=1)
    pred_instance_seq = pred_instance_seq.detach().cpu().numpy()

    pred_depth_seq = pred_depth_seq.detach().cpu().numpy()              

    for frame in range(len(pred_instance_seq)):
        fig, axes = plt.subplots(1, 2, figsize=(18, 12))

        # Overlay RGB and Instance Prediction with transparency for value 0
        rgb_image = cv2.cvtColor(input_seq[idx + frame],cv2.COLOR_BGR2RGB)
        instance_prediction = pred_instance_seq[frame]
        masked_instance = np.ma.masked_where(instance_prediction == 0, instance_prediction)

        axes[0].imshow(rgb_image)  # RGB image
        overlay = axes[0].imshow(masked_instance, cmap='tab10', alpha=0.8)  # Overlay with transparency
        axes[0].set_title('RGB + Instance Prediction')
        axes[0].axis('off')

        # Add legend directly to the combined plot
        unique_instances_pred = np.unique(instance_prediction)
        legend_patches_pred = [
            mpatches.Patch(color=overlay.cmap(overlay.norm(val)), label=f'Instance {C2bboxID_seq[idx+frame][val][0]}')
            for val in unique_instances_pred if (val != 0 and C2bboxID_seq[idx+frame][val] is not None)
        ]
        axes[0].legend(
            handles=legend_patches_pred,
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0,
            title="Instance Legend"
        )

        # Depth Prediction
        depth_prediction = pred_depth_seq[frame]
        im_depth = axes[1].imshow(depth_prediction, cmap='plasma')
        axes[1].set_title('Depth Prediction')
        axes[1].axis('off')

        # Add a vertical colorbar for depth
        cbar = fig.colorbar(im_depth, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Depth Value')
        
        # Save the visualization
        plt.tight_layout()
        output_file_path = os.path.join(output_dir, f'frame_{idx+frame:04d}_visualization.png')
        plt.savefig(output_file_path)
        plt.close()

# Create video from images
frame_paths = sorted(glob.glob(os.path.join(output_dir, 'frame_*.png')))
frame = cv2.imread(frame_paths[0])
height, width, _ = frame.shape

# Define video writer
video_file = os.path.join(output_dir, 'output_video.mp4')
video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 8, (width, height))

for frame_path in frame_paths:
    frame = cv2.imread(frame_path)
    video_writer.write(frame)

video_writer.release()

print(f"Video saved at {video_file}")