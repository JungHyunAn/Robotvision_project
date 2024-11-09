import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
import PIL.Image as Image
import random
from torch.nn import init
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort
from Motion_estimator import Image_depth, Track_image_sequence, Construct_initial_guess, GT2DetectID


def produce_entire_sequence(raw_root, mots_root, depth_root, s_idx, l_idx):
    entire_input_seq = [] # List of all raw frames
    entire_instance_seq = [] # List of all MOTS instance ID gt in mask dict
    entire_class_seq = [] # List of all MOTS class ID gt
    entire_depth_seq = [] # List of all depth gt

    # Produce input/instance/class/depth sequences
    for i in range(s_idx, l_idx+1):
        if i < 10:
            img_path = raw_root + '\\00000' + str(i) + '.png'
            MOTS_path = mots_root + '\\00000' + str(i) + '.png'
            depth_path = depth_root + '\\000000000' + str(i) + '.png'
        elif i < 100:
            img_path = raw_root + '\\0000' + str(i) + '.png'
            MOTS_path = mots_root + '\\0000' + str(i) + '.png'
            depth_path = depth_root + '\\00000000' + str(i) + '.png'
        elif i < 1000:
            img_path = raw_root + '\\000' + str(i) + '.png'
            MOTS_path = mots_root + '\\000' + str(i) + '.png'
            depth_path = depth_root + '\\0000000' + str(i) + '.png'
        else:
            img_path = raw_root + '\\00' + str(i) + '.png'
            MOTS_path = mots_root + '\\00' + str(i) + '.png'
            depth_path = depth_root + '\\000000' + str(i) + '.png'

        # Read raw images
        entire_input_seq.append(cv2.imread(img_path))

        # Read MOTS instance and class ID gt
        MOTS_map = np.array(Image.open(MOTS_path))

        class_map = MOTS_map // 1000
        class_map[class_map == 10] = 0
        entire_class_seq.append(class_map)

        instance_map = MOTS_map % 1000
        instance_dict = {}
        for instance_id in np.unique(instance_map):
            instance_dict[instance_id] = (instance_map == instance_id).astype(int)
        entire_instance_seq.append(instance_dict)

        # Read Depth gt
        depth_map = cv2.imread(depth_path, -1) / 256.0
        entire_depth_seq.append(depth_map)

    return entire_input_seq, entire_class_seq, entire_instance_seq, entire_depth_seq


def train_model(model, YOLO_model, depth_model, criterion, optimizer, train_root_list, val_root_list, n_epochs, device):
    # train/val_root_list = [[raw_root, mots_root, depth_root, start_index, last_index, camera_intrinsic], ...]
    
    model.train()
    model.to(device)

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        # Train for each video sequence
        for raw_root, mots_root, depth_root, s_idx, l_idx, cam_int in tqdm(train_root_list,desc=f"Epoch [{epoch+1}/{n_epochs}]"):
            
            # Produce entire sequence for video
            entire_input_seq, entire_class_seq, entire_instance_seq, entire_depth_seq = produce_entire_sequence(raw_root, mots_root, depth_root, s_idx, l_idx)

            # Slice entire_seq in average of 50 frames
            n_batch = (l_idx - s_idx) // 50 + 1
            batch_breaks = [s_idx]
            batch_breaks = batch_breaks + random.sample(range(s_idx+2, l_idx), n_batch - 1)
            batch_breaks.append(l_idx + 1)
            
            for i in range(n_batch):
                # Sequence for each batch
                input_seq = entire_input_seq[batch_breaks[i] : batch_breaks[i+1]]
                class_seq = entire_class_seq[batch_breaks[i] : batch_breaks[i+1]]
                instance_seq = entire_instance_seq[batch_breaks[i] : batch_breaks[i+1]]
                depth_seq = entire_depth_seq[batch_breaks[i] : batch_breaks[i+1]]

                # Run DeepSORT tracking
                tracker = DeepSort(max_age=30, n_init=0, nn_budget=200)
                bbox_seq = Track_image_sequence(input_seq, YOLO_model, tracker, batch_breaks[i+1] - batch_breaks[i])

                # Construct Initial Guess Sequence
                intital_guess_seq = []
                for frame in range(batch_breaks[i+1] - batch_breaks[i]):
                    depth_map = Image_depth(input_seq[frame], depth_model, cam_int)
                    intital_guess_seq.append(Construct_initial_guess(input_seq[frame], bbox_seq[frame], depth_map))

                # Construct dict between GT instance ID and bbox instance ID / Modify instance_seq
                GT2bbox_instance_dict = GT2DetectID(bbox_seq, instance_seq)
                instance_seq = [np.vectorize(lambda x: GT2bbox_instance_dict.get(x, x))(arr) for arr in instance_seq]
                
                # Forward propagate in model
                pred_class_seq, pred_instance_seq, pred_depth_seq = model(input_seq)

                # Compute loss
                loss = criterion(pred_class_seq, class_seq, pred_instance_seq, instance_seq, pred_depth_seq, depth_seq)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update Epoch loss
                epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / 5896
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_epoch_loss:.4f}")