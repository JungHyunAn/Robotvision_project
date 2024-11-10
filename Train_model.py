import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
import PIL.Image as Image
import random
import os
from torch.nn import init
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort
from Motion_estimator import Image_depth, Track_image_sequence, Construct_initial_guess, GT2DetectID, calculate_mota


def produce_entire_sequence(raw_root, mots_root, depth_root, s_idx, l_idx):
    entire_input_seq = [] # List of all raw frames
    entire_instance_seq = [] # List of all MOTS instance ID gt in mask dict
    entire_class_seq = [] # List of all MOTS class ID gt
    entire_depth_seq = [] # List of all depth gt

    # Produce input/instance/class/depth sequences
    for i in range(s_idx, l_idx+1):
        if i < 10:
            img_path = raw_root + '/00000' + str(i) + '.png'
            MOTS_path = mots_root + '/00000' + str(i) + '.png'
            depth_path = depth_root + '/000000000' + str(i) + '.png'
        elif i < 100:
            img_path = raw_root + '/0000' + str(i) + '.png'
            MOTS_path = mots_root + '/0000' + str(i) + '.png'
            depth_path = depth_root + '/00000000' + str(i) + '.png'
        elif i < 1000:
            img_path = raw_root + '/000' + str(i) + '.png'
            MOTS_path = mots_root + '/000' + str(i) + '.png'
            depth_path = depth_root + '/0000000' + str(i) + '.png'
        else:
            img_path = raw_root + '/00' + str(i) + '.png'
            MOTS_path = mots_root + '/00' + str(i) + '.png'
            depth_path = depth_root + '/000000' + str(i) + '.png'

        if not (os.path.isfile(img_path) and os.path.isfile(MOTS_path) and os.path.isfile(depth_path)):
            continue

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
            n_batch = (len(entire_input_seq) - 1) // 50 + 1
            batch_breaks = [s_idx]
            batch_breaks = batch_breaks + sorted(random.sample(range(s_idx+1, len(entire_input_seq)), n_batch - 1))
            batch_breaks.append(len(entire_input_seq))
            
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
                initial_guess_seq = []
                for frame in range(batch_breaks[i+1] - batch_breaks[i]):
                    depth_map = Image_depth(input_seq[frame], depth_model, cam_int)
                    initial_guess_seq.append(Construct_initial_guess(input_seq[frame], bbox_seq[frame], depth_map))

                # Construct dict between GT instance ID and bbox instance ID / Modify instance_seq
                GT2bbox_instance_dict = GT2DetectID(bbox_seq, instance_seq)
                instance_seq = [{GT2bbox_instance_dict.get(k, k): v for k, v in instance_dict.items()} for instance_dict in instance_seq]
                
                # Forward propagate in model
                initial_guess_seq = np.transpose(np.array(initial_guess_seq), (0, 3, 1, 2))
                class_seq = np.array(class_seq)
                depth_seq = np.array(depth_seq)

                initial_guess_seq = torch.from_numpy(initial_guess_seq).float().to(device)
                class_seq = torch.from_numpy(class_seq).float().to(device)
                depth_seq = torch.from_numpy(depth_seq).float().to(device)

                pred_class_seq, pred_instance_seq, pred_depth_seq = model(initial_guess_seq)

                # Compute loss
                instance_seq = [[k*v for k, v in instance_dict.items()].sum(axis=0) for instance_dict in instance_seq] # Concatentate mask
                instance_seq = np.array(instance_seq)
                instance_seq = torch.from_numpy(instance_seq).float().to(device)
                loss = criterion(pred_class_seq, class_seq, pred_instance_seq, instance_seq, pred_depth_seq, depth_seq)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update Epoch loss
                epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / 5896
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_epoch_loss:.4f}")

        # Evaluation every 5 epochs
        min_mse = float("inf")
        min_cross_entropy = float("inf")
        max_mota = 0
        if epoch % 5 == 0 and epoch:
            model.eval()
            val_mse_loss = 0.0
            val_cross_entropy_loss = 0.0
            total_mota = 0.0
            total_frames = 0

            for raw_root, mots_root, depth_root, s_idx, l_idx, cam_int in val_root_list:
                with torch.no_grad():
                    # Produce entire sequence for video
                    entire_input_seq, entire_class_seq, entire_instance_seq, entire_depth_seq = produce_entire_sequence(raw_root, mots_root, depth_root, s_idx, l_idx)

                    # Forward propagate
                    tracker = DeepSort(max_age=30, n_init=0, nn_budget=200)
                    bbox_seq = Track_image_sequence(entire_input_seq, YOLO_model, tracker, len(entire_input_seq))

                    initial_guess_seq = [Construct_initial_guess(entire_input_seq[frame], bbox_seq, Image_depth(entire_input_seq[frame], depth_model, cam_int)) for frame in range(len(entire_input_seq))]
                    initial_guess_seq = np.transpose(np.array(initial_guess_seq), (0, 3, 1, 2))
                    class_seq = np.array(entire_class_seq)
                    depth_seq = np.array(entire_depth_seq)

                    initial_guess_seq = torch.from_numpy(initial_guess_seq).float().to(device)
                    class_seq = torch.from_numpy(class_seq).long().to(device)  # For cross-entropy, class labels should be long type
                    depth_seq = torch.from_numpy(depth_seq).float().to(device)

                    pred_class_seq, pred_instance_seq, pred_depth_seq = model(initial_guess_seq)

                    # Compute validation losses
                    # MSE Loss for Depth Prediction
                    mse_loss = F.mse_loss(pred_depth_seq, depth_seq)
                    val_mse_loss += mse_loss.item()

                    # Cross-Entropy Loss for Class Prediction
                    cross_entropy_loss = F.cross_entropy(pred_class_seq, class_seq)
                    val_cross_entropy_loss += cross_entropy_loss.item()

                    # Calculate MOTA for Instance Prediction
                    for frame_idx in range(len(pred_instance_seq)):
                        gt_instances = entire_instance_seq[frame_idx]
                        pred_instances = pred_instance_seq[frame_idx]
                        mota = calculate_mota(gt_instances, pred_instances)  # Custom function to calculate MOTA for the frame
                        total_mota += mota
                        total_frames += 1

            # Average the losses and MOTA score
            avg_mse_loss = val_mse_loss / len(val_root_list)
            avg_cross_entropy_loss = val_cross_entropy_loss / len(val_root_list)
            avg_mota = total_mota / total_frames if total_frames > 0 else 0.0

            print(f"Validation - Epoch [{epoch+1}/{n_epochs}], Depth MSE Loss (Depth): {avg_mse_loss:.4f}, Class Cross-Entropy Loss (Class): {avg_cross_entropy_loss:.4f}, Instance MOTA: {avg_mota:.4f}")

            # Save model if 2 of 3 improve
            if (avg_mse_loss < min_mse and avg_cross_entropy_loss < min_cross_entropy) \
            or (avg_cross_entropy_loss < min_cross_entropy and avg_mota > max_mota) \
            or (avg_mota > max_mota and avg_mse_loss < min_mse):
                model_save_path = os.path.join("models", f"model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at {model_save_path}")

            model.train()
