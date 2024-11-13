import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
import PIL.Image as Image
import random
import os
from scipy import stats
from torch.nn import init
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort
from Motion_estimator import Image_depth, Track_image_sequence, Construct_initial_guess, GT2DetectID, calculate_motsa


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


def produce_batch_root(raw_root, mots_root, depth_root, s_idx, l_idx):
    # Produce input/MOTS/depth roots for entire sequence
    entire_input_root = []
    entire_MOTS_root = []
    entire_depth_root = []
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
        
        entire_input_root.append(img_path)
        entire_MOTS_root.append(MOTS_path)
        entire_depth_root.append(depth_path)

    batched_input_root = []
    batched_MOTS_root = []
    batched_depth_root = []

    n_batch = (len(entire_input_root) - 1) // 50 + 1
    batch_breaks = [0]
    batch_breaks = batch_breaks + sorted(random.sample(range(1, len(entire_input_root)), n_batch - 1))
    batch_breaks.append(len(entire_input_root))
    
    for i in range(n_batch):
        batched_input_root.append(entire_input_root[batch_breaks[i]:batch_breaks[i+1]])
        batched_MOTS_root.append(entire_MOTS_root[batch_breaks[i]:batch_breaks[i+1]])
        batched_depth_root.append(entire_depth_root[batch_breaks[i]:batch_breaks[i+1]])

    return batched_input_root, batched_MOTS_root, batched_depth_root, batch_breaks
    

def produce_batch_sequence(batched_input_root, batched_MOTS_root, batched_depth_root):
    batched_input_seq = [] # List of raw frames in batch
    batched_instance_seq = [] # List of MOTS instance ID gt mask dict in batch
    batched_class_seq = [] # List of MOTS class ID gt in batch
    batched_depth_seq = [] # List of depth gt in batch

    for i in range(len(batched_input_root)):
        batched_input_seq.append(cv2.imread(batched_input_root[i]))

        MOTS_map = np.array(Image.open(batched_MOTS_root[i]))

        class_map = MOTS_map // 1000
        class_map[class_map == 10] = 0
        batched_class_seq.append(class_map)

        instance_map = MOTS_map % 1000
        instance_dict = {}
        for instance_id in np.unique(instance_map):
            instance_dict[instance_id] = (instance_map == instance_id).astype(int)
        batched_instance_seq.append(instance_dict)

        # Read Depth gt
        depth_map = cv2.imread(batched_depth_root[i], -1) / 256.0
        batched_depth_seq.append(depth_map)

    return batched_input_seq, batched_class_seq, batched_instance_seq, batched_depth_seq


def train_model(model, YOLO_model, depth_model, criterion, optimizer, train_root_list, val_root_list, n_epochs, device, max_size=30):
    # train/val_root_list = [[raw_root, mots_root, depth_root, start_index, last_index, camera_intrinsic], ...]
    
    model.train()
    model.to(device)

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        total_frames = 0
        # Train for each video sequence
        for raw_root, mots_root, depth_root, s_idx, l_idx, cam_int in tqdm(train_root_list,desc=f"Epoch [{epoch+1}/{n_epochs}]"):
            batched_input_root_seq, batched_MOTS_root_seq, batched_depth_root_seq, batch_breaks = produce_batch_root(raw_root, mots_root, depth_root, s_idx, l_idx)
            
            # Divide video sequence into batches of different length
            for batch in range(len(batched_input_root_seq)):
                batched_input_root = batched_input_root_seq[batch]
                batched_MOTS_root = batched_MOTS_root_seq[batch]
                batched_depth_root = batched_depth_root_seq[batch]

                total_frames += len(batched_input_root_seq[batch])
                
                # Load images for batch / List of np.arrays
                input_seq, class_seq, instance_seq, depth_seq = produce_batch_sequence(batched_input_root, batched_MOTS_root, batched_depth_root)

                # Run DeepSORT tracking
                tracker = DeepSort(max_age=30, n_init=0, nn_budget=200)
                bbox_seq = Track_image_sequence(input_seq, YOLO_model, tracker, len(batched_input_root))

                # Construct Initial Guess Sequence
                initial_guess_seq = []
                for frame in range(len(batched_input_root)):
                    depth_map = Image_depth(input_seq[frame], depth_model, cam_int)
                    initial_guess_seq.append(Construct_initial_guess(input_seq[frame], bbox_seq[frame], depth_map))

                # Construct dict between GT instance ID and bbox instance ID / Modify instance_seq
                GT2bbox_instance_dict = GT2DetectID(bbox_seq, instance_seq)
                instance_seq = [{GT2bbox_instance_dict.get(k, k): v for k, v in instance_dict.items()} for instance_dict in instance_seq]
                
                # Convert types of gt
                initial_guess_seq = np.transpose(np.array(initial_guess_seq), (0, 3, 1, 2))
                class_seq = np.array(class_seq)
                class_seq = class_seq.astype(np.int16)

                instance_seq = [np.sum([k * v for k, v in instance_dict.items()], axis=0)  for instance_dict in instance_seq] # Concatentate mask
                instance_seq = np.array(instance_seq)

                depth_seq = np.array(depth_seq)

                hidden = None
                # Slice batches into mini_batches such that size doesn't exceed max_size
                for idx in range(0, len(input_seq), max_size):
                    ldx = min(idx + max_size, len(input_seq))
                    initial_guess_mini = torch.from_numpy(initial_guess_seq[idx : ldx, :, :, :]).float().to(device)
                    
                    if hidden is not None:
                        hidden[0] = hidden[0].detach()
                        hidden[1] = hidden[1].detach()
                        hidden[2] = hidden[2].detach()

                    pred_class_mini, pred_instance_mini, pred_depth_mini, hidden = model(initial_guess_mini, hidden)

                    # Compute loss in mini_batch
                    class_mini = torch.from_numpy(class_seq[idx : ldx, :, :]).long().to(device)
                    class_mini = F.one_hot(class_mini, num_classes=3).float()
                    class_mini = class_mini.to(device)

                    instance_mini = torch.from_numpy(instance_seq[idx : ldx, :, :]).float().to(device)

                    depth_mini = torch.from_numpy(depth_seq[idx : ldx, :, :]).float().to(device)

                    loss = criterion(pred_class_mini, class_mini, pred_instance_mini, instance_mini, pred_depth_mini, depth_mini)

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update Epoch loss
                    epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / total_frames
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_epoch_loss:.4f}")

        # Evaluation every 5 epochs
        min_mse = float("inf")
        min_cross_entropy = float("inf")
        max_motsa = 0
        if epoch % 5 == 0:
            model.eval()
            val_mse_loss = 0.0
            val_cross_entropy_loss = 0.0
            total_motsa = 0.0
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
                    class_seq = class_seq.astype(np.int16)
                    depth_seq = np.array(entire_depth_seq)

                    mse_loss = 0
                    cross_entropy_loss = 0
                    
                    hidden = None
                    for idx in range(0, len(entire_input_seq), max_size):
                        ldx = min(idx+max_size, len(entire_input_seq))
                        initial_guess_mini = torch.from_numpy(initial_guess_seq[idx : ldx]).float().to(device)

                        if hidden is not None:
                            hidden[0] = hidden[0].detach()
                            hidden[1] = hidden[1].detach()
                            hidden[2] = hidden[2].detach()

                        pred_class_seq, pred_instance_seq, pred_depth_seq = model(initial_guess_mini, hidden)

                        # Round pred_instance_seq to integer ids
                        pred_instance_seq = pred_instance_seq.detach().cpu().numpy()
                        pred_instance_seq = np.rint(pred_instance_seq).astype(int)

                        # Transform pred_class_seq to maximum class id
                        pred_class_seq = torch.argmax(pred_class_seq, dim=3)

                        # Construct pred_instance_dict_seq as dictionary of masks for each frame & Match class Ids to instance id
                        pred_instance_dict_seq = []
                        pred_instance2class_seq = []
                        for frame_idx in range(len(entire_instance_seq)):
                            pred_instance_dict = {}
                            pred_instance2class = {}
                            pred_class = pred_class_seq[frame_idx, :, :].detach().cpu().numpy()
                            pred_class_revised = np.zeros_like(pred_class)
                            for instance_id in np.unique(entire_instance_seq[frame_idx, :, :]):
                                pred_instance_dict[instance_id] = (entire_instance_seq[frame_idx, :, :] == instance_id).astype(int)
                                pred_instance2class[instance_id] = stats.mode(np.multiply(pred_instance_dict[instance_id], pred_class))
                                pred_class_revised += pred_instance2class[instance_id] * pred_instance_dict[instance_id]
                            pred_class_seq[frame_idx, :, :] = pred_class_revised
                            pred_instance_dict_seq.append(pred_instance_dict)
                            pred_instance2class_seq.append(pred_instance2class)
                        

                        # Compute validation losses
                        # MSE Loss for Depth Prediction
                        class_seq_mini = torch.from_numpy(class_seq[idx : ldx]).long().to(device)
                        class_seq_mini = F.one_hot(class_seq_mini, num_classes=3).float()
                        class_seq_mini = class_seq_mini.to(device)
        
                        depth_seq_mini = torch.from_numpy(depth_seq[idx : ldx]).float().to(device)

                        mse_loss += F.mse_loss(pred_depth_seq, depth_seq_mini)

                        # Cross-Entropy Loss for Class Prediction
                        cross_entropy_loss += F.cross_entropy(pred_class_seq, class_seq)

                        # Calculate MOTSA for Instance Prediction
                        for frame_idx in range(len(pred_instance_seq)):
                            gt_instances = entire_instance_seq[frame_idx]
                            pred_instances = pred_instance_seq[frame_idx]
                            last_matches = []
                            motsa = calculate_motsa(gt_instances, pred_instances, last_matches)  # Custom function to calculate MOTSA for the frame
                            total_motsa += motsa
                            total_frames += 1
                
                val_mse_loss += mse_loss.item()
                val_cross_entropy_loss += cross_entropy_loss.item()

            # Average the losses and MOTSA score
            avg_mse_loss = val_mse_loss / len(val_root_list)
            avg_cross_entropy_loss = val_cross_entropy_loss / len(val_root_list)
            avg_motsa = total_motsa / total_frames if total_frames > 0 else 0.0

            print(f"Validation - Epoch [{epoch+1}/{n_epochs}], Depth MSE Loss (Depth): {avg_mse_loss:.4f}, Class Cross-Entropy Loss (Class): {avg_cross_entropy_loss:.4f}, Instance MOTSA: {avg_motsa:.4f}")

            # Save model if 2 of 3 improve
            if (avg_mse_loss < min_mse and avg_cross_entropy_loss < min_cross_entropy) \
            or (avg_cross_entropy_loss < min_cross_entropy and avg_motsa > max_motsa) \
            or (avg_motsa > max_motsa and avg_mse_loss < min_mse):
                model_save_path = os.path.join("models", f"model_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved at {model_save_path}")

            model.train()

