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
from Train_model import produce_entire_sequence


def val_model(model, YOLO_model, depth_model, val_root_list, device, max_size=30, pre_trained_path = None):
    if pre_trained_path is not None:
        model.load_state_dict(torch.load(pre_trained_path))

    model.eval()
    model.to(device)
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

            initial_guess_seq = [Construct_initial_guess(entire_input_seq[frame], bbox_seq[frame], Image_depth(entire_input_seq[frame], depth_model, cam_int)) for frame in range(len(entire_input_seq))]
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

                pred_class_seq, pred_instance_seq, pred_depth_seq, hidden = model(initial_guess_mini, hidden)

                # Round pred_instance_seq to integer ids
                pred_instance_seq = pred_instance_seq.detach().cpu().numpy()
                pred_instance_seq = np.rint(pred_instance_seq).astype(int)

                # Transform pred_class_seq to maximum class id
                pred_class_seq = torch.argmax(pred_class_seq, dim=3)

                # Construct pred_instance_dict_seq as dictionary of masks for each frame & Match class Ids to instance id
                pred_instance_dict_seq = []
                pred_instance2class_seq = []
                for frame_idx in range(len(pred_instance_seq)):
                    pred_instance_dict = {}
                    pred_instance2class = {}
                    pred_class = pred_class_seq[frame_idx, :, :].detach().cpu().numpy()
                    pred_class_revised = np.zeros_like(pred_class)
                    for instance_id in np.unique(pred_instance_seq[frame_idx, :, :]):
                        if instance_id == 0:
                            continue
                        pred_instance_dict[instance_id] = (pred_instance_seq[frame_idx, :, :] == instance_id).astype(int)
                        mask_flattened = np.multiply(pred_instance_dict[instance_id], pred_class).ravel()
                        pred_instance2class[instance_id] = stats.mode(mask_flattened[np.nonzero(mask_flattened)])
                        pred_class_revised += pred_instance2class[instance_id] * pred_instance_dict[instance_id]
                    pred_class_revised = torch.from_numpy(pred_class_revised).to(device=device, dtype=torch.long)
                    pred_class_seq[frame_idx, :, :] = pred_class_revised
                    pred_instance_dict_seq.append(pred_instance_dict)
                    pred_instance2class_seq.append(pred_instance2class)
                

                # Compute validation losses
                # MSE Loss for Depth Prediction
                class_seq_mini = torch.from_numpy(class_seq[idx : ldx]).long().to(device)
                class_seq_mini = F.one_hot(class_seq_mini, num_classes=3).float()
                class_seq_mini = class_seq_mini.to(device)

                pred_class_seq = F.one_hot(pred_class_seq, num_classes=3).float()

                depth_seq_mini = torch.from_numpy(depth_seq[idx : ldx]).float().to(device)

                mse_loss += F.mse_loss(pred_depth_seq, depth_seq_mini).item()

                # Cross-Entropy Loss for Class Prediction
                cross_entropy_loss += F.cross_entropy(pred_class_seq, class_seq_mini).item()

                # Calculate MOTSA for Instance Prediction
                for frame_idx in range(len(pred_instance_seq)):
                    gt_instances = entire_instance_seq[frame_idx]
                    pred_instances = pred_instance_dict_seq[frame_idx]
                    last_matches = []
                    motsa = calculate_motsa(gt_instances, pred_instances, last_matches)  # Custom function to calculate MOTSA for the frame
                    total_motsa += motsa
                    total_frames += 1
        
        val_mse_loss += mse_loss
        val_cross_entropy_loss += cross_entropy_loss

    # Average the losses and MOTSA score
    avg_mse_loss = val_mse_loss / len(val_root_list)
    avg_cross_entropy_loss = val_cross_entropy_loss / len(val_root_list)
    avg_motsa = total_motsa / total_frames if total_frames > 0 else 0.0

    print(f"Depth MSE Loss (Depth): {avg_mse_loss:.4f}, Class Cross-Entropy Loss (Class): {avg_cross_entropy_loss:.4f}, Instance MOTSA: {avg_motsa:.4f}")
