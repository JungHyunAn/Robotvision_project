import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
import PIL.Image as Image
import random
import os
import time
from scipy import stats
from torch.nn import init
from tqdm import tqdm
from deep_sort_realtime.deepsort_tracker import DeepSort
from Motion_estimator import Image_depth, Track_image_sequence, Construct_initial_guess, calculate_iou, Instance_eval
from Train_model import produce_entire_sequence


def eval_model(model, YOLO_model, depth_model, val_root_list, device, max_size=15, pre_trained_path = None):
    if pre_trained_path is not None:
        model.load_state_dict(torch.load(pre_trained_path))

    model.eval()
    model.to(device)

    for raw_root, mots_root, depth_root, s_idx, l_idx, cam_int in val_root_list:
        
        initial_guess_depth_rel_sum = 0.0
        depth_rel_sum = 0.0
        iou_number = [0, 0] # car, pedestrian
        initial_iou_sum = [0, 0] # car, pedestrian
        instance_iou_sum = [0, 0] # car, pedestrian

        tracked_initial_iou_sum = [0, 0]
        tracked_instance_iou_sum = [0, 0]

        instance_tracking_sum = [0, 0, 0] # True positive rate, Positive predictive value, Id switch
        time_sum = [0, 0, 0, 0] # DeepSort time / Monodepth time / Initial guess production time / GRU Module time
        total_frames = 0

        with torch.no_grad():
            # Produce entire sequence for video
            entire_input_seq, entire_class_seq, entire_instance_seq, entire_depth_seq = produce_entire_sequence(raw_root, mots_root, depth_root, s_idx, l_idx)

            # Forward propagate
            start = time.time()
            tracker = DeepSort(max_age=30, n_init=0, nn_budget=200)
            bbox_seq = Track_image_sequence(entire_input_seq, YOLO_model, tracker, len(entire_input_seq)) #List for bboxes for each frame
            end = time.time()
            time_sum[0] += (end - start) * 1000 # DeepSort time

            initial_guess_seq = []
            C2bboxID_seq = []
            for frame in range(len(entire_input_seq)):
                start = time.time()
                depth_map = Image_depth(entire_input_seq[frame], depth_model, cam_int)
                end = time.time()
                time_sum[1] += (end - start) * 1000 # Monodepth time

                # Calculate Relative error by monodepth model
                gt_depth_map = entire_depth_seq[frame][entire_depth_seq[frame] > 1e-8]
                initial_guess_depth_rel_sum += np.average(np.abs(depth_map[entire_depth_seq[frame] > 1e-8] - gt_depth_map) / np.abs(gt_depth_map))
                
                start = time.time()
                initial_guess_seq.append(Construct_initial_guess(entire_input_seq[frame], bbox_seq[frame], depth_map, C2bboxID_seq))
                end = time.time()
                time_sum[2] += (end - start) * 1000 # Initial guess production time
            
            hidden = None
            for idx in tqdm(range(0, len(entire_input_seq), max_size), desc="Processing Batches"): # slice entire_input_seq by max_size=30
                ldx = min(idx+max_size, len(entire_input_seq)) # last index of batch
                initial_guess_mini = np.transpose(np.array(initial_guess_seq[idx : ldx]), (0, 3, 1, 2))
                initial_guess_mini = torch.from_numpy(initial_guess_mini).float().to(device) # slice precalculated initial_guess

                if hidden is not None:
                    hidden[0] = hidden[0].detach()

                start = time.time()
                pred_instance_seq, pred_depth_seq, hidden = model(initial_guess_mini, hidden)
                end = time.time()
                time_sum[3] += (end - start) * 1000 # GRU Module time
                
                # Detach initial_guess_mini 
                initial_guess_mini = initial_guess_mini.detach().cpu().numpy()
                # Detach pred_depth_seq
                pred_depth_seq = pred_depth_seq.detach().cpu().numpy()
                # Get single map from pred_instance_seq
                pred_instance_seq = torch.argmax(pred_instance_seq, dim=1)
                pred_instance_seq = pred_instance_seq.detach().cpu().numpy()                
                
                # Calculate each criterion for frame
                for frame in range(len(pred_instance_seq)):
                    gt_depth_map = np.array(entire_depth_seq[idx+frame])

                    pred_depth_map = pred_depth_seq[frame, :, :][gt_depth_map > 1e-8]
                    pred_instance_map = pred_instance_seq[frame, :, :]
                    pred_instance_dict = {i: (pred_instance_map==i).astype(int) for i in range(11)}

                    gt_depth_map = gt_depth_map[gt_depth_map > 1e-8]
                    gt_instance_dict = entire_instance_seq[idx+frame]
                    gt_class_map = np.array(entire_class_seq[idx+frame])

                    # criterion for depth map
                    depth_rel_sum += np.average(np.abs(pred_depth_map - gt_depth_map) / np.abs(gt_depth_map)) 

                    # criterion for istance map
                    n_car = 0
                    n_pedestrian = 0
                    car_iou = 0
                    pedestrian_iou = 0
                    initial_car_iou = 0
                    initial_pedestrian_iou = 0

                    tracked_car_iou = 0
                    tracked_pedestrian_iou = 0
                    tracked_initial_car_iou = 0
                    tracked_initial_pedestrian_iou = 0

                    for gt_mask in gt_instance_dict.values():
                        instance_class = np.sum(np.multiply(gt_class_map, gt_mask)) // np.sum(gt_mask) # class for mask
                        assert instance_class in [1, 2]

                        max_iou = 0
                        max_initial_iou = 0
                        for pred_key, pred_mask in pred_instance_dict.items():
                            iou = calculate_iou(gt_mask, pred_mask)
                            initial_iou = calculate_iou(gt_mask, initial_guess_mini[frame, 2+pred_key])
                            if iou > max_iou:
                                max_iou = iou
                                max_initial_iou = initial_iou
                            if iou > 0.5:
                                if instance_class == 1:
                                    tracked_car_iou += iou
                                    tracked_initial_car_iou += initial_iou
                                else:
                                    tracked_pedestrian_iou += iou
                                    tracked_initial_pedestrian_iou += initial_iou

                        if instance_class == 1:
                            n_car += 1
                            car_iou += max_iou
                            initial_car_iou += max_initial_iou
                        else:
                            n_pedestrian += 1
                            pedestrian_iou += max_iou
                            initial_pedestrian_iou += max_initial_iou                                
                                
                    if n_car != 0:
                        iou_number[0] += 1
                        instance_iou_sum[0] += car_iou / n_car
                        initial_iou_sum[0] +=  initial_car_iou / n_car
                        tracked_instance_iou_sum[0] += tracked_car_iou / n_car
                        tracked_initial_iou_sum[0] += tracked_initial_car_iou / n_car                
                    if n_pedestrian != 0:
                        iou_number[1] += 1
                        instance_iou_sum[1] += pedestrian_iou / n_pedestrian
                        initial_iou_sum[1] += initial_pedestrian_iou / n_pedestrian
                        tracked_instance_iou_sum[1] += tracked_pedestrian_iou / n_pedestrian
                        tracked_initial_iou_sum[1] += tracked_initial_pedestrian_iou / n_pedestrian

                    last_matches = []
                    tpr, pvr, ids = Instance_eval(gt_instance_dict, pred_instance_dict, last_matches)
                    instance_tracking_sum[0] += tpr
                    instance_tracking_sum[1] += pvr
                    instance_tracking_sum[2] += ids
                    total_frames += 1

        print('\nLoss for \"' + raw_root + '\" is as follows')
        print(f'Average depth relative error : {depth_rel_sum/total_frames:.4f}, Metric 3D - {initial_guess_depth_rel_sum/total_frames:.4f}')
        if iou_number[0] != 0 and iou_number[1] != 0:
            print(f'Average initial instance IoU : Background - Car - {initial_iou_sum[0]/iou_number[0]:.4f}, Pedestrian - {initial_iou_sum[1]/iou_number[1]:.4f}')
            print(f'Average final instance IoU : Background - Car - {instance_iou_sum[0]/iou_number[0]:.4f}, Pedestrian - {instance_iou_sum[1]/iou_number[1]:.4f}')
            print(f'Average tracked initial instance IoU : Background - Car - {tracked_initial_iou_sum[0]/iou_number[0]:.4f}, Pedestrian - {tracked_initial_iou_sum[1]/iou_number[1]:.4f}')
            print(f'Average tracked final instance IoU : Background - Car - {tracked_instance_iou_sum[0]/iou_number[0]:.4f}, Pedestrian - {tracked_instance_iou_sum[1]/iou_number[1]:.4f}')
        elif iou_number[0] == 0 and iou_number[1] != 0:
            print(f'Average initial instance IoU : Background - Car - No Cars tracked, Pedestrian - {initial_iou_sum[1]/iou_number[1]:.4f}')
            print(f'Average final instance IoU : Background - Car - No Cars tracked, Pedestrian - {instance_iou_sum[1]/iou_number[1]:.4f}')
            print(f'Average tracked initial instance IoU : Background - Car - No Cars tracked, Pedestrian - {tracked_initial_iou_sum[1]/iou_number[1]:.4f}')
            print(f'Average tracked final instance IoU : Background - Car - No Cars tracked, Pedestrian - {tracked_instance_iou_sum[1]/iou_number[1]:.4f}')
        elif iou_number[0] != 0 and iou_number[1] == 0:
            print(f'Average initial instance IoU : Background - Car - {initial_iou_sum[0]/iou_number[0]:.4f}, Pedestrian - No Pedestrians tracked')
            print(f'Average final instance IoU : Background - Car - {instance_iou_sum[0]/iou_number[0]:.4f}, Pedestrian - No Pedestrians tracked')
            print(f'Average tracked initial instance IoU : Background - Car - {tracked_initial_iou_sum[0]/iou_number[0]:.4f}, Pedestrian - No Pedestrians tracked')
            print(f'Average tracked final instance IoU : Background - Car - {tracked_instance_iou_sum[0]/iou_number[0]:.4f}, Pedestrian - No Pedestrians tracked')
        else:
            print('No instance tracked')
        print(f'Average tracking criterions : TPR - {instance_tracking_sum[0]/total_frames:.4f}, PVR - {instance_tracking_sum[1]/total_frames:.4f}, IDs - {instance_tracking_sum[2]/total_frames:.4f}')
        print(f'Timing results : DeepSORT - {time_sum[0]/total_frames:.4f}ms, Monodepth - {time_sum[1]/total_frames:.4f}ms, Initial guess - {time_sum[2]/total_frames:.4f}ms, GRU Module - {time_sum[3]/total_frames:.4f}ms')
        print('\n')