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
from Motion_estimator import Image_depth, Track_image_sequence, Construct_initial_guess, GT2DetectID


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


        instance_dict = {}
        for obj_id in np.unique(MOTS_map):
            if obj_id in [0, 10000]: continue
            instance_dict[obj_id%1000] = (MOTS_map == obj_id).astype(int)
        batched_instance_seq.append(instance_dict)

        # Read Depth gt
        depth_map = cv2.imread(batched_depth_root[i], -1) / 256.0
        batched_depth_seq.append(depth_map)

    return batched_input_seq, batched_class_seq, batched_instance_seq, batched_depth_seq


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
        class_map[class_map == 10] = 0 # set unknown as background
        entire_class_seq.append(class_map)

        instance_dict = {}
        for obj_id in np.unique(MOTS_map):
            if obj_id in [0, 10000]: continue
            instance_dict[obj_id%1000] = (MOTS_map == obj_id).astype(int)
        entire_instance_seq.append(instance_dict)

        # Read Depth gt
        depth_map = cv2.imread(depth_path, -1) / 256.0
        entire_depth_seq.append(depth_map)

    return entire_input_seq, entire_class_seq, entire_instance_seq, entire_depth_seq


def train_model(model, YOLO_model, depth_model, criterion, optimizer, train_root_list, val_root_list, n_epochs, device, max_size=30, pre_trained_path = None):
    # train/val_root_list = [[raw_root, mots_root, depth_root, start_index, last_index, camera_intrinsic], ...]
    if pre_trained_path is not None:
        model.load_state_dict(torch.load(pre_trained_path))

    model.train()
    model.to(device)
    criterion.to(device)

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        total_frames = 0
        # Train for each video sequence
        for raw_root, mots_root, depth_root, s_idx, l_idx, cam_int in tqdm(train_root_list,desc=f"Epoch [{epoch+1}/{n_epochs}]"):
            batched_input_root_seq, batched_MOTS_root_seq, batched_depth_root_seq, batch_breaks = produce_batch_root(raw_root, mots_root, depth_root, s_idx, l_idx)

            # Divide video sequence into batches of different length
            for batch in range(len(batched_input_root_seq)):
                batched_input_root = batched_input_root_seq[batch] # List of input roots for batch
                batched_MOTS_root = batched_MOTS_root_seq[batch]
                batched_depth_root = batched_depth_root_seq[batch]

                total_frames += len(batched_input_root_seq[batch])

                # Load images for entire batch / List of np.arrays
                batched_input_seq, batched_class_seq, batched_instance_seq, batched_depth_seq = produce_batch_sequence(batched_input_root, batched_MOTS_root, batched_depth_root)
                batched_depth_seq = np.array(batched_depth_seq)

                # Run DeepSORT tracking
                tracker = DeepSort(max_age=30, n_init=0, nn_budget=100)
                batched_bbox_seq = Track_image_sequence(batched_input_seq, YOLO_model, tracker, len(batched_input_root)) # Bounding boxes for entire batch

                # Construct Initial Guess Sequence
                batched_initial_guess_seq = [] # Initial guess list for entire batch
                batched_C2bboxID_seq = []
                for frame in range(len(batched_input_root)):
                    depth_map = Image_depth(batched_input_seq[frame], depth_model, cam_int)
                    batched_initial_guess_seq.append(Construct_initial_guess(batched_input_seq[frame], batched_bbox_seq[frame], depth_map, batched_C2bboxID_seq))
                             
                batched_initial_guess_seq = np.transpose(np.array(batched_initial_guess_seq), (0, 3, 1, 2))              
                batched_instance_gt_seq = GT2DetectID(batched_initial_guess_seq, batched_instance_seq)

                hidden = None
                # Slice batches into mini_batches such that size doesn't exceed max_size
                for idx in range(0, len(batched_initial_guess_seq), max_size):
                    ldx = min(idx + max_size, len(batched_initial_guess_seq))
                    initial_guess_mini = torch.from_numpy(batched_initial_guess_seq[idx : ldx, :, :, :]).float().to(device)

                    if hidden is not None:
                        hidden[0] = hidden[0].detach()
                        hidden[1] = hidden[1].detach()
                        hidden[2] = hidden[2].detach()

                    pred_instance_mini, pred_depth_mini, hidden = model(initial_guess_mini, hidden)

                    # Compute loss in mini_batch
                    instance_mini = torch.from_numpy(batched_instance_gt_seq[idx : ldx, :, :, :]).float().to(device)

                    depth_mini = torch.from_numpy(batched_depth_seq[idx : ldx, :, :]).float().to(device)

                    loss = criterion(pred_instance_mini, instance_mini, pred_depth_mini, depth_mini, initial_guess_mini[:, 1, :, :], device)

                    # Backpropagation
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update Epoch loss
                    epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / total_frames
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_epoch_loss:.4f}")

        max_val_loss = 999999999
        if epoch % 3 == 0:
            val_loss = 0
            model.eval()
            for raw_root, mots_root, depth_root, s_idx, l_idx, cam_int in val_root_list:
                seq_val_loss = 0
                with torch.no_grad():
                    # Produce entire sequence for video
                    entire_input_seq, entire_class_seq, entire_instance_seq, entire_depth_seq = produce_entire_sequence(raw_root, mots_root, depth_root, s_idx, l_idx)

                    # Forward propagate
                    tracker = DeepSort(max_age=30, n_init=0, nn_budget=100)
                    bbox_seq = Track_image_sequence(entire_input_seq, YOLO_model, tracker, len(entire_input_seq)) #List for bboxes for each frame

                    C2bboxID_seq = []
                    initial_guess_seq = [Construct_initial_guess(entire_input_seq[frame], bbox_seq[frame], Image_depth(entire_input_seq[frame], depth_model, cam_int), C2bboxID_seq) for frame in range(len(entire_input_seq))]
                    initial_guess_seq = np.transpose(np.array(initial_guess_seq), (0, 3, 1, 2))
                    class_seq = np.array(entire_class_seq)
                    class_seq = class_seq.astype(np.int16)
                    depth_seq = np.array(entire_depth_seq)

                    tracker = DeepSort(max_age=30, n_init=0, nn_budget=200)
                    bbox_seq = Track_image_sequence(entire_input_seq, YOLO_model, tracker, len(batched_input_root)) # Bounding boxes for entire batch

                    # Construct dict between GT instance ID and bbox instance ID / Modify instance_seq
                    GT2bbox_instance_dict = GT2DetectID(bbox_seq, entire_instance_seq)
                    instance_seq = [{GT2bbox_instance_dict[k]: v for k, v in instance_dict.items()} for instance_dict in entire_instance_seq] # Dict from reassigned instance id to masks

                    # Convert data types of gt
                    instance_gt_seq = []
                    for frame in range(len(entire_input_seq)):
                        instance_gt = np.zeros((11, len(initial_guess_seq[0][0]), len(initial_guess_seq[0][0][0])))
                        instance_gt[0, :, :] = 1
                        C2bboxID = C2bboxID_seq[frame]
                        for channel in C2bboxID.keys():
                            if C2bboxID[channel] is not None:
                                if C2bboxID[channel][0] in instance_seq[frame].keys():
                                    instance_gt[channel, :, :] = instance_seq[frame][C2bboxID[channel][0]]
                                    instance_gt[0, :, :] = instance_gt[0, :, :] - instance_gt[channel, :, :]
                        instance_gt[0, :, :] = np.maximum(instance_gt[0, :, :], 0)
                        instance_gt_seq.append(instance_gt)

                    hidden = None
                    # Slice seq into mini_batches such that size doesn't exceed max_size
                    for idx in range(0, len(entire_input_seq), max_size):
                        ldx = min(idx + max_size, len(initial_guess_seq))
                        initial_guess_mini = torch.from_numpy(initial_guess_seq[idx : ldx, :, :, :]).float().to(device)

                        if hidden is not None:
                            hidden[0] = hidden[0].detach()
                            hidden[1] = hidden[1].detach()
                            hidden[2] = hidden[2].detach()

                        pred_instance_mini, pred_depth_mini, hidden = model(initial_guess_mini, hidden)

                        # Compute loss in mini_batch
                        instance_mini = torch.from_numpy(instance_gt_seq[idx : ldx, :, :, :]).float().to(device)

                        depth_mini = torch.from_numpy(depth_seq[idx : ldx, :, :]).float().to(device)

                        loss = criterion(pred_instance_mini, instance_mini, pred_depth_mini, depth_mini, initial_guess_mini[:, 1, :, :], device)

                        # Update Epoch loss
                        seq_val_loss += loss.item()
                    val_loss += seq_val_loss
                    print(f"Validation - Epoch [{epoch+1}/{n_epochs}], Combined Loss: {seq_val_loss:.4f}")
                if val_loss < max_val_loss:
                    model_save_path = os.path.join("models", f"revised_model_epoch_{epoch}.pth")
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Model saved at {model_save_path} with validation loss of {val_loss:.4f}")
                    max_val_loss = val_loss
        model.train()