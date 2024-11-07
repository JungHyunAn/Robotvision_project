import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from tqdm import tqdm
from Motion_estimator import Image_depth, Track_image_sequence, Construct_initial_guess


def train_convgru(model, tracker, depth_model, criterion, optimizer, train_root_list, val_root_list, n_epochs, device):
    # _root_list = [[train_root, start_index, last_index, camera_intrinsic], ...]
    
    model.train()
    model.to(device)

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for train_root, s_idx, l_idx, cam_int in tqdm(train_root_list,desc=f"Epoch [{epoch+1}/{n_epochs}]"):
            entire_input_seq = []





        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch [{epoch+1}/{n_epochs}]")
            for batch_idx, (input_seq, target_seq) in enumerate(tepoch):
                # Move data to the device
                input_seq, target_seq = input_seq.to(device), target_seq.to(device)

                # Initialize the hidden state
                hidden = None

                # Forward pass through the model
                output_seq, _ = model.sequence_forward(input_seq, hidden)

                # Compute the loss
                loss = criterion(output_seq, target_seq)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update epoch loss
                epoch_loss += loss.item()

                # Update tqdm progress bar
                tepoch.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_epoch_loss:.4f}")

        min_loss = float('inf')


