import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init



class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size, device=input_.device)

        # data size is [channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=0)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=0)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):

    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvGRU, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes] * n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes] * n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i - 1]

            cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells


    def forward(self, x, hidden=None):
        '''
        Parameters
        ----------
        x : 4D input tensor. (time, channels, height, width).
        hidden : list of 3D initial hidden states, one for each layer. Each hidden state is of shape (channels, height, width).

        Returns
        -------
        hidden_seq : 4D hidden representation. (time, channels, height, width).
        '''
        seq_len, _, height, width = x.size()

        # If no hidden state is provided, initialize it with zeros for each layer
        if hidden is None:
            hidden = [torch.zeros(self.hidden_sizes[i], height, width, device=x.device)
                      for i in range(self.n_layers)]

        # Iterate through the sequence
        hidden_seq = []
        for t in range(seq_len):
            input_ = x[t, :, :, :]
            new_hidden = []
            for layer_idx in range(self.n_layers):
                hidden[layer_idx] = self.cells[layer_idx](input_, hidden[layer_idx])
                input_ = hidden[layer_idx]
                new_hidden.append(hidden[layer_idx])
            hidden = new_hidden
            hidden_seq.append(hidden[-1])

        # Stack hidden states along the time dimension
        if len(hidden_seq) > 0:
            hidden_seq = torch.stack(hidden_seq, dim=0)
        else:
            print("Warning: hidden_seq is empty")
            return None  # Or handle the empty case appropriately


        return hidden_seq, hidden


class PreprocessingCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(PreprocessingCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # Downsample by 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # Downsample by 2
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # Downsample by 2
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # Downsample by 2
        self.conv8 = nn.Conv2d(256, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x1 = F.relu(self.conv2(x1))
        size1 = x1.size()
        x2, indice1 = self.maxpool1(x1)

        x2 = F.relu(self.conv3(x2))
        x2 = F.relu(self.conv4(x2))
        size2 = x2.size()
        x3, indice2 = self.maxpool2(x2)

        x3 = F.relu(self.conv5(x3))
        x3 = F.relu(self.conv6(x3))
        size3 = x3.size()
        x, indice3 = self.maxpool3(x3)

        x = F.relu(self.conv7(x))
        size4 = x.size()
        x, indice4 = self.maxpool4(x)

        x = F.relu(self.conv8(x))

        return x, x1, x2, x3, indice1, size1, indice2, size2, indice3, size3, indice4, size4


class PostprocessingCNN(nn.Module):
    def __init__(self, input_channels):
        super(PostprocessingCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=3, stride=1, padding=1)
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)  # Upsample by 2
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.maxunpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.maxunpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.maxunpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)

        self.conv8 = nn.Conv2d(input_channels, 256, kernel_size=3, stride=1, padding=1)
        self.maxunpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)  # Upsample by 2
        self.conv9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.maxunpool6 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv10 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.maxunpool7 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv12 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.maxunpool8 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv14 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)

        self.conv_instance = nn.Conv2d(32, 11, kernel_size=1)
        self.conv_depth = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, x1, x2, x3, indice1, size1, indice2, size2, indice3, size3, indice4, size4):
        x_inst = F.relu(self.conv1(x)) # (256, h/16, w/16)
        x_inst = self.maxunpool1(x_inst, indice1, output_size=size1)
        
        x_inst = F.relu(self.conv2(x_inst)) # (128, h/8, w/8)
        x_inst = self.maxunpool2(x_inst, indice2, output_size=size2)

        x_inst = torch.cat((x_inst, x1), 1) # (256, h/4, w/4)
        x_inst = F.relu(self.conv3(x_inst)) # (128, h/4, w/4)
        x_inst = F.relu(self.conv4(x_inst)) # (64, h/4, w/4)
        x_inst = self.maxunpool3(x_inst, indice3, output_size=size3)

        x_inst = torch.cat((x_inst, x2), 1) # (128, h/2, w/2)
        x_inst = F.relu(self.conv5(x_inst)) # (64, h/2, w/2)
        x_inst = F.relu(self.conv6(x_inst)) # (32, h/2, w/2)
        x_inst = self.maxunpool4(x_inst, indice4, output_size=size4)

        x_inst = torch.cat((x_inst, x3), 1) # (64, h, w)
        x_inst = F.relu(self.conv7(x_inst)) # (32, h, w)
        instance_map = self.conv_instance(x_inst) # (11, h, w)
        #instance_map = instance_map.permute(0, 2, 3, 1)

        x_depth = F.relu(self.conv8(x)) # (256, h/16, w/16)
        x_depth = self.maxunpool5(x_depth, indice1, output_size=size1)
        
        x_depth = F.relu(self.conv9(x_depth)) # (128, h/8, w/8)
        x_depth = self.maxunpool6(x_depth, indice2, output_size=size2)

        x_depth = torch.cat((x_depth, x1), 1) # (256, h/4, w/4)
        x_depth = F.relu(self.conv10(x_depth)) # (128, h/4, w/4)
        x_depth = F.relu(self.conv11(x_depth)) # (64, h/4, w/4)
        x_depth = self.maxunpool7(x_depth, indice3, output_size=size3)

        x_depth = torch.cat((x_depth, x2), 1) # (128, h/2, w/2)
        x_depth = F.relu(self.conv12(x_depth)) # (64, h/2, w/2)
        x_depth = F.relu(self.conv13(x_depth)) # (32, h/2, w/2)
        x_depth = self.maxunpool8(x_depth, indice4, output_size=size4)

        x_depth = torch.cat((x_depth, x3), 1) # (64, h, w)
        x_depth = F.relu(self.conv14(x_depth)) # (32, h, w)

        depth_map = self.conv_depth(x_depth)
        depth_map = torch.squeeze(depth_map, dim=1) # (h, w)

        return instance_map, depth_map


class Pred_model(nn.Module):
    def __init__(self, input_channels=13, hidden_size=256, GRU_kernel=3, GRU_layers=1):
        super(Pred_model, self).__init__()
        self.preCNN = PreprocessingCNN(input_channels, hidden_size)
        self.GRU = ConvGRU(hidden_size, hidden_size, GRU_kernel, GRU_layers)
        self.postCNN = PostprocessingCNN(hidden_size)

    def forward(self, x, hidden=None):
        '''
        Input:
        x : 4D input tensor with shape (time, channels, height, width)
        hidden : Final hidden state for CGRU

        Output:
        instance_seq : 4D class ID output tensor with shape (time, height, width, instanceID channel)
        depth_seq : 3D class ID output tensor with shape (time, height, width)
        '''
        x, x1, x2, x3, indice1, size1, indice2, size2, indice3, size3, indice4, size4 = self.preCNN(x)
        x, hidden = self.GRU(x, hidden)
        instance_seq, depth_seq = self.postCNN(x, x3, x2, x1, indice4, size4, indice3, size3, indice2, size2, indice1, size1)
        return instance_seq, depth_seq, hidden


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def dice_loss(self, pred, target, smooth=1e-6):
        """
        Dice Loss for multiple labels without using one-hot encoding.
        Arguments:
            pred: Tensor of shape (batch, num_classes, height, width), containing the predicted softmax scores for each class.
            target: Tensor of shape (batch, num_classes, height, width), containing the ground truth labels for each pixel as integer values.
            smooth: A small value to avoid division by zero.
        Returns:
            Dice loss value averaged across all classes.
        """
        # Apply softmax to the predictions to get probabilities
        pred_softmax = F.softmax(pred, dim=1)  # Shape: (batch, num_classes, height, width)

        # Create a mask for each class by comparing the target to each class index
        # For each class, target_mask has the shape (batch, height, width) and values are either 0 or 1
        batch_size, num_classes, height, width = pred.shape

        # Initialize variables for calculating the Dice loss for each class
        dice_loss_total = 0.0

        # Loop through each class and calculate the Dice loss
        for class_idx in range(num_classes):
            # Create a binary mask for the current class in the target
            target_mask = target[:, class_idx, :, :]  # Shape: (batch, height, width)

            # Extract the prediction for the current class
            pred_class = pred_softmax[:, class_idx, :, :]  # Shape: (batch, height, width)

            # Compute the intersection and union for the current class
            intersection = (pred_class * target_mask).sum(dim=(1, 2))
            pred_sum = pred_class.sum(dim=(1, 2))
            target_sum = target_mask.sum(dim=(1, 2))

            # Compute Dice coefficient for the current class and average over the batch
            dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
            dice_loss_total += (1 - dice).mean()

        # Average Dice loss across all classes
        dice_loss_avg = dice_loss_total / num_classes

        return dice_loss_avg

    def forward(self, instance_pred, instance_gt, depth_pred, depth_gt, depth_metric3d, device):
        # Cross entropy loss expects class_pred to have raw logits and class_gt to have integer class labels
        # Make sure class_gt is of shape (batch, height, width) with integer values representing class labels
        assert instance_pred.shape == instance_gt.shape, f'instances has shape of {instance_pred.shape}, {instance_gt.shape}'
        assert depth_pred.shape == depth_gt.shape
        assert instance_pred.dim() == 4, f'instance_pred has shape of {instance_pred.shape}'
        assert depth_pred.dim() == 3, f'depth_pred has shape of {depth_pred.shape}'

        batch_size, num_classes, height, width = instance_pred.shape

        # Flatten ground truth to calculate pixel count per class
        instance_gt_flat = torch.argmax(instance_gt, dim=1).view(-1).long()
        class_counts = torch.bincount(instance_gt_flat, minlength=num_classes)

        class_weights = []
        for i in range(num_classes):
            if i == 0:
                class_weights.append(1)
            elif class_counts[i] == 0:
                class_weights.append(1)
            else:
                class_weights.append(float((class_counts[0]/class_counts[i]).item()))

        class_weights = torch.from_numpy(np.array(class_weights)).to(device)
        # Create the CrossEntropyLoss with weights
        cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        instance_loss_ce = cross_entropy_loss(instance_pred, instance_gt)
        intance_loss_dice = self.dice_loss(instance_pred, instance_gt)
        instance_loss = 0.5*instance_loss_ce + 50*intance_loss_dice

        # Mask the depth predictions and ground truth based on the condition depth_gt > 1e-8
        mask = depth_gt > 1e-8
        depth_pred_masked = depth_pred[mask]
        depth_gt_masked = depth_gt[mask]

        mask_comp = depth_gt <= 1e-8
        depth_pred_masked_comp = depth_pred[mask_comp]
        depth_metric3d_masked_comp = depth_metric3d[mask_comp]
        # Calculate MSE loss only for the masked values
        if depth_gt_masked.numel() > 0:  # Check if there are valid elements to avoid division by zero
            depth_loss = self.mse_loss(depth_pred_masked, depth_gt_masked) + 0.2*self.mse_loss(depth_pred_masked_comp, depth_metric3d_masked_comp)
        else:
            depth_loss = torch.tensor(0.0, device=depth_gt.device)

        # Total combined loss with weights for different components
        total_loss = 50*instance_loss + depth_loss
        print(instance_loss_ce.item(), intance_loss_dice.item(), depth_loss.item(), total_loss.item())

        return total_loss