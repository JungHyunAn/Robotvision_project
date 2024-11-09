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

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
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
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).

        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''
        if hidden is None:
            hidden = [None] * self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden

    def sequence_forward(self, x_seq, hidden=None):
        '''
        Parameters
        ----------
        x_seq : 5D input tensor. (time, batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).

        Returns
        -------
        output_seq : list of updated hidden representations for each time step.
        final_hidden : list of final hidden states for each layer.
        '''
        seq_len = x_seq.size(0)
        if hidden is None:
            hidden = [None] * self.n_layers

        current_hidden = hidden
        output_seq = []

        # Iterate over the sequence
        for t in range(seq_len):
            x_t = x_seq[t]
            current_hidden = self.forward(x_t, current_hidden)
            output_seq.append(current_hidden[-1])  # Append the output of the last layer

        # Stack outputs for each time step
        output_seq = torch.stack(output_seq, dim=0)

        return output_seq, current_hidden


class PreprocessingCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(PreprocessingCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # Downsample by 2
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # Downsample by 2
        self.conv5 = nn.Conv2d(16, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x, indice2 = self.maxpool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x, indice1 = self.maxpool2(x)
        x = F.relu(self.conv5(x))
        return x, indice1, indice2
    

class PostprocessingCNN(nn.Module):
    def __init__(self, input_channels):
        super(PostprocessingCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.maxunpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)  # Upsample by 2
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.maxunpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)  # Upsample by 2
        self.conv_class = nn.Conv2d(16, 3, kernel_size=1)
        self.conv_instance = nn.Conv2d(16, 1, kernel_size=1)
        self.conv_depth = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x, indices1, indices2):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxunpool1(x, indices1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxunpool2(x, indices2)

        class_map = self.conv_class(x)

        instance_map = self.conv_instance(x)

        depth_map = self.conv_depth(x)

        return class_map, instance_map, depth_map


class Pred_model(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_size, GRU_kernel, GRU_layers):
        super(Pred_model, self).__init__()
        self.preCNN = PreprocessingCNN(input_channels, hidden_size)
        self.GRU = ConvGRU(hidden_size, hidden_size, GRU_kernel, GRU_layers)
        self.postCNN = PostprocessingCNN(hidden_size, output_channels)

    def forward(self, x):
        x, indices1, indices2 = self.preCNN(x)
        x = self.GRU(x)
        x = self.postCNN(x, indices1, indices2)
        return x


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, class_pred, class_gt, instance_pred, instance_gt, depth_pred, depth_gt):
        # Cross entropy loss expects class_pred to have raw logits and class_gt to have integer class labels
        # Make sure class_gt is of shape (batch, height, width) with integer values representing class labels
        class_loss = self.cross_entropy_loss(class_pred, class_gt)

        # MSE loss for instance and depth predictions
        instance_loss = self.mse_loss(instance_pred, instance_gt)
        depth_loss = self.mse_loss(depth_pred, depth_gt)
        
        # Total combined loss with weights for different components
        total_loss = 15 * class_loss + 15 * instance_loss + depth_loss
        
        return total_loss
