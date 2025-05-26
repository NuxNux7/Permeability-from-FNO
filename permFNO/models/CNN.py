# Based on Nvidia Modulus FNO functions
# https://github.com/NVIDIA/modulus/blob/main/modulus/models/fno/fno.py

# Based on the FNO papers GitHub from Zongyi Li and Daniel Zhengyu Huang
# https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/models/fno.py

# Based on the FFNO papers GitHub from Alasdair Tran, Alexander Mathews, Lexing Xie and Cheng Soon Ong
# https://github.com/alasdairtran/fourierflow/blob/main/fourierflow/modules/factorized_fno/mesh_3d.py

from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

import logging


logger = logging.getLogger(__name__)



class CNN3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        coord_features: bool = True,
        use_weight_norm: bool = False,

    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.coord_features = coord_features
        self.use_weight_norm = use_weight_norm

        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 3

       # Inception Module
        self.conv1_7x7 = nn.Conv3d(self.in_channels, 16, kernel_size=7, stride=2, padding=3)
        self.conv2_7x7 = nn.Conv3d(16, 16, kernel_size=7, stride=2, padding=3)
        
        self.conv1_15x15 = nn.Conv3d(self.in_channels, 16, kernel_size=15, stride=2, padding=7)
        self.conv2_15x15 = nn.Conv3d(16, 16, kernel_size=15, stride=2, padding=7)
        
        self.inception_bn = nn.BatchNorm3d(32)
        self.inception_pool = nn.Conv3d(32, 16, kernel_size=2, stride=2, padding=0)
        
        # Deep Learning Module
        self.deep_conv1 = nn.Conv3d(16, 32, kernel_size=5, stride=1, padding=2)
        self.deep_conv2 = nn.Conv3d(32, 32, kernel_size=5, stride=1, padding=2)
        self.deep_bn = nn.BatchNorm3d(32)
        self.deep_spatial_dropout = nn.Dropout3d(0.1)
        self.deep_pool = nn.Conv3d(32, 32, kernel_size=2, stride=2, padding=0)
        
        # Regression Module
        self.fc1 = nn.Linear(32 * 8 * 4 * 4, 128)  # Adjust size based on input dimensions
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)
    
        # Inception Module
        path1 = F.relu(self.conv1_7x7(x))
        path1 = F.relu(self.conv2_7x7(path1))
        
        path2 = F.relu(self.conv1_15x15(x))
        path2 = F.relu(self.conv2_15x15(path2))
        
        # Concatenate both paths
        inception_out = torch.cat([path1, path2], dim=1)
        inception_out = self.inception_bn(inception_out)
        inception_out = F.relu(self.inception_pool(inception_out))
        
        # Deep Learning Module
        deep_out = F.relu(self.deep_conv1(inception_out))
        deep_out = F.relu(self.deep_conv2(deep_out))
        deep_out = self.deep_bn(deep_out)
        deep_out = self.deep_spatial_dropout(deep_out)
        deep_out = F.relu(self.deep_pool(deep_out))
        
        # Flatten for Regression Module
        flattened = torch.flatten(deep_out, start_dim=1)
        
        # Regression Module
        reg_out = F.relu(self.fc1(flattened))
        reg_out = self.dropout1(reg_out)
        reg_out = F.relu(self.fc2(reg_out))
        reg_out = self.fc3(reg_out)
        
        return reg_out

    def meshgrid(self, shape: List[int], device: torch.device):
        bsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        return torch.cat((grid_x, grid_y, grid_z), dim=1)
    



