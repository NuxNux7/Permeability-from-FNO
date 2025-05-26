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





class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        coord_features: bool = True,

    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.coord_features = coord_features

        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 3

        # Define the U-Net architecture
        self.encoder1 = self.down(1, 32, 3)
        self.encoder2 = self.down(32, 64, 3)
        self.encoder3 = self.down(64, 128, 3)
        self.encoder4 = self.down(128, 256, 3)

        self.bottleneck = self.convX2(256, 512, 3)

        self.decoder1 = self.up(512, 256, 3)
        self.decoder2 = self.up(256+128, 128, 3)
        self.decoder3 = self.up(128+64, 64, 3)
        self.decoder4 = self.up(64+32, 32, 3)

        self.postconv = nn.Conv3d(32+self.in_channels, in_channels, kernel_size=1, padding=0)
        self.final_activation = nn.Sigmoid()

    
    def convX2(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def down(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            self.convX2(in_channels, out_channels, kernel_size),
            nn.MaxPool3d(2),
        )
    
    def up(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            self.convX2(in_channels, out_channels, kernel_size)
        )

    def forward(self, x: Tensor) -> Tensor:

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        # Encoder
        x0 = x
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x = self.encoder4(x3)

        x = self.bottleneck(x)

        x = self.decoder1(x)
        x = torch.concat((x, x3), dim=1)
        x = self.decoder2(x)
        x = torch.concat((x, x2), dim=1)
        x = self.decoder3(x)
        x = torch.concat((x, x1), dim=1)
        x = self.decoder4(x)
        x = torch.concat((x, x0), dim=1)
        x = self.postconv(x)
        x = self.final_activation(x)
        return x

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
    


class UNet2D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        coord_features: bool = True,

    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.coord_features = coord_features

        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 2

        # Define the U-Net architecture
        self.encoder1 = self.down(1, 32, 3)
        self.encoder2 = self.down(32, 64, 3)
        self.encoder3 = self.down(64, 128, 3)
        self.encoder4 = self.down(128, 256, 3)

        self.bottleneck = self.convX2(256, 512, 3)

        self.decoder1 = self.up(512, 256, 3)
        self.decoder2 = self.up(256+128, 128, 3)
        self.decoder3 = self.up(128+64, 64, 3)
        self.decoder4 = self.up(64+32, 32, 3)

        self.postconv = nn.Conv3d(32+self.in_channels, in_channels, kernel_size=1, padding=0)
        self.final_activation = nn.Sigmoid()

    
    def convX2(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def down(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            self.convX2(in_channels, out_channels, kernel_size),
            nn.MaxPool3d(2),
        )
    
    def up(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            self.convX2(in_channels, out_channels, kernel_size)
        )

    def forward(self, x: Tensor) -> Tensor:

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        # Encoder
        x0 = x
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x = self.encoder4(x3)

        x = self.bottleneck(x)

        x = self.decoder1(x)
        x = torch.concat((x, x3), dim=1)
        x = self.decoder2(x)
        x = torch.concat((x, x2), dim=1)
        x = self.decoder3(x)
        x = torch.concat((x, x1), dim=1)
        x = self.decoder4(x)
        x = torch.concat((x, x0), dim=1)
        x = self.postconv(x)
        x = self.final_activation(x)
        return x

    def meshgrid(self, shape: List[int], device: torch.device):
        bsize, size_x, size_y = shape[0], shape[2], shape[3]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        return torch.cat((grid_x, grid_y), dim=1)
    




class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class UNet3DShubh(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        coord_features: bool = True,

    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.coord_features = coord_features

        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 3

        # Define the U-Net architecture
        self.preconv = nn.Conv3d(self.in_channels, 8, kernel_size=3, padding=1)
        self.encoder1 = self.down(8, 16, 7)
        self.encoder2 = self.down(16, 32, 5)
        self.encoder3 = self.down(32, 64, 5)
        self.encoder4 = self.down(64, 128, 3)
        self.encoder5 = self.down(128, 256, 3)
        #self.encoder6 = self.down(256, 512, 3)

        #self.decoder1 = self.up(512, 256, 3)
        self.decoder2 = self.up(256, 128, 3)
        self.decoder3 = self.up(2*128, 64, 3)
        self.decoder4 = self.up(2*64, 32, 5)
        self.decoder5 = self.up(2*32, 16, 5)
        self.decoder6 = self.up(2*16, 8, 7)
        self.postconv = nn.Conv3d(8+self.in_channels, in_channels, kernel_size=3, padding=1)


    def down(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=2),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(p=0.01),
        )
    
    def up(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=1),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:


        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        # Encoder
        x0 = x
        x = self.preconv(x)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
        #x6 = self.encoder6(x5)

        # Decoder
        #x = self.decoder1(x6)
        #x = torch.concat((x, x5), dim=1)
        x = self.decoder2(x5)
        x = torch.concat((x, x4), dim=1)
        x = self.decoder3(x)
        x = torch.concat((x, x3), dim=1)
        x = self.decoder4(x)
        x = torch.concat((x, x2), dim=1)
        x = self.decoder5(x)
        x = torch.concat((x, x1), dim=1)
        x = self.decoder6(x)
        x = torch.concat((x, x0), dim=1)
        x = self.postconv(x)

        return x

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


