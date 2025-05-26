# Based on Nvidia Modulus spectral layers
# https://github.com/NVIDIA/modulus/blob/main/modulus/models/layers/spectral_layers.py

# Based on the FNO papers GitHub from Zongyi Li and Daniel Zhengyu Huang
# https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/models/fno.py

# Based on the FFNO papers GitHub from Alasdair Tran, Alexander Mathews, Lexing Xie and Cheng Soon Ong
# https://github.com/alasdairtran/fourierflow/blob/main/fourierflow/modules/factorized_fno/mesh_3d.py


from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpectralConv1d(nn.Module):
    """1D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply, at most floor(N/2) + 1
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.empty(in_channels, out_channels, self.modes1, 2)
        )
        self.reset_parameters()

    def compl_mul1d(
        self,
        input: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        cweights = torch.view_as_complex(weights)
        return torch.einsum("bix,iox->box", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        bsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            bsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1],
            self.weights1,
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)


class SpectralConv2d(nn.Module):
    """2D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.empty(in_channels, out_channels, self.modes1, self.modes2, 2)
        )
        self.weights2 = nn.Parameter(
            torch.empty(in_channels, out_channels, self.modes1, self.modes2, 2)
        )
        self.reset_parameters()

    def compl_mul2d(self, input: Tensor, weights: Tensor) -> Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        cweights = torch.view_as_complex(weights)
        #cweights = weights
        return torch.einsum("bixy,ioxy->boxy", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * torch.rand(self.weights2.data.shape)


class SpectralConv3d(nn.Module):
    """3D Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    modes3 : int
        Number of Fourier modes to multiply in third dimension, at most floor(N/2) + 1
    """

    def __init__(
        self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.weights2 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.weights3 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.weights4 = nn.Parameter(
            torch.empty(
                in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2
            )
        )
        self.reset_parameters()

    def compl_mul3d(
        self,
        input: Tensor,
        weights: Tensor,
    ) -> Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
        cweights = torch.view_as_complex(weights)
        # = weights
        return torch.einsum("bixyz,ioxyz->boxyz", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * torch.rand(self.weights2.data.shape)
        self.weights3.data = self.scale * torch.rand(self.weights3.data.shape)
        self.weights4.data = self.scale * torch.rand(self.weights4.data.shape)



# ==========================================
# Factorized Spectral Convolution NEW!
# ==========================================


class FactorizedSpectralConv2d(nn.Module):
    """Functional 2D Fourier layer. It does FFT for each direction, linear transform, addition and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    weights : List[nn.ParameterList]
        Used for weight sharing
    """

    def __init__(
        self, in_channels: int, out_channels: int,
        modes1: int, modes2: int,
        weights: List[nn.ParameterList],
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2

        self.dir = ["iox", "ioy"]

        self.scale = 1 / (in_channels * out_channels)

        # set weights
        if weights is None:
            self.weights = nn.ParameterList([])

            for mode in [self.modes1, self.modes2]:
                weight = torch.empty((self.in_channels, self.out_channels, mode, 2), dtype=torch.float32)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.weights.append(param)

        else:
            self.weights = weights


    def compl_mul1d_2d(
        self,
        input: Tensor,
        weights: Tensor,
        dim: int
    ) -> Tensor:
        """Complex multiplication in 1d of 3d

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor
        dim: int
            Direction of multiplication

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channel, x, y), (in_channel, out_channel, x / y) -> (batch, out_channel, x, y)

        calculation = "bixy," + self.dir[dim] + "->boxy"

        cweights = torch.view_as_complex(weights)

        return torch.einsum(calculation, input, cweights)


    def forward(self, x: Tensor) -> Tensor:
        batchsize, I, nx, ny = x.shape

        # Dimension Y
        x_fty = torch.fft.rfftn(x, dim=-1, norm='ortho')

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            nx,
            ny // 2 + 1,
            dtype=x_fty.dtype,
            device=x.device,
        )

        out_ft[:, :, :, : self.modes2] = self.compl_mul1d_2d(
            x_fty[:, :, :, : self.modes2], self.weights[1],
            dim=1
        )

        xy = torch.fft.irfft(out_ft, n=ny, dim=-1, norm='ortho')


        # Dimension X
        x_ftx = torch.fft.rfftn(x, dim=-2, norm='ortho')

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            nx // 2 + 1,
            ny,
            dtype=x_ftx.dtype,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, :] = self.compl_mul1d_2d(
            x_ftx[:, :, : self.modes1, :], self.weights[0],
            dim=0
        )

        xx = torch.fft.irfft(out_ft, n=nx, dim=-2, norm='ortho')

        # Combine Dimensions
        x = xx + xy

        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""

        for weight in self.weights:
            nn.init.xavier_normal_(weight)


class FactorizedSpectralConv3d(nn.Module):
    """Functional 3D Fourier layer. It does FFT for each direction, linear transform, addition and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    modes3 : int
        Number of Fourier modes to multiply in third dimension, at most floor(N/2) + 1
    weights : List[nn.ParameterList]
        Used for weight sharing
    """

    def __init__(
        self, in_channels: int, out_channels: int,
        modes1: int, modes2: int, modes3: int,
        weights: List[nn.ParameterList],
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.dir = ["iox", "ioy", "ioz"]

        self.scale = 1 / (in_channels * out_channels)

        # set weights
        if weights is None:
            self.weights = nn.ParameterList([])

            for mode in [self.modes1, self.modes2, self.modes3]:
                weight = torch.empty((self.in_channels, self.out_channels, mode, 2), dtype=torch.float32)
                #weight = torch.empty((self.in_channels, self.out_channels, mode), dtype=torch.complex64)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.weights.append(param)

        else:
            self.weights = weights


    def compl_mul1d_3d(
        self,
        input: Tensor,
        weights: Tensor,
        dim: int
    ) -> Tensor:
        """Complex multiplication in 1d of 3d

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor
        dim: int
            Direction of multiplication

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channel, x, y, z), (in_channel, out_channel, x / y / z) -> (batch, out_channel, x, y, z)
        cast = (input.dtype == torch.complex32)
        if cast:
            input = input.type(torch.complex64)

        calculation = "bixyz," + self.dir[dim] + "->boxyz"
        cweights = torch.view_as_complex(weights)

        #result = contract(calculation, input, cweights)
        result = torch.einsum(calculation, input, cweights)
        if cast:
            return result.type(torch.complex32)
        else:
            return result
        

    def myEinsum(self, input, weights, dim):

        cweights = torch.view_as_complex(weights)

        if dim == 0:
            input = input.permute(0, 3, 4, 2, 1).unsqueeze_(-2)
        elif dim == 1:
            input = input.permute(0, 2, 4, 3, 1).unsqueeze_(-2)
        elif dim == 2:
            input = input.permute(0, 2, 3, 4, 1).unsqueeze_(-2)
        else:
            raise ValueError("dim must be 0, 1, or 2")
    
    
        cweights = cweights.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        output = torch.matmul(input, cweights)
    
        if dim == 0:
            return output.squeeze_(-2).permute(0, 4, 3, 1, 2)
        elif dim == 1:
            return output.squeeze_(-2).permute(0, 4, 1, 3, 2)
        elif dim == 2:
            return output.squeeze_(-2).permute(0, 4, 1, 2, 3)
    

    def forward(self, x: Tensor) -> Tensor:
        batchsize, I, nx, ny, nz = x.shape

        # Compute Fourier coeffcients up to factor of e^(- something constant)

        # Dimesion Z
        x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            nx,
            ny,
            nz // 2 + 1,
            dtype=x_ftz.dtype,
            device=x.device,
        )

        out_ft[:, :, :, :, : self.modes3] = self.compl_mul1d_3d(
            x_ftz[:, :, :, :, : self.modes3], self.weights[2],
            2
        )

        xz = torch.fft.irfft(out_ft, n=nz, dim=-1, norm='ortho')


        # Dimension Y
        x_fty = torch.fft.rfft(x, dim=-2, norm='ortho')

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            nx,
            ny // 2 + 1,
            nz,
            dtype=x_fty.dtype,
            device=x.device,
        )

        out_ft[:, :, :, : self.modes2, :] = self.compl_mul1d_3d(
            x_fty[:, :, :, : self.modes2, :], self.weights[1],
            1
        )

        xy = torch.fft.irfft(out_ft, n=ny, dim=-2, norm='ortho')


        # Dimension X
        x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            nx // 2 + 1,
            ny,
            nz,
            dtype=x_ftx.dtype,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, :, :] = self.compl_mul1d_3d(
            x_ftx[:, :, : self.modes1, :, :], self.weights[0],
            0
        )

        xx = torch.fft.irfft(out_ft, n=nx, dim=-3, norm='ortho')

        # Combine Dimensions
        x = xx + xy + xz

        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""

        for weight in self.weights:
            nn.init.xavier_normal_(weight)