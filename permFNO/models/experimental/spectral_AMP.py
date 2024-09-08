# Copy of Nvidia Modulus layers + FFNO addition


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


class SpectralConv4d(nn.Module):
    """4D Fourier layer. It does FFT, linear transform, and Inverse FFT.

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
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        modes4: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.weights2 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.weights3 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.weights4 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.weights5 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.weights6 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.weights7 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.weights8 = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                self.modes3,
                self.modes4,
                2,
            )
        )
        self.reset_parameters()

    def compl_mul4d(
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
        return torch.einsum("bixyzt,ioxyzt->boxyzt", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-4, -3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-4),
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # print(f'mod: size x: {x_ft.size()}, out: {out_ft.size()}')
        # print(f'mod: x_ft[weight4]: {x_ft[:, :, self.modes1 :, self.modes2 :, : -self.modes3, :self.modes4].size()} weight4: {self.weights4.size()}')

        out_ft[
            :, :, : self.modes1, : self.modes2, : self.modes3, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3, : self.modes4],
            self.weights1,
        )
        out_ft[
            :, :, -self.modes1 :, : self.modes2, : self.modes3, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3, : self.modes4],
            self.weights2,
        )
        out_ft[
            :, :, : self.modes1, -self.modes2 :, : self.modes3, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3, : self.modes4],
            self.weights3,
        )
        out_ft[
            :, :, : self.modes1, : self.modes2, -self.modes3 :, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, : self.modes1, : self.modes2, -self.modes3 :, : self.modes4],
            self.weights4,
        )
        out_ft[
            :, :, -self.modes1 :, -self.modes2 :, : self.modes3, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3, : self.modes4],
            self.weights5,
        )
        out_ft[
            :, :, -self.modes1 :, : self.modes2, -self.modes3 :, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, -self.modes1 :, : self.modes2, -self.modes3 :, : self.modes4],
            self.weights6,
        )
        out_ft[
            :, :, : self.modes1, -self.modes2 :, -self.modes3 :, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, : self.modes1, -self.modes2 :, -self.modes3 :, : self.modes4],
            self.weights7,
        )
        out_ft[
            :, :, -self.modes1 :, -self.modes2 :, -self.modes3 :, : self.modes4
        ] = self.compl_mul4d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, -self.modes3 :, : self.modes4],
            self.weights8,
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-4), x.size(-3), x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.data = self.scale * torch.rand(self.weights1.data.shape)
        self.weights2.data = self.scale * torch.rand(self.weights2.data.shape)
        self.weights3.data = self.scale * torch.rand(self.weights3.data.shape)
        self.weights4.data = self.scale * torch.rand(self.weights4.data.shape)
        self.weights5.data = self.scale * torch.rand(self.weights5.data.shape)
        self.weights6.data = self.scale * torch.rand(self.weights6.data.shape)
        self.weights7.data = self.scale * torch.rand(self.weights7.data.shape)
        self.weights8.data = self.scale * torch.rand(self.weights8.data.shape)



# ==========================================
# Functional Spectral Convolution NEW!
# ==========================================


class FunctionalSpectralConv2d(nn.Module):
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
                weight = torch.FloatTensor(self.in_channels, self.out_channels, mode, 2)
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
            dtype=torch.cfloat,
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
            dtype=torch.cfloat,
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


class FunctionalSpectralConv3d(nn.Module):
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
                weight = torch.empty((self.in_channels, self.out_channels, mode), dtype=torch.complex64)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.weights.append(param)

        else:
            self.weights = weights


    '''def compl_mul1d_3d(
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

        calculation = "bixyz," + self.dir[dim] + "->boxyz"

        return torch.einsum(calculation, input, weights)'''
    
    def compl_mul1d_3d(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
        dim: int
    ) -> torch.Tensor:
        batch, in_channel, x, y, z = input.shape
        _, out_channel, dim_size = weights.shape

        # Reshape input and weights for matrix multiplication
        if dim == 0:  # x dimension
            input_expanded = input.unsqueeze(1)
            weights_expanded = weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            output = (input_expanded * weights_expanded).sum(dim=2)
        elif dim == 1:  # y dimension
            input_expanded = input.unsqueeze(1)
            weights_expanded = weights.unsqueeze(0).unsqueeze(-2).unsqueeze(-1)
            output = (input_expanded * weights_expanded).sum(dim=2)
        elif dim == 2:  # z dimension
            input_expanded = input.unsqueeze(1)
            weights_expanded = weights.unsqueeze(0).unsqueeze(-2).unsqueeze(-3)
            output = (input_expanded * weights_expanded).sum(dim=2)
        else:
            raise ValueError("Invalid dimension. Must be 0, 1, or 2.")

        return output


    def forward(self, x: Tensor) -> Tensor:
        batchsize, I, nx, ny, nz = x.shape

        # Compute Fourier coeffcients up to factor of e^(- something constant)

        # Apply padding for AMP TESTING
        next_power_of_two_nx = int(2 ** torch.ceil(torch.log2(torch.tensor(nx))).item())
        next_power_of_two_ny = int(2 ** torch.ceil(torch.log2(torch.tensor(ny))).item())
        next_power_of_two_nz = int(2 ** torch.ceil(torch.log2(torch.tensor(nz))).item())

        # Calculate the padding sizes for each dimension
        pad_nx = next_power_of_two_nx - nx
        pad_ny = next_power_of_two_ny - ny
        pad_nz = next_power_of_two_nz - nz

        # Pad the tensor (the order of padding is reversed: last dimension first)
        if pad_nx > 0 or pad_ny > 0 or pad_nz > 0:
            # Padding format: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
            x = F.pad(x, (0, pad_nz, 0, pad_ny, 0, pad_nx))

        # Dimesion Z
        x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            next_power_of_two_nx,
            next_power_of_two_ny,
            next_power_of_two_nz // 2,
            dtype=x_ftz.dtype,
            device=x.device,
        )

        #out_ft = x_ftz.new_zeros(batchsize, I, nx, ny, nz // 2 + 1)

        out_ft[:, :, :, :, : self.modes3] = self.compl_mul1d_3d(
            x_ftz[:, :, :, :, : self.modes3], self.weights[2],
            2
        )
        print(out_ft.shape)

        xz = torch.fft.irfft(out_ft, n=next_power_of_two_nz, dim=-1, norm='ortho')


        # Dimension Y
        x_fty = torch.fft.rfft(x, dim=-2, norm='ortho')

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            next_power_of_two_nx,
            next_power_of_two_ny // 2,
            next_power_of_two_nz,
            dtype=x_fty.dtype,
            device=x.device,
        )

        out_ft[:, :, :, : self.modes2, :] = self.compl_mul1d_3d(
            x_fty[:, :, :, : self.modes2, :], self.weights[1],
            1
        )

        xy = torch.fft.irfft(out_ft, n=next_power_of_two_ny, dim=-2, norm='ortho')


        # Dimension X
        x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            next_power_of_two_nx // 2,
            next_power_of_two_ny,
            next_power_of_two_nz,
            dtype=x_ftx.dtype,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, :, :] = self.compl_mul1d_3d(
            x_ftx[:, :, : self.modes1, :, :], self.weights[0],
            0
        )

        xx = torch.fft.irfft(out_ft, n=next_power_of_two_nx, dim=-3, norm='ortho')

        # Combine Dimensions
        x = xx + xy + xz

        #print(x.shape)

        #remove padding
        if pad_nx > 0:
            x = x[..., :nx, :, :]
        if pad_ny > 0:
            x = x[..., :ny, :]
        if pad_nz > 0:
            x = x[..., :nz]

        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""

        for weight in self.weights:
            nn.init.xavier_normal_(weight)