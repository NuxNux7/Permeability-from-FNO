
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

import logging

from .siren import SirenArch
from .feedForward import FeedForwardBlock

#from ..layers.convFC import *
from .layers.spectralLayers import *


logger = logging.getLogger(__name__)

# ===================================================================
# ===================================================================
# Legacy FNO
# ===================================================================
# ===================================================================

class FNO2DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        nr_fno_layers: int = 4,
        nr_ff_blocks: int = 1,
        fno_layer_size: int = 32,
        fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn = nn.GELU(),
        coord_features: bool = True,
        use_weight_norm = False,
    ) -> None:

        super().__init__()
        self.in_channels = in_channels
        self.nr_fno_layers = nr_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.use_weight_norm = use_weight_norm

        # Spectral modes to have weights
        if isinstance(fno_modes, int):
            fno_modes = [fno_modes, fno_modes]
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 2
        self.activation_fn = activation_fn

        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        # Initial lift layer
        self.lift_layer = nn.Conv2d(self.in_channels, self.fno_width, 1)
        if self.use_weight_norm:
            self.lift_layer = weight_norm(self.lift_layer)

        # Build Neural Fourier Operators
        for _ in range(self.nr_fno_layers):
            self.spconv_layers.append(
                SpectralConv2d(
                    self.fno_width, self.fno_width, fno_modes[0], fno_modes[1]
                )
            )
            conv = nn.Conv2d(self.fno_width, self.fno_width, 1)
            if self.use_weight_norm:
                conv = weight_norm(conv)
            self.conv_layers.append(conv)

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding]
        padding = padding + [0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:2]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.dim() == 4
        ), "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D FNO"

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_layer(x)
        # (left, right, top, bottom)
        x = F.pad(x, (0, self.pad[0], 0, self.pad[1]), mode=self.padding_type)
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(
                    conv(x) + w(x)
                )  # Spectral Conv + GELU causes JIT issue!
            else:
                x = conv(x) + w(x)

        # remove padding
        x = x[..., : self.ipad[1], : self.ipad[0]]

        return x

    def meshgrid(self, shape: List[int], device: torch.device):
        bsize, size_x, size_y = shape[0], shape[2], shape[3]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        return torch.cat((grid_x, grid_y), dim=1)


class FNO3DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        nr_fno_layers: int = 4,
        nr_ff_blocks: int = 1,
        fno_layer_size: int = 32,
        fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn = nn.GELU(),
        coord_features: bool = True,
        use_weight_norm = False,

    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.nr_fno_layers = nr_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        self.use_weight_norm = use_weight_norm

        # Spectral modes to have weights
        if isinstance(fno_modes, int):
            fno_modes = [fno_modes, fno_modes, fno_modes]
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 3
        self.activation_fn = activation_fn

        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        # Initial lift layer
        self.lift_layer = nn.Conv3d(self.in_channels, self.fno_width, 1)
        if self.use_weight_norm:
            self.lift_layer = weight_norm(self.lift_layer)

        # Build Neural Fourier Operators
        for _ in range(self.nr_fno_layers):
            self.spconv_layers.append(
                SpectralConv3d(
                    self.fno_width,
                    self.fno_width,
                    fno_modes[0],
                    fno_modes[1],
                    fno_modes[2],
                )
            )

            conv = nn.Conv3d(self.fno_width, self.fno_width, 1)
            if self.use_weight_norm:
                conv = weight_norm(conv)
            self.conv_layers.append(conv)

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding, padding]
        padding = padding + [0, 0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:3]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

    def forward(self, x: Tensor) -> Tensor:

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_layer(x)
        # (left, right, top, bottom, front, back)
        x = F.pad(
            x,
            (0, self.pad[0], 0, self.pad[1], 0, self.pad[2]),
            mode=self.padding_type,
        )
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(
                    conv(x) + w(x)
                )  # Spectral Conv + GELU causes JIT issue!
            else:
                x = conv(x) + w(x)

        x = x[..., : self.ipad[2], : self.ipad[1], : self.ipad[0]]
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


# ===================================================================
# ===================================================================
# 2D/3D Functional FNO NEW!
# ===================================================================
# ===================================================================

class FunctionalFNO2DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        nr_fno_layers: int = 4,
        nr_ff_blocks: int = 2,
        fno_layer_size: int = 32,
        fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn = nn.GELU(),
        coord_features: bool = True,
        weight_sharing: bool = False,
        use_weight_norm: bool = False,
        batch_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.nr_fno_layers = nr_fno_layers
        self.nr_ff_blocks = nr_ff_blocks
        self.fno_width = fno_layer_size

        # Features
        self.coord_features = coord_features
        self.weight_sharing = weight_sharing
        self.use_weight_norm = use_weight_norm
        self.batch_norm = batch_norm
        self.dropout = dropout

        # Spectral modes to have weights
        if isinstance(fno_modes, int):
            fno_modes = [fno_modes, fno_modes, fno_modes]
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 2
        self.activation_fn = activation_fn

        self.spconv_layers = nn.ModuleList()
        self.ff_blocks = nn.ModuleList()
        

        # Initial lift layer
        self.lift_layer = nn.Conv2d(self.in_channels, self.fno_width, kernel_size=1)
        if self.use_weight_norm:
            self.lift_layer = weight_norm(self.lift_layer)

        # Initialize weights
        if self.weight_sharing:
            self.weights = nn.ParameterList([])

            for mode in fno_modes:
                weight = torch.empty((self.in_channels, self.out_channels, mode, 2), dtype=torch.float32)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.weights.append(param)

        else:
            self.weights = None


        # Build Functional Neural Fourier Operators
        for k in range(self.nr_fno_layers):
            self.spconv_layers.append(
                FunctionalSpectralConv2d(
                    self.fno_width,
                    self.fno_width,
                    fno_modes[0],
                    fno_modes[1],
                    self.weights,
                )
            )
            use_dropout = self.dropout
            use_activation_fn = self.activation_fn

            # for last layer no activation and dropout
            '''if k == (self.nr_fno_layers - 1):
                use_dropout = 0.0
                use_activation_fn = None'''
            
            self.ff_blocks.append(FeedForwardBlock(
                2,
                self.fno_width,
                use_activation_fn,
                use_weight_norm,
                self.batch_norm,
                use_dropout,
                self.nr_ff_blocks,
                factor=4))
            
        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding]
        padding = padding + [0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:2]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type


    def forward(self, x: Tensor) -> Tensor:

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_layer(x)
        # (left, right, top, bottom, front, back)
        x = F.pad(
            x,
            (0, self.pad[0], 0, self.pad[1]),
            mode=self.padding_type,
        )
        
        # Spectral layers
        for i in range(self.nr_fno_layers):

            x_spconv = self.spconv_layers[i](x)
            x_out = self.ff_blocks[i](x_spconv)

            x = x + x_out

        x_out = x_out[..., : self.ipad[1], : self.ipad[0]]
        return x_out

    def meshgrid(self, shape: List[int], device: torch.device):
        bsize, size_x, size_y = shape[0], shape[2], shape[3]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        return torch.cat((grid_x, grid_y), dim=1)


class FunctionalFNO3DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        nr_fno_layers: int = 4,
        nr_ff_blocks: int = 2,
        fno_layer_size: int = 32,
        fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn = nn.GELU(),
        coord_features: bool = True,
        weight_sharing: bool = False,
        use_weight_norm: bool = False,
        batch_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.nr_fno_layers = nr_fno_layers
        self.nr_ff_blocks = nr_ff_blocks
        self.fno_width = fno_layer_size

        # Features
        self.coord_features = coord_features
        self.weight_sharing = weight_sharing
        self.use_weight_norm = use_weight_norm
        self.batch_norm = batch_norm
        self.dropout = dropout

        # Spectral modes to have weights
        if isinstance(fno_modes, int):
            fno_modes = [fno_modes, fno_modes, fno_modes]
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 3
        self.activation_fn = activation_fn

        self.spconv_layers = nn.ModuleList()
        self.ff_blocks = nn.ModuleList()


        # Initial lift layer
        self.lift_layer = nn.Conv3d(self.in_channels, self.fno_width, 1)
        if self.use_weight_norm:
            self.lift_layer = weight_norm(self.lift_layer)

        # Initialize weights
        if self.weight_sharing:
            self.weights = nn.ParameterList([])

            for mode in fno_modes:
                weight = torch.empty((self.in_channels, self.out_channels, mode, 2), dtype=torch.float32)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.weights.append(param)

        else:
            self.weights = None


        # Build Functional Neural Fourier Operators
        for k in range(self.nr_fno_layers):
            self.spconv_layers.append(
                FunctionalSpectralConv3d(
                    self.fno_width,
                    self.fno_width,
                    fno_modes[0],
                    fno_modes[1],
                    fno_modes[2],
                    self.weights,
                )
            )

            use_dropout = self.dropout
            use_activation_fn = self.activation_fn

            # for last layer no activation and dropout
            #if k == (self.nr_fno_layers - 1):
                #use_dropout = 0.0
                #use_activation_fn = None
            
            self.ff_blocks.append(FeedForwardBlock(
                3,
                self.fno_width,
                use_activation_fn,
                self.use_weight_norm,
                self.batch_norm,
                use_dropout,
                self.nr_ff_blocks,
                factor=4))

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding, padding]
        padding = padding + [0, 0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:3]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type


    def forward(self, x: Tensor) -> Tensor:

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_layer(x)
        # (left, right, top, bottom, front, back)
        x = F.pad(
            x,
            (0, self.pad[0], 0, self.pad[1], 0, self.pad[2]),
            mode=self.padding_type,
        )
        
        # Spectral layers
        for i in range(self.nr_fno_layers):

            x_spconv = self.spconv_layers[i](x)
            x_out = self.ff_blocks[i](x_spconv)

            x = x + x_out

        x_out = x_out[..., : self.ipad[2], : self.ipad[1], : self.ipad[0]]
        return x_out

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


# ===================================================================
# Helper functions
# ===================================================================

def grid_to_points2d(value: Tensor):
    value = torch.permute(value, (0, 2, 3, 1))
    value = value.reshape(-1, value.size(-1))
    return value


def points_to_grid2d(value: Tensor, shape: List[int]):
    value = value.reshape(shape[0], shape[2], shape[3], value.size(-1))
    value = torch.permute(value, (0, 3, 1, 2))
    return value


def grid_to_points3d(value: Tensor):
    value = torch.permute(value, (0, 2, 3, 4, 1))
    value = value.reshape(-1, value.size(-1))
    return value


def points_to_grid3d(value: Tensor, shape: List[int]):
    value = value.reshape(shape[0], shape[2], shape[3], shape[4], value.size(-1))
    value = torch.permute(value, (0, 4, 1, 2, 3))
    return value



class FNOArch(nn.Module):
    """Fourier neural operator (FNO) model.

    Note
    ----
    The FNO architecture supports options for 1D, 2D and 3D fields which can
    be controlled using the `dimension` parameter.


    Parameters
    ----------
    input_keys : List[Key]
        Input key list. The key dimension size should equal the variables channel dim.
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    decoder_net : Arch
        Pointwise decoder network, input key should be the latent variable
    nr_fno_layers : int, optional
        Number of spectral convolution layers, by default 4
    fno_modes : Union[int, List[int]], optional
        Number of Fourier modes with learnable weights, by default 16
    padding : int, optional
        Padding size for FFT calculations, by default 8
    padding_type : str, optional
        Padding type for FFT calculations ('constant', 'reflect', 'replicate'
        or 'circular'), by default "constant"
    activation_fn : Activation, optional
        Activation function, by default Activation.GELU
    coord_features : bool, optional
        Use coordinate meshgrid as additional input feature, by default True
    functional: bool, optional
        Use a residual, feed-forward setup coupled with dinstinc fft for each dimension, by default False
    weight_sharing: bool, optional
        Use the same weights in every fourier layer. Usefull for deep F-FNO nets, by default False
    batch_norm: bool, optional
        Use batch normalization layer after convolution, by default Ffalse
    dropout: bool, optional
        Add a dropout layer after linear layer, by default False

    Variable Shape
    --------------
    Input variable tensor shape:

    - 1D: :math:`[N, size, W]`
    - 2D: :math:`[N, size, H, W]`
    - 3D: :math:`[N, size, D, H, W]`

    Output variable tensor shape:

    - 1D: :math:`[N, size,  W]`
    - 2D: :math:`[N, size, H, W]`
    - 3D: :math:`[N, size, D, H, W]`

    Example
    -------
    1D FNO model

    >>> decoder = FullyConnectedArch([Key("z", size=32)], [Key("y", size=2)])
    >>> fno_1d = FNOArch([Key("x", size=2)], dimension=1, decoder_net=decoder)
    >>> model = fno_1d.make_node()
    >>> input = {"x": torch.randn(20, 2, 64)}
    >>> output = model.evaluate(input)

    2D FNO model

    >>> decoder = ConvFullyConnectedArch([Key("z", size=32)], [Key("y", size=2)])
    >>> fno_2d = FNOArch([Key("x", size=2)], dimension=2, decoder_net=decoder)
    >>> model = fno_2d.make_node()
    >>> input = {"x": torch.randn(20, 2, 64, 64)}
    >>> output = model.evaluate(input)

    3D FNO model

    >>> decoder = Siren([Key("z", size=32)], [Key("y", size=2)])
    >>> fno_3d = FNOArch([Key("x", size=2)], dimension=3, decoder_net=decoder)
    >>> model = fno_3d.make_node()
    >>> input = {"x": torch.randn(20, 2, 64, 64, 64)}
    >>> output = model.evaluate(input)
    """

    def __init__(
        self,
        dimension: int,
        nr_fno_layers: int = 4,
        nr_ff_blocks: int = 2,
        fno_modes: Union[int, List[int]] = 16,
        padding: int = 8,
        padding_type: str = "constant",
        decoder_net: nn.Module = SirenArch(32, 1, 32, 2),
        coord_features: bool = True,
        functional: bool = False,
        weight_norm: bool = False,
        weight_sharing: bool = False,
        batch_norm: bool = False,
        dropout: bool = False,
    ) -> None:
        super().__init__()

        self.dimension = dimension
        self.nr_fno_layers = nr_fno_layers
        self.nr_ff_blocks = nr_ff_blocks
        self.fno_modes = fno_modes
        self.padding = padding
        self.padding_type = padding_type
        self.activation_fn = nn.GELU()
        self.coord_features = coord_features
        self.weight_norm = weight_norm

        # decoder net
        self.decoder_net = decoder_net

        # F-FNO
        self.functional = functional
        self.weight_sharing = weight_sharing
        self.batch_norm = batch_norm
        self.dropout = dropout

        in_channels = 1
        self.fno_layer_size = 32

        if self.functional:
            if self.dimension == 3:
                FNOModel = FunctionalFNO3DEncoder
                self.grid_to_points = grid_to_points3d  # For JIT
                self.points_to_grid = points_to_grid3d  # For JIT
            elif self.dimension == 2:
                FNOModel = FunctionalFNO2DEncoder
                self.grid_to_points = grid_to_points2d  # For JIT
                self.points_to_grid = points_to_grid2d  # For JIT
            else:
                raise NotImplementedError(
                    "Invalid dimensionality. Only 2D and 3D F-FNO implemented"
                )
            
            self.spec_encoder = FNOModel(
                in_channels,
                nr_fno_layers=self.nr_fno_layers,
                nr_ff_blocks=self.nr_ff_blocks,
                fno_layer_size=self.fno_layer_size,
                fno_modes=self.fno_modes,
                padding=self.padding,
                padding_type=self.padding_type,
                activation_fn=self.activation_fn,
                coord_features=self.coord_features,
                weight_sharing=self.weight_sharing,
                use_weight_norm=self.weight_norm,
                batch_norm=self.batch_norm,
                dropout=self.dropout,
            )
        else:
            if self.weight_sharing:
                raise NotImplementedError(
                    "Weight sharing is only implemented for F-FNO"
                )

            if self.dimension == 2:
                FNOModel = FNO2DEncoder
                self.grid_to_points = grid_to_points2d  # For JIT
                self.points_to_grid = points_to_grid2d  # For JIT
            elif self.dimension == 3:
                FNOModel = FNO3DEncoder
                self.grid_to_points = grid_to_points3d  # For JIT
                self.points_to_grid = points_to_grid3d  # For JIT
            else:
                raise NotImplementedError(
                    "Invalid dimensionality. Only 2D and 3D FNO implemented"
                )

            self.spec_encoder = FNOModel(
                in_channels,
                nr_fno_layers=self.nr_fno_layers,
                fno_layer_size=self.fno_layer_size,
                fno_modes=self.fno_modes,
                padding=self.padding,
                padding_type=self.padding_type,
                activation_fn=self.activation_fn,
                coord_features=self.coord_features,
                use_weight_norm=self.weight_norm
            )

    def forward(self, input: Tensor) -> Tensor:
        y_latent = self.spec_encoder(input)

        # Reshape to pointwise inputs if not a conv FC model
        flatten = (len(input.shape) == 5)
        y_shape = y_latent.shape
        if len(y_shape) > 2 and flatten:
            y_latent = self.grid_to_points(y_latent)

        y = self.decoder_net(y_latent)

        # Convert back into grid
        if len(y_shape) > 2 and flatten:
            y = self.points_to_grid(y, y_shape)

        return y