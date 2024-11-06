# Based on the FFNO papers GitHub from Alasdair Tran, Alexander Mathews, Lexing Xie and Cheng Soon Ong
# https://github.com/alasdairtran/fourierflow/blob/main/fourierflow/modules/feedforward.py

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class FeedForwardBlock(nn.Module):
    """
    A feed-forward neural network block with configurable parameters.

    This block can be used as a building block for more complex neural network architectures.
    It allows for customization of the dimensionality, channel size, activation function,
    normalization, and dropout.

    Args:
        dims (int): The dimensionality of the input tensor (1, 2, or 3).
        fno_layer_size (int, optional): The size of the internal layers in the feed-forward block. Default is 32.
        activation_fn (nn.Module, optional): The activation function to be used in the block. Default is nn.GELU().
        use_weight_norm (bool, optional): Whether to enable weight normalization. Default is False.
        batch_norm (bool, optional): Whether to enable batch normalization. Default is False.
        dropout (float, optional): The dropout rate to be applied. Default is 0.0.
        num_blocks (int, optional): The number of feed-forward blocks to stack. Default is 1.
        factor (int, optional): A factor to adjust the channel size of the internal layers. Default is 1.

    Raises:
        ValueError: If the `dims` parameter is not 1, 2, or 3.
    """
        
    def __init__(self,
                 dims: int,
                 fno_layer_size: int = 32,
                 activation_fn = nn.GELU(),
                 use_weight_norm: bool = False,
                 batch_norm: bool = False,
                 dropout: float = 0.0,
                 num_blocks: int = 1,
                 factor: int = 1):
        super(FeedForwardBlock, self).__init__()

        if dims not in [1, 2, 3]:
            raise ValueError("Invalid dimensionality. Only 1D, 2D and 3D ConvBlock implemented")

        conv_class = getattr(nn, f'Conv{dims}d')
        bn_class = getattr(nn, f'BatchNorm{dims}d')
        dropout_class = getattr(nn, f'Dropout{dims}d')

        def create_block(block):
            # channel multiplication inbetween
            input_channel = factor * fno_layer_size
            if block == 0:
                input_channel = fno_layer_size

            output_channel = factor * fno_layer_size
            if block == (num_blocks - 1):
                output_channel = fno_layer_size

            layers = []
            layers.append(conv_class(input_channel, output_channel, 1))
            if use_weight_norm:
                layers[0] = weight_norm(layers[0])
            
            if batch_norm:
                layers.append(bn_class(output_channel))
            
            if (activation_fn is not None) and (block != (num_blocks - 1)):
                layers.append(activation_fn)
            
            if dropout > 0:
                layers.append(dropout_class(p=dropout))
            
            return nn.Sequential(*layers)

        self.blocks = nn.Sequential(*[create_block(block) for block in range(num_blocks)])


    def forward(self, x):
        return self.blocks(x)