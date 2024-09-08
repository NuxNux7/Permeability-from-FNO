import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class FeedForwardBlock(nn.Module):
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
                weight_norm(layers[0])
            
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