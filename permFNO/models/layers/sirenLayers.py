# Copy on Nvidia Modulus Sym Siren functions
# https://github.com/NVIDIA/modulus-sym/blob/main/modulus/sym/models/siren.py

# Based on Siren functions from Sitzmann, Vincent, et al.
# https://github.com/vsitzmann/siren

import enum
import math

import torch
import torch.nn as nn
from torch import Tensor

from torch.nn.utils.parametrizations import weight_norm


class SirenLayerType(enum.Enum):
    """
    SiReN layer types.
    """

    FIRST = enum.auto()
    HIDDEN = enum.auto()
    LAST = enum.auto()


class SirenLayer(nn.Module):
    """
    SiReN layer.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    layer_type : SirenLayerType
        Layer type.
    omega_0 : float
        Omega_0 parameter in SiReN.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_type: SirenLayerType = SirenLayerType.HIDDEN,
        omega_0: float = 30.0,
        use_weight_norm: bool = False,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.layer_type = layer_type
        self.omega_0 = omega_0
        self.use_weight_norm = use_weight_norm

        self.linear = nn.Linear(in_features, out_features, bias=True)

        self.apply_activation = layer_type in {
            SirenLayerType.FIRST,
            SirenLayerType.HIDDEN,
        }

        # set parameters and apply weight
        self.reset_parameters(use_weight_norm)
        if self.use_weight_norm:
            self.linear = weight_norm(self.linear)


    def reset_parameters(self, use_weight_norm) -> None:
        """Reset layer parameters."""
        weight_ranges = {
            SirenLayerType.FIRST: 1.0 / self.in_features,
            SirenLayerType.HIDDEN: math.sqrt(6.0 / self.in_features) / self.omega_0,
            SirenLayerType.LAST: math.sqrt(6.0 / self.in_features),
        }
        weight_range = weight_ranges[self.layer_type]

        nn.init.uniform_(self.linear.weight, -weight_range, weight_range)

        k_sqrt = math.sqrt(1.0 / self.in_features)
        nn.init.uniform_(self.linear.bias, -k_sqrt, k_sqrt)


    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.apply_activation:
            x = torch.sin(self.omega_0 * x)
        return x