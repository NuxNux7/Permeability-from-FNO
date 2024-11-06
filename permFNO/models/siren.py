# Based on Nvidia Modulus Sym Siren functions
# https://github.com/NVIDIA/modulus-sym/blob/main/modulus/sym/models/siren.py

# Based on Siren functions from Sitzmann, Vincent, et al.
# https://github.com/vsitzmann/siren

from typing import List, Dict, Tuple, Optional

import torch.nn as nn
from torch import Tensor

from .layers.sirenLayers import SirenLayer, SirenLayerType


class SirenArch(nn.Module):
    """Sinusoidal Representation Network (SIREN).

    Parameters
    ----------
    in_features : int
        Number of input channels
    out_features : int
        Number of output channels
    layer_size : int, optional
        Layer size for every hidden layer of the model, by default 512
    nr_layers : int, optional
        Number of hidden layers of the model, by default 6
    first_omega : float, optional
        Scales first weight matrix by this factor, by default 30
    omega : float, optional
        Scales the weight matrix of all hidden layers by this factor, by default 30
    normalization : Dict[str, Tuple[float, float]], optional
        Normalization of input to network, by default None
    weight_norm: bool, optional
        Apply normalization to the weights of the linear layer


    Variable Shape
    --------------
    - Input variable tensor shape: :math:`[N, size]`
    - Output variable tensor shape: :math:`[N, size]`

    Example
    -------
    Siren model (2 -> 64 -> 64 -> 2)

    >>> arch = .siren.SirenArch(
    >>>    [Key("x", size=2)],
    >>>    [Key("y", size=2)],
    >>>    layer_size = 64,
    >>>    nr_layers = 2)
    >>> model = arch.make_node()
    >>> input = {"x": torch.randn(64, 2)}
    >>> output = model.evaluate(input)

    Note
    ----
    Reference: Sitzmann, Vincent, et al.
    Implicit Neural Representations with Periodic Activation Functions.
    https://arxiv.org/abs/2006.09661.
    """

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 1,
        layer_size: int = 512,
        nr_layers: int = 6,
        first_omega: float = 30.0,
        omega: float = 30.0,
        weight_norm: bool = False,
    ) -> None:
        super().__init__()

        layers_list = []

        layers_list.append(
            SirenLayer(
                in_features,
                layer_size,
                SirenLayerType.FIRST,
                first_omega,
                weight_norm,
            )
        )

        for _ in range(nr_layers - 1):
            layers_list.append(
                SirenLayer(layer_size, layer_size, SirenLayerType.HIDDEN, omega, weight_norm)
            )

        layers_list.append(
            SirenLayer(layer_size, out_features, SirenLayerType.LAST, omega, weight_norm)
        )

        self.layers = nn.Sequential(*layers_list)

    def forward(self, input: Tensor) -> Tensor:
        x = input
        y = self.layers(x)
        return y
