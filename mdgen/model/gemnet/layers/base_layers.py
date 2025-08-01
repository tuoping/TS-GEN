"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
Adapted from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet/layers/base_layers.py.
"""

import math
from collections.abc import Callable
from typing import Optional

import torch

from ..initializers import he_orthogonal_init


class Dense(torch.nn.Module):
    """
    Combines dense layer with scaling for swish activation.

    Parameters
    ----------
        units: int
            Output embedding size.
        activation: str
            Name of the activation function to use.
        bias: bool
            True if use bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation: Optional[str] = None,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation == "siqu":
            self._activation = SiQU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise NotImplementedError("Activation function not implemented for GemNet (yet).")

    def reset_parameters(self, initializer: Callable = he_orthogonal_init):
        initializer(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = self._activation(x)
        return x


class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor):
        return self._activation(x) * self.scale_factor


class SiQU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor):
        return x * self._activation(x)


class ResidualLayer(torch.nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Parameters
    ----------
        units: int
            Output embedding size.
        nLayers: int
            Number of dense layers.
        layer_kwargs: str
            Keyword arguments for initializing the layers.
    """

    def __init__(self, units: int, nLayers: int = 2, layer: Callable = Dense, **layer_kwargs):
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(
            *[
                layer(in_features=units, out_features=units, bias=False, **layer_kwargs)
                for _ in range(nLayers)
            ]
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2)

    def forward(self, input: torch.Tensor):
        x = self.dense_mlp(input)
        x = input + x
        x = x * self.inv_sqrt_2
        return x
