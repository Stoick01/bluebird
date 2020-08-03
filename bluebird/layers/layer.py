"""
Basic layer arhitecture
All other layers and activations inherit this basic arhitecture
"""

from typing import Dict

import numpy as np

from bluebird.tensor import Tensor

class Layer:
    def __init__(self, shape: int = 0) -> None:
        self.shape = shape

        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def build(self, input_size) -> None:
        """
        Used to finalize building layers
        """
        raise NotImplementedError

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produces output for inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagates the gradinet through the layer
        """
        raise NotImplementedError