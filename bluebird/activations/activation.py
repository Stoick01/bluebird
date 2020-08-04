"""
Basic arhitecture for activation function
"""

from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    Applies function to inputs
    """

    def __init__(self, f: F, f_prime: F) -> None:
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs

        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad