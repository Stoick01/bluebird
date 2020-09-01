"""
Basic arhitecture for activation function
"""

from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

import bluebird.utils as utl

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    Applies function to inputs
    """

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        self.inputs = inputs

        return utl.scale(self.f(inputs))

    def backward(self, grad: Tensor) -> Tensor:
        return utl.scale(self.f_prime(self.inputs) * grad)


# TO DO: Sigmoid, Softplus, Softsign, SELU, ELU, exponential