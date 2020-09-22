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
    Represents Activation Layer
    Applies function when going forward in the network, and it's derivation when going backwards
    """

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        self.inputs = inputs

        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return utl.fix_overflow(self.f_prime(self.inputs) * grad)


# TO DO: Softplus, Softsign, SELU, ELU, exponential, leaky Relu