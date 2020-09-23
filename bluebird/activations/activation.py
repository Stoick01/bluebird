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
    Default activation, that all other activations inherit

    Args:
        f: activation function, Type: F (callable)
        f_prime: derivation of activation function, Type: F (Callable)

    Example:

       class CustomActivation(Activation):
            def __init__(self, f: F, f_prime: F) -> None:
                super().__init__(f, f_prime)
                ...
            
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