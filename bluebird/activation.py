"""
Activation functions
"""

from typing import Callable

import numpy as np

from .tensor import Tensor
from .layers import Layer

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    Applies function to inputs
    """

    def __init__(self, f: F, f_prime: F, shape: int = 0) -> None:
        super().__init__(0)
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs

        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)

    return 1 - y ** 2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


def relu(x: Tensor) -> Tensor:
    x[x < 0] = 0
    return x

def relu_prime(x: Tensor) -> Tensor:
    x[x < 0] = 0
    x[x > 0] = 1
    return x

class Relu(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)