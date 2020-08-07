"""
Relu activation function
"""

from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

from .activation import Activation


def relu(x: Tensor, epsilon: float = 0.1) -> Tensor:
    return np.maximum(epsilon * x, x)

def relu_prime(x: Tensor, epsilon: float = 0.1) -> Tensor:
    x = 1.0 * (x > 0)
    x[x == 0] = epsilon
    return x

class Relu(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)