"""
Relu activation function
"""

from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

from .activation import Activation


def relu(x: Tensor) -> Tensor:
    return np.maximum(x, 0.0)

def relu_prime(x: Tensor) -> Tensor:
    x[x < 0] = 0
    x[x > 0] = 1
    return x

class Relu(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)