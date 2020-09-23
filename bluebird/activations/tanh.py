"""
Tanh activation function
"""

from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

from .activation import Activation

def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)

    return 1 - y ** 2

class Tanh(Activation):
    """
    Tanh activation function

    function:
        f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    derivation:
        f'(x) = 1 - f(x)^2 
    """

    def __init__(self):
        super().__init__(tanh, tanh_prime)
