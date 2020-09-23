"""
Sigmoid activation function
"""

from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

from .activation import Activation


def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + np.exp(x))

def sigmoid_prime(x: Tensor) -> Tensor:
    return sigmoid(x) * (1 - sigmoid(x))

class Sigmoid(Activation):
    """
    Sigmoid activation function

    function:
        f(x) = 1 / (x + e^x)

    derivation:
        f'(x) = f(x) * (1 - f(x))
    """

    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)