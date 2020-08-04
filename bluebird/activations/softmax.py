"""
Softmax activation function
"""

from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

from .activation import Activation

def softmax(x):
    exp = np.exp(x)
    return exp / (np.sum(exp, axis=0) + 1e-8)

def softmax_prime(x):
    return softmax(x) * (1 - softmax(x))

class Softmax(Activation):
    def __init__(self):
        super().__init__(softmax, softmax_prime)