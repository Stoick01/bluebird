"""
Softmax activation function
"""

from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

from .activation import Activation

def softmax(x):
    exp = np.exp(x - x.max())
    return exp / (np.sum(exp, axis=0) + 1e-8)

def softmax_prime(x):
    f = softmax(x)
    return f * (1 - f)

class Softmax(Activation):
    def __init__(self):
        super().__init__(softmax, softmax_prime)