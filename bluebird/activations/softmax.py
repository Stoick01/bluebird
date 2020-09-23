"""
Softmax activation function
"""
from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

import bluebird.utils as utl

from .activation import Activation

def softmax(x):
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    sum_exp = exp.sum(axis=-1, keepdims=True)
    return exp / (sum_exp + 1e-8)

def softmax_prime(x):
    f = softmax(x)
    return f * (1 - f)

class Softmax(Activation):
    """
    Softmax activation function

    function:
        f(x) = e^x / sum(e^x)

    derivation:
        f'(x) = f(x) * (1 - f(x))
    """

    def __init__(self):
        super().__init__(softmax, softmax_prime)