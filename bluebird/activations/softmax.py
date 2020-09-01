"""
Softmax activation function
"""
from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

from .activation import Activation

def softmax(x):
    exp = np.exp((x-x.mean(axis=-1, keepdims=True)) / np.sqrt(x.var(axis=-1, keepdims=True) + 1e-8))
    sum_exp = exp.sum(axis=-1, keepdims=True)
    return exp / sum_exp

def softmax_prime(x):
    f = softmax(x)
    return f * (1 - f)

class Softmax(Activation):
    def __init__(self):
        super().__init__(softmax, softmax_prime)