"""
Softmax
=======
"""
from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

import bluebird.utils as utl

from .activation import Activation

def softmax(x):
    """
    Softmax activation function.

    function:
        f(x) = e^x / sum(e^x)

    Args:
        x (:obj:`Tensor`): input to the function

    Returns:
        :obj:`Tensor`: f(x), applies activation function
    """

    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    sum_exp = exp.sum(axis=-1, keepdims=True)
    return exp / (sum_exp + 1e-8)

def softmax_prime(x):
    """
    Derivation of the softmax activation function.

    derivation:
        f'(x) = f(x) * (1 - f(x))

    Args:
        x (:obj:`Tensor`): input to the function

    Returns:
        :obj:`Tensor`: f'(x), applies derivation of activation function
    """

    f = softmax(x)
    return f * (1 - f)

class Softmax(Activation):
    """
    Softmax activation function as object.

    Inherits all of its atributes from base Activation class.

    Only functions are specified, which you can see in previous page.

    Example::

        softmax = Softmax()
        net = NeuralNet([
                ...
                softmax,
                ...
            ])
    """

    def __init__(self):
        super().__init__(softmax, softmax_prime)