"""
Softplus
=======
"""
from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

import bluebird.utils as utl

from .activation import Activation

def softplus(x):
    """
    Softplus activation function.

    function:
        f(x) = len(1 + e^x)

    Args:
        x (:obj:`Tensor`): input to the function

    Returns:
        :obj:`Tensor`: f(x), applies activation function
    """

    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return np.log(1 + exp)

def softplus_prime(x):
    """
    Derivation of the softplus activation function.

    derivation:
        f'(x) = f(x) * (1 - f(x))

    Args:
        x (:obj:`Tensor`): input to the function

    Returns:
        :obj:`Tensor`: f'(x), applies derivation of activation function
    """

    exp = np.exp(-(x - x.max(axis=1, keepdims=True)))
    print(exp)
    return 1 / (1 + exp)

class Softplus(Activation):
    """
    Softplus activation function as object.

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
        super().__init__(softplus, softplus_prime)