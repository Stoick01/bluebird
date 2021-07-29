"""
Leaky Relu
==========

Leaky Relu (Rectified Linear Unit).
"""

from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

from .activation import Activation


def leaky_relu(x: Tensor, alpha) -> Tensor:
    """
    Relu activation function.

    function:
        f(x) = 0 if x < 0

        f(x) = c * x if x > 0

    Args:
        x (:obj:`Tensor`): input to the function
        alpha (float): small value that multiplies the input

    Returns:
        :obj:`Tensor`: f(x), applies activation function
    """

    return np.maximum(x, x * alpha)

def leaky_relu_prime(x: Tensor, alpha) -> Tensor:
    """
    Derivation of the relu activation function.

    derivation:
        f'(x) = c if x < 0

        f'(x) = 1 if x > 0

    Args:
        x (:obj:`Tensor`): input to the function
        alpha (float): small value that multiplies the input

    Returns:
        :obj:`Tensor`: f'(x), applies derivation of activation function
    """

    x[x > 0] = 1
    x[x < 0] = alpha
    return x

class LeakyRelu(Activation):
    """
    Relu activation function as object.

    Inherits all of its atributes from base Activation class.

    Only functions are specified, which you can see in previous page.

    It is important to note that leaky relu activation works only with small variances,
    so weights should be initializes with a weight initializes that does that.

    Example::

        leaky_relu = LeakyRelu()
        net = NeuralNet([
                ...
                leaky_relu,
                ...
            ])
    """

    def __init__(self, alpha: float = 0):
        """

        """
        super().__init__(leaky_relu, leaky_relu_prime, alpha)