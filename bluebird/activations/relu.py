"""
Relu
====

Relu (Rectified Linear Unit).
"""

from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

from .activation import Activation


def relu(x: Tensor) -> Tensor:
    """
    Relu activation function.

    function:
        f(x) = 0 if x < 0

        f(x) = x if x > 0

    Args:
        x (:obj:`Tensor`): input to the function

    Returns:
        :obj:`Tensor`: f(x), applies activation function
    """

    return np.maximum(x, 0.0)

def relu_prime(x: Tensor) -> Tensor:
    """
    Derivation of the relu activation function.

    derivation:
        f'(x) = 0 if x < 0

        f'(x) = 1 if x > 0

    Args:
        x (:obj:`Tensor`): input to the function

    Returns:
        :obj:`Tensor`: f'(x), applies derivation of activation function
    """

    x[x < 0] = 0
    x[x > 0] = 1
    return x

class Relu(Activation):
    """
    Relu activation function as object.

    Inherits all of its atributes from base Activation class.

    Only functions are specified, which you can see in previous page.

    It is important to note that relu activation works only with small variances,
    so weights should be initializes with a weight initializes that does that.

    Example::

        relu = Relu()
        net = NeuralNet([
                ...
                relu,
                ...
            ])
    """

    def __init__(self):
        super().__init__(relu, relu_prime)