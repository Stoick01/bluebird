"""
Sigmoid
=======
"""

from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

from .activation import Activation


def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid activation function.

    function:
        f(x) = 1 / (x + e^x)

    Args:
        x (:obj:`Tensor`): input to the function

    Returns:
        :obj:`Tensor`: f(x), applies activation function
    """

    return 1 / (1 + np.exp(x))

def sigmoid_prime(x: Tensor) -> Tensor:
    """
    Derivation of the sigmoid activation function.

    derivation:
        f'(x) = f(x) * (1 - f(x))

    Args:
        x (:obj:`Tensor`): input to the function

    Returns:
        :obj:`Tensor`: f'(x), applies derivation of activation function
    """

    return sigmoid(x) * (1 - sigmoid(x))

class Sigmoid(Activation):
    """
    Sigmoid activation function as object.

    Inherits all of its atributes from base Activation class.

    Only functions are specified, which you can see in previous page.

    Example::

        sigmoid = Sigmoid()
        net = NeuralNet([
                ...
                sigmoid,
                ...
            ])
    """

    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)