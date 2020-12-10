"""
Tanh
====

Hyperbolic tangent.
"""

from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

from .activation import Activation

def tanh(x: Tensor) -> Tensor:
    """
    Tenh activation function.

    function:
        f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Args:
        x (:obj:`Tensor`): input to the function

    Returns:
        :obj:`Tensor`: f(x), applies activation function
    """

    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    """
    Derivation of the tanh activation function.

    derivation:
        f'(x) = 1 - f(x)^2 

    Args:
        x (:obj:`Tensor`): input to the function

    Returns:
        :obj:`Tensor`: f'(x), applies derivation of activation function
    """
    y = tanh(x)

    return 1 - y ** 2

class Tanh(Activation):
    """
    Tanh activation function as object.

    Inherits all of its atributes from base Activation class.

    Only functions are specified, which you can see in previous page.

    Example::

        tanh = Tanh()
        net = NeuralNet([
                ...
                tanh,
                ...
            ])
    """

    def __init__(self):
        super().__init__(tanh, tanh_prime)
