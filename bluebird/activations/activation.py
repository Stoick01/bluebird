"""
Activation
==========

Base activation class, that all other activation function inherit.
"""

from typing import Callable

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Layer

import bluebird.utils as utl

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    Default activation, that all other activations inherit

    Example::

       class CustomActivation(Activation):
            def __init__(self, f: F, f_prime: F) -> None:
                super().__init__(f, f_prime)
                # where f is activation function and f_prime is its derivation
                ...

            # forward and backward methods should not be touched (unless you know what you are doing)
            
    """

    def __init__(self, f: F, f_prime: F) -> None:
        """
        Initializes the object.

        Function and it's derivation are of the type:
        ``F = Callable[[Tensor], Tensor]``

        Args:
            f (F[Callable]): activation function
            f_prime (F[Callable]): derivation of activation function

        """

        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        """ 
        Called each time the data passes throughout the nework.

        Args:
            inputs (:obj:`Tensor`): output from the previous layer
            training (bool, optional): set to true during training, and is false when network predicts

        Returns:
            :obj:`Tensor`: f(inputs), applies the activation function to the data
        
        """

        self.inputs = inputs

        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        """
        Used to calculate the gradients of weights and biases.

        Args:
            grad (:obj:`Tensor`): gradient from previous layer or loss function.

        Returns:
            :obj:`Tensor`: f_prime(grad), applies deravtion of the activation function to the gradient

        """

        return utl.fix_overflow(self.f_prime(self.inputs) * grad)


# TO DO: Softplus, Softsign, SELU, ELU, exponential, leaky Relu