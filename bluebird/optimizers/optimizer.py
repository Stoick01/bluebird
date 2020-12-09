"""
Optimizer
=========

Base optimizer class that all other optimizers inherit.

"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nn import NeuralNet

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Input
from bluebird.activations import Activation

class Optimizer:
    """
    Base optimizer class that all other optimizers inherit

    Example::

        class CustomOptimizer(Optimizer):
            def build(self, net: 'NeuralNet') -> None:
                self.net = net
                
                ... Any aditional variables you wish to initialize

            def step(self) -> None:
                for param, grad in self.net.get_params_and_grads():
                    ... Update params and grads

    """

    def step(self) -> None:
        """
        At each step the neural net is updated.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

    def build(self, net: 'NeuralNet') -> None:
        """
        Set aditional parameters needed for step.

        Raises:
            NotImplementedError

        """
        raise NotImplementedError

# TO DO: RMSprop, Adadelta, Adamax, Nadam, Ftrl