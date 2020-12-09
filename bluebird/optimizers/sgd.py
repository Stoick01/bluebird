"""
Stohastic Gradient Descent
==========================

Basic stohastic gradient descent, updates weights based on learning rate.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nn import NeuralNet

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Input
from bluebird.activations import Activation

from .optimizer import Optimizer


class SGD(Optimizer):
    """
    Stohastic Gradient Descent

    Example:
        
        optim = SGD(lr=0.005)
        net.build(optimizer=optim)

    """

    def __init__(self, lr: float = 0.001) -> None:
        """
        Initializes the object.

        Args:
            lr (float, optional): learning rate, defaults to 0.001
        """
        self.lr = lr

    def build(self, net: 'NeuralNet') -> None:
        """
        Called before training, optimizer needs the model to be able to iterate over params.

        This function is called douring build in your model.

        Args:
            net (:obj:`NeuralNet`): your model

        """

        self.net = net

    def step(self) -> None:
        """
        Traning step.

        This function is run during each of your training steps, it updates the model
        
        """
        
        for param, grad in self.net.get_params_and_grads():
            param -= self.lr * grad
