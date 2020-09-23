"""
Stohastic Gradient Descent
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

    Args:
        lr: learning rate
            default: 0.001
            type: float

    Example:
        
        >>> optim = SGD(lr=0.005)
        >>> net.build(optimizer=optim, loss=CategoricalCrossEntropy())

    """

    def __init__(self, lr: float = 0.001) -> None:
        self.lr = lr

    def build(self, net: 'NeuralNet') -> None:
        """
        Called before training, optimizer needs the model to be able to iterate over params

        Args:
            net: your model, Type: NeuralNet

        Example:
            
            >>> optim = SGD(lr=0.005)
            >>> optim.build(net)

        """

        self.net = net

    def step(self) -> None:
        """
        Run training step

        Example:
            
            >>> optim = SGD(lr=0.005)
            >>> optim.step()

        """
        
        for param, grad in self.net.get_params_and_grads():
            param -= self.lr * grad
