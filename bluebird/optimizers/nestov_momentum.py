"""
Nestov momentum
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nn import NeuralNet

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Input
from bluebird.activations import Activation

from .optimizer import Optimizer


class NestovMomentum(Optimizer):
    """
    Stohastic Gradient Descent with momentum,
    Gradient accelerates and converges faster than with regular SGD


    Args:
        lr: learning rate
            default: 0.001
            type: float

    Example:
        
        >>> optim = NestrovMomentum(lr=0.005)
        >>> net.build(optimizer=optim, loss=CategoricalCrossEntropy())

    """

    def __init__(self,
                 lr: float = 0.001) -> None:
        self.lr = lr
        self.vs = None

    def build(self, net: 'NeuralNet') -> None:
        """
        Called before training, optimizer needs the model to be able to iterate over params
        It also creates momentum vecotors

        Args:
            net: your model, Type: NeuralNet

        Example:
            
            >>> optim = NestrovMomentum(lr=0.005)
            >>> optim.build(net)

        """

        self.net = net

        if self.vs == None:
            self.vs = []
            for layer in self.net.get_layers():
                if isinstance(layer, Input) or isinstance(layer, Activation):
                    continue

                self.vs.append(np.zeros((layer.input_size, layer.output_size)))
                self.vs.append(np.zeros(layer.output_size))

    def step(self) -> None:
        """
        Run training step

        Example:
            
            >>> optim = NestrovMomentum(lr=0.005)
            >>> optim.step()

        """
        
        for ((param, grad), v) in zip(self.net.get_params_and_grads(), self.vs):
            v = param * v - self.lr * grad
            param += v

