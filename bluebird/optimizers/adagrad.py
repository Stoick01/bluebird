"""
Adaptive gradient
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nn import NeuralNet

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Input
from bluebird.activations import Activation

from .optimizer import Optimizer


class AdaGrad(Optimizer):
    """
    Adaptive gradient descent


    Args:
        lr: learning rate
            default: 0.001
            type: float
        epsilon: small value to scape the division by zero, best to leave it alone
            default: 1e-8
            type: float

    Example:
        
        >>> optim = AdaGrad(lr=0.005)
        >>> net.build(optimizer=optim, loss=CategoricalCrossEntropy())

    """

    def __init__(self,
                 lr: float = 0.001,
                 epsilon: float = 1e-8) -> None:
        self.lr = lr
        self.epsilon = epsilon
        self.an = None

    def build(self, net: 'NeuralNet') -> None:
        """
        Called before training, optimizer needs the model to be able to iterate over params

        Args:
            net: your model, Type: NeuralNet

        Example:
            
            >>> optim = AdaGrad(lr=0.005)
            >>> optim.build(net)

        """

        self.net = net

        if self.an == None:
            self.an = []
            for layer in self.net.get_layers():
                if isinstance(layer, Input) or isinstance(layer, Activation):
                    continue
                self.an.append(np.zeros((layer.input_size, layer.output_size)))
                self.an.append(np.zeros(layer.output_size))

    def step(self) -> None:
        """
        Run training step

        Example:
            
            >>> optim = AdaGrad(lr=0.005)
            >>> optim.step()

        """
              
        for ((param, grad), a) in zip(self.net.get_params_and_grads(), self.an):
            a += grad ** 2
            param -= self.lr * grad / np.sqrt(a + self.epsilon)