"""
Nestov momentum
===============

Nestrov momentum is a type of accelerated gradient descent, it converges faster and reduces oscilation.

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
    Nestrov momentum.

    Stohastic Gradient Descent with momentum,
    Gradient accelerates and converges faster than with regular SGD.

    Example::
        
        optim = NestrovMomentum(lr=0.005)
        net.build(optimizer=optim)

    """

    def __init__(self,
                 lr: float = 0.001) -> None:
        """
        Initializes the object.

        Args:
            lr (float, optional): learning rate, defaults to 0.001
        """
        self.lr = lr
        self.vs = None

    def build(self, net: 'NeuralNet') -> None:
        """
        Called before training, optimizer needs the model to be able to iterate over params.

        This function is called douring build in your model.

        Args:
            net (:obj:`NeuralNet`): your model

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
        Traning step.

        This function is run during each of your training steps, it updates the model
        
        """
        
        for ((param, grad), v) in zip(self.net.get_params_and_grads(), self.vs):
            v = param * v - self.lr * grad
            param += v

