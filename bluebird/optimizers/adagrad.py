"""
Adaptive gradient descent
=========================

Unlike other optimizers, learning rate adapts to the data, it's well suited for sparse data.
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
    Adaptive gradient descent.

    Example::

        optim = AdaGrad(lr=0.005)
        net.build(optimizer=optim, loss=CategoricalCrossEntropy())

    """

    def __init__(self,
                 lr: float = 0.001,
                 epsilon: float = 1e-8) -> None:
        """
        Initializes the object.

        Args:
            lr (float, optional): learning rate, defaults to 0.001
            epsilon (float, otpional): small value to prevent division by zero, best not to touch it, defaults to 1e-8
        """
        self.lr = lr
        self.epsilon = epsilon
        self.an = None

    def build(self, net: 'NeuralNet') -> None:
        """
        Called before training, optimizer needs the model to be able to iterate over params.

        This function is called douring build in your model.

        Args:
            net (:obj:`NeuralNet`): your model

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
        Traning step.

        This function is run during each of your training steps, it updates the model
        
        """
              
        for ((param, grad), a) in zip(self.net.get_params_and_grads(), self.an):
            a += grad ** 2
            param -= self.lr * grad / np.sqrt(a + self.epsilon)