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
    def __init__(self,
                 lr: float = 0.001,
                 epsilon: float = 1e-8) -> None:
        self.lr = lr
        self.epsilon = epsilon
        self.an = None

    def build(self, net: 'NeuralNet') -> None:
        self.net = net

        if self.an == None:
            self.an = []
            for layer in self.net.get_layers():
                if isinstance(layer, Input) or isinstance(layer, Activation):
                    continue
                self.an.append(np.zeros((layer.input_size, layer.output_size)))
                self.an.append(np.zeros(layer.output_size))

    def step(self) -> None:       
        for ((param, grad), a) in zip(self.net.get_params_and_grads(), self.an):
            a += grad ** 2
            param -= self.lr * grad / np.sqrt(a + self.epsilon)