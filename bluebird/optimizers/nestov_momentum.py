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
    def __init__(self,
                 lr: float = 0.001,
                 b: float = 0.8) -> None:
        self.lr = lr
        self.b = b
        self.vs = None

    def build(self, net: 'NeuralNet') -> None:
        self.net = net

        if self.vs == None:
            self.vs = []
            for layer in self.net.get_layers():
                if isinstance(layer, Input) or isinstance(layer, Activation):
                    continue

                self.vs.append(np.zeros((layer.input_size, layer.output_size)))
                self.vs.append(np.zeros(layer.output_size))

    def step(self) -> None:
        for ((param, grad), v) in zip(self.net.get_params_and_grads(), self.vs):
            v = param * v - self.lr * grad
            param += v

