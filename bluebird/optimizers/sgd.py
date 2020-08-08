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
    def __init__(self, lr: float = 0.001) -> None:
        self.lr = lr

    def build(self, net: 'NeuralNet') -> None:
        self.net = net

    def step(self) -> None:
        for param, grad in self.net.get_params_and_grads():
            param -= self.lr * grad
