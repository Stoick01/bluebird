"""
Stohastic Gradient Descent
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nn import NeuralNet

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Input
from bluebird.activation import Activation

from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr: float = 0.001) -> None:
        self.lr = lr

    def build(self, net: 'NeuralNet') -> None:
        self.net = net

    def step(self, predicted: Tensor, targets: Tensor) -> None:
        grad = self.net.loss.grad(predicted, targets)

        for layer in self.net.get_layers():

            if isinstance(layer, Input):
                continue

            if isinstance(layer, Activation):
                grad = layer.backward(grad)
                continue

            grad_b = np.sum(grad, axis=0)
            grad_w = layer.inputs.T @ grad

            layer.params["w"] -= self.lr * grad_w
            layer.params["b"] -= self.lr * grad_b

            grad = layer.backward(grad)
