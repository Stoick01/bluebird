"""
Adaptive gradient
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nn import NeuralNet

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Input
from bluebird.activation import Activation

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
                    self.an.append(0)
                    continue

                self.an.append([np.zeros((layer.input_size, layer.output_size)), np.zeros(layer.output_size)])

    def step(self, predicted: Tensor, targets: Tensor) -> None:
        grad = self.net.loss.grad(predicted, targets)

        for layer, a in zip(self.net.get_layers(), self.an):

            if isinstance(layer, Input):
                continue

            if isinstance(layer, Activation):
                grad = layer.backward(grad)
                continue

            grad_b = np.sum(grad, axis=0)
            grad_w = layer.inputs.T @ grad

            a[0] += grad_w ** 2
            a[1] += grad_b ** 2

            layer.params["w"] -= grad_w * self.lr / (np.sqrt(a[0]) + self.epsilon) 

            layer.params["w"] -= grad_b * self.lr / (np.sqrt(a[1]) + self.epsilon)

            grad = grad @ layer.params["w"].T