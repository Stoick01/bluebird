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
        # grad = self.net.loss.grad(predicted, targets)

        # for layer, v in zip(self.net.get_layers(), self.vs):

        #     if isinstance(layer, Input):
        #         continue

        #     if isinstance(layer, Activation):

        #         grad = layer.backward(grad)
        #         continue


        #     grad_b = np.sum(grad, axis=0)
        #     grad_w = layer.inputs.T @ grad

        #     v[0] = self.b * v[0] - self.lr * np.sum(grad_w, axis=0)
        #     v[1] = self.b * v[1] - self.lr * np.sum(grad_b, axis=0)

        #     layer.params["w"] += v[0]
        #     layer.params["b"] += v[1]

        #     grad = layer.backward(grad) + self.b * grad @ v[0].T
        for ((param, grad), v) in zip(self.net.get_params_and_grads(), self.vs):
            v = param * v - self.lr * grad
            param += v

