"""
Dense layer,
Combination of linear and activation layer
"""

from typing import Dict

import numpy as np

from bluebird.tensor import Tensor
from bluebird.activations import Activation
from bluebird.weight_initializers import WeightInitializer

from .layer import Layer
from .linear import Linear

class Dense(Layer):
    def __init__(self, output_size: int, activation: Activation = None) -> None:
        self.output_size = output_size
        self.hidden = activation

    def build(self, input_size: int, weight_initializer: WeightInitializer) -> None:
        self.layer = Linear(self.output_size)
        self.layer.build(input_size, weight_initializer)
        
        self.input_size = input_size
        self.params = self.layer.params
        self.grads = self.layer.grads

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        self.inputs = inputs

        self.outputs = self.layer.forward(inputs, training)

        if self.hidden != None:
            self.outputs = self.hidden.forward(self.outputs, training)

        return self.outputs

    def backward(self, grad: Tensor) -> Tensor:
        self.grad = grad

        if self.hidden != None:
            self.grad = self.hidden.backward(self.grad)

        self.grad = self.layer.backward(self.grad)

        return self.grad