"""
Dense layer,
Combination of linear and activation layer
"""

from typing import Dict

import numpy as np

from bluebird.tensor import Tensor
from bluebird.activations import Activation

from .layer import Layer
from .linear import Linear

class Dense(Layer):
    def __init__(self, output_size: int, activation: Activation = None) -> None:
        self.output_size = output_size
        self.hidden = activation

    def build(self, input_size: int) -> None:
        self.layer = Linear(self.output_size)
        self.layer.build(input_size)
        
        self.input_size = input_size
        self.params = self.layer.params

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs

        self.outputs = self.layer.forward(inputs)

        if self.hidden != None:
            self.outputs = self.hidden.forward(self.outputs)

        return self.outputs

    def backward(self, grad: Tensor) -> Tensor:
        self.grad = grad

        if self.hidden != None:
            self.grad = self.hidden.backward(self.grad)

        self.grad = self.layer.backward(self.grad)

        return self.grad