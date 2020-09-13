"""
Dense layer,
Combination of linear and activation layer
"""

from typing import Dict

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bluebird.activations import Activation

import numpy as np

from .layer import Layer
from .linear import Linear

import bluebird as bb
from bluebird.tensor import Tensor
from bluebird.weight_initializers import WeightInitializer, GlorotUniformWeightInitializer, ZerosWeightInitializer

from bluebird.exceptions import TypeException


class Dense(Layer):
    def __init__(self, output_size: int, activation: 'Activation', 
                 weight_initializer: WeightInitializer = GlorotUniformWeightInitializer(),
                 bias_initializer: WeightInitializer = ZerosWeightInitializer()) -> None:
        if not isinstance(output_size, int):
            raise TypeException("output_size", "int")

        # if not isinstance(activation, bb.activations.Activation):
        #     raise TypeException("activation", "Activation")

        if not isinstance(weight_initializer, WeightInitializer):
            raise TypeException("weight_initializer", "WeightInitializer")

        if not isinstance(bias_initializer, WeightInitializer):
            raise TypeException("bias_initializer", "WeightInitializer")

        self.output_size = output_size
        self.hidden = activation

        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_size: int) -> None:
        self.layer = Linear(self.output_size, self.weight_initializer, self.bias_initializer)
        self.layer.build(input_size)
        
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