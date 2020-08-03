"""
Basic input layer
Designed for specifing number of input neurons
All other preprocessing input layers inherit Input layer
"""

from typing import Dict

import numpy as np

from bluebird.tensor import Tensor
from .layer import Layer

class Input(Layer):
    def __init__(self, shape: int) -> None:
        super().__init__(shape)
        self.input_size = shape

    def build(self, output_size: int = 0) -> None:
        self.output_size = self.input_size

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs

    def backward(self, output: Tensor) -> Tensor:
        self.output = output
        return output