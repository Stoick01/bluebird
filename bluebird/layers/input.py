"""
Basic input layer
Designed for specifing number of input neurons
All other preprocessing input layers inherit Input layer
"""

from typing import Dict, Tuple

import numpy as np

from bluebird.tensor import Tensor
from .layer import Layer

class Input(Layer):
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size

    def build(self) -> None:
        self.output_size = self.input_size

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs

        if inputs.shape[1] != self.input_size:
            raise TypeError("Invalid input shape")

        return inputs

    def backward(self, output: Tensor) -> Tensor:
        self.output = output
        return output