"""
Flatten layer,
Designed for flattening matrix of n rows and m columns to n*m array
"""

from typing import Dict
from typing import Tuple

import numpy as np

from bluebird.tensor import Tensor
from .input import Input

class Flatten(Input):
    def __init__(self, input_size: Tuple) -> None:
        super().__init__(input_size)

    def build(self) -> None:
        self.output_size = 1

        for i in self.input_size:
            self.output_size *= i

        self.backward_shape = (-1,) + self.input_size

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs.reshape(-1, self.output_size)
        
        if inputs.shape[1:] != self.input_size:
            raise TypeError("Invalid input shape")

        return self.inputs

    def backward(self, output: Tensor) -> Tensor:
        self.output = output
        return output.reshape(self.backward_shape)