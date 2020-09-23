"""
Basic input layer
Designed for specifing number of input neurons
All other preprocessing input layers inherit Input layer
"""

from typing import Dict, Tuple

import numpy as np

from bluebird.tensor import Tensor
from bluebird.exceptions import TypeException

from .layer import Layer

class Input(Layer):
    """
    Simple input layer
    it just passes the inputed values forward
    Input (or layer that inherits Input) should be the first layer in network

    Args:
        input_size: size of inputed tensor, Type: int

    Example:

        >>> input = Input(50)
        >>> net = NeuralNet([
                    input,
                    ...
                ])

    """

    def __init__(self, input_size: int) -> None:

        super().__init__()
        self.input_size = input_size

    def build(self) -> None:
        self.output_size = self.input_size

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        self.inputs = inputs

        if inputs.shape[1] != self.input_size:
            raise TypeError("Invalid input shape")

        return inputs

    def backward(self, output: Tensor) -> Tensor:
        self.output = output
        return output