"""
Basic linear layer
"""

from typing import Dict

import numpy as np

from bluebird.tensor import Tensor
from .layer import Layer

class Linear(Layer):
    """
    output = input @ w + b
    """

    def __init__(self, shape: int) -> None:
        super().__init__(shape)
        self.output_size = shape

    def build(self, input_size):
        self.input_size = input_size

        self.params["w"] = np.random.randn(input_size, self.output_size)
        self.params["b"] = np.random.randn(self.output_size)
        

    def forward(self, inputs: Tensor) -> Tensor:
        """
        output = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        return grad @ self.params["w"].T