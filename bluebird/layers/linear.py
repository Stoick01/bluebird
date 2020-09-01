"""
Basic linear layer
"""

import numpy as np

from bluebird.tensor import Tensor
from bluebird.weight_initializers import WeightInitializer
import bluebird.utils as utl

from .layer import Layer

class Linear(Layer):
    """
    output = input @ w + b
    """

    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size

    def build(self, input_size, weight_initializer: WeightInitializer):
        self.input_size = input_size

        self.params["w"] = weight_initializer.init((input_size, self.output_size))
        self.params["b"] = weight_initializer.init((self.output_size,))
        

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        """
        output = inputs @ w + b
        """
        self.inputs = inputs
        return utl.scale(inputs @ self.params["w"] + self.params["b"])

    def backward(self, grad: Tensor) -> Tensor:

        self.grads["b"] = utl.scale(np.sum(grad, axis=0))
        self.grads["w"] = utl.scale(self.inputs.T @ grad)
        return utl.scale(grad @ self.params["w"].T)