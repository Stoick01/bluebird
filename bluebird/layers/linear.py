"""
Basic linear layer
"""

import numpy as np

from bluebird.tensor import Tensor
from bluebird.weight_initializers import WeightInitializer, ZerosWeightInitializer, HeWeightInitializer
import bluebird.utils as utl

from .layer import Layer

class Linear(Layer):
    """
    Simple linear layer
    output = input @ w + b

    Args:
        output_size: number of neurons, Type: int
        weight_initializer: defines how weights are initialized
            type: WeightInitializer
            default: HeWeightInitializer
        bias_initializer: defines how weights are initialized
            type: WeightInitializer
            default: ZerosWeightInitializer

    Example:

        >>> linear = Lainear(50)
        >>> net = NeuralNet([
                    ...
                    linear
                    ...
                ])
    """

    def __init__(self, output_size: int,
                 weight_initializer: WeightInitializer = HeWeightInitializer(),
                 bias_initializer: WeightInitializer = ZerosWeightInitializer()) -> None:

        if not isinstance(output_size, int):
            raise TypeException("output_size", "int")

        if not isinstance(weight_initializer, WeightInitializer):
            raise TypeException("weight_initializer", "WeightInitializer")

        if not isinstance(bias_initializer, WeightInitializer):
            raise TypeException("bias_initializer", "WeightInitializer")

        super().__init__()
        self.output_size = output_size

        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_size: int):
        self.input_size = input_size

        self.params["w"] = self.weight_initializer.init((input_size, self.output_size))
        self.params["b"] = self.bias_initializer.init((self.output_size,))
        

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        """
        output = inputs @ w + b
        """
        self.inputs = inputs
        return utl.fix_overflow(inputs @ self.params["w"] + self.params["b"])

    def backward(self, grad: Tensor) -> Tensor:

        self.grads["b"] = utl.fix_overflow(np.sum(grad, axis=0))
        self.grads["w"] = utl.fix_overflow(self.inputs.T @ grad)
        return utl.fix_overflow(grad @ self.params["w"].T)