"""
Flatten layer
=============

Designed for flattening Tensor of n rows and m columns to n*m Tensor.
"""

from typing import Dict
from typing import Tuple

import numpy as np

from bluebird.tensor import Tensor
from bluebird.exceptions import TypeException

from .input import Input

class Flatten(Input):
    """
    Flatten layer is a type of an Input layer.
    It transforms Tensor with dimensions (n, m) to a Tensor with dimensions of (n*m, 1).

    Example::

        input = Flatten((5, 10))
        net = NeuralNet([
                input,
                ...
            ])

    """

    def __init__(self, input_size: Tuple) -> None:
        """
        Initializes the object.

        Args:
            input_size (Tuple): dimensions of a input Tensor

        """

        if not isinstance(input_size, Tuple):
            raise TypeException("input_size", "Tuple")

        super().__init__(input_size)

    def build(self) -> None:
        """
        Called by the model, before its training step.

        Prepares the input size for next layer.

        """

        self.output_size = 1

        for i in self.input_size:
            self.output_size *= i

        self.backward_shape = (-1,) + self.input_size

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        """
        Called each time the data passes throughout the nework.

        Args:
            inputs (:obj:`Tensor`): input data to the network
            training (bool, optional): set to true during training, and is false when network predicts

        Returns:
            :obj:`Tensor`: reshaped Tensor (n*m, 1)
        
        """
        
        self.inputs = inputs.reshape(-1, self.output_size)
        
        if inputs.shape[1:] != self.input_size:
            raise TypeError("Invalid input shape")

        return self.inputs

    def backward(self, output: Tensor) -> Tensor:
        """
        Reshapes Tensor back to (n, m).

        Args:
            grad (:obj:`Tensor`): gradient from previous layer.

        Returns:
            :obj:`Tensor`: reshaped Tensor (n. m)

        """

        self.output = output
        return output.reshape(self.backward_shape)