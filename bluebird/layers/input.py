"""
Input layer
===========

Input type layers are first layers in network.
They are used to help with preprocessing data, and to initialize the input size for the rest of the network.

Every model must have an input type layer.

"""

from typing import Dict, Tuple

import numpy as np

from bluebird.tensor import Tensor
from bluebird.exceptions import TypeException

from .layer import Layer

class Input(Layer):
    """
    Input is the base input type layer, it just passes inputed values to the next layer.

    Input type layer must be the first layer of the network.

    Example::

        input = Input(50)
        net = NeuralNet([
                input,
                ...
            ])

    """

    def __init__(self, input_size: int) -> None:
        """
        Initializes the object.

        Args:
            input_size (int): size of inputed Tensor

        """

        super().__init__()
        self.input_size = input_size

    def build(self) -> None:
        """
        Called by the model, before its training step.

        Prepares the input size for next layer.

        """

        self.output_size = self.input_size

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        """
        Called each time the data passes throughout the nework.

        Args:
            inputs (:obj:`Tensor`): input data to the network
            training (bool, optional): set to true during training, and is false when network predicts

        Returns:
            :obj:`Tensor`: Tensor that it has recived
        
        """

        self.inputs = inputs

        if inputs.shape[1] != self.input_size:
            raise TypeError("Invalid input shape")

        return inputs

    def backward(self, output: Tensor) -> Tensor:
        """
        Passes the same Tensor just backwards.

        Args:
            grad (:obj:`Tensor`): gradient from previous layer.

        Returns:
            :obj:`Tensor`: Tensor that it has recived.

        """

        self.output = output
        return output