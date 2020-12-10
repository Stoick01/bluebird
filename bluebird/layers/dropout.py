"""
Droput layer
============

Droput layer disables random inputs during training.
"""

from typing import Dict

import numpy as np

from bluebird.tensor import Tensor
from bluebird.exceptions import TypeException

import bluebird.utils as utl

from .layer import Layer
from .linear import Linear

class Dropout(Linear):
    """
    Dropout ignores some of the inputs and scales other ones.
    Usefull to escape overfitting.

    Inherits Linear layer, and changes only constructor and forward method.

    Backward is the same as in Linear.

    Example::

        dropout = Dropout(50, dropout_rate=0.02)
        net = NeuralNet([
                ...
                dropout
                ...
            ])

    """

    def __init__(self, output_size: int, droput_rate: float) -> None:
        """
        Initializes the object.

        Args:
            output_size (int): dimension of the output 
            dropout_rate (float): percent of inputs the network ignores (1:=100%)

        """

        if not isinstance(output_size, int):
            raise TypeException("output_size", "int")

        if not isinstance(droput_rate, float):
            raise TypeException("dropout_rate", "float")

        super().__init__(output_size)
        self.droput_rate = droput_rate

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        """
        Called each time the data passes throughout the nework.

        It only disables inputs during training, otherwise it acts like regular Linear layer.

        Args:
            inputs (:obj:`Tensor`): output from the previous layer
            training (bool, optional): set to true during training, and is false when network predicts

        Returns:
            :obj:`Tensor`: processed input data
        
        """

        if training:
            self.inputs = inputs / (1 - self.droput_rate)
            indices = np.random.choice(np.arange(self.inputs.shape[1]), 
                    replace=False,
                    size=int(self.inputs.shape[1] * self.droput_rate))
            z = np.zeros(inputs.shape[0])
            for ind in indices:
                self.inputs[:, ind] = 0
        else:
            self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]
