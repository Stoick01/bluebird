"""
Droput layer,
randomly sets input units to 0
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
    It ignores some of inouts and scales other ones,
    Usefull to escape overfitting

    Args:
        output_size: number of neurons, Type: int
        dropout_rate: percent of inputs the network ignores, Type: float
            (1:=100%)

    Example:

        >>> dropout = Dropout(50, dropout_rate=0.02)
        >>> net = NeuralNet([
                    ...
                    dropout
                    ...
                ])

    """

    def __init__(self, output_size: int, droput_rate: float) -> None:
        if not isinstance(output_size, int):
            raise TypeException("output_size", "int")

        if not isinstance(droput_rate, float):
            raise TypeException("dropout_rate", "float")

        super().__init__(output_size)
        self.droput_rate = droput_rate

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        """
        output = inputs @ w + b
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
