"""
Droput layer,
randomly sets input units to 0
"""

from typing import Dict

import numpy as np

from bluebird.tensor import Tensor

from .layer import Layer
from .linear import Linear

class Dropout(Linear):
    def __init__(self, output_size: int, droput_rate: int) -> None:
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
