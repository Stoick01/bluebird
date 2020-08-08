"""
Base optimizer arhitecture
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nn import NeuralNet

import numpy as np

from bluebird.tensor import Tensor
from bluebird.layers import Input
from bluebird.activations import Activation

class Optimizer:
    def step(self) -> None:
        """
        At each step the neural net is updated
        """
        raise NotImplementedError

    def build(self, net: 'NeuralNet') -> None:
        """
        Set aditional parameters needed for step
        """
        raise NotImplementedError