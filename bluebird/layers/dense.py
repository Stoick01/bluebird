"""
Dense layer,
Combination of linear and activation layer
"""

from typing import Dict

import numpy as np

from bluebird.tensor import Tensor
from bluebird.activation import Activation

from .layer import Layer

class Dense(Layer):
    def __init__(self, input_size: int, activation: Activation) -> None: