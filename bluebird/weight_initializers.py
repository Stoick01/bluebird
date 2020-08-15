"""
Set of classes for weight inititalization
"""

from typing import Tuple

import numpy as np

from .tensor import Tensor

class WeightInitializer:
    def init(self, dimension) -> Tensor:
        """
        Function that initializes weights
        """

        raise NotImplementedError


class RandomWeightInitializer:
    def init(self, dimension) -> Tensor:
        return np.random.randn(*dimension)


# TO DO: RandomUniform, Zeros, Ones
