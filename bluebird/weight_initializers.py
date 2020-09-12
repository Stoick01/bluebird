"""
Set of classes for weight inititalization
"""

from typing import Tuple

import numpy as np

from .tensor import Tensor

class WeightInitializer:
    def init(self, dimension: Tuple) -> Tensor:
        """
        Function that initializes weights
        """

        raise NotImplementedError



class RandomWeightInitializer:
    def init(self, dimension: Tuple) -> Tensor:
        return np.random.randn(*dimension)

class RandomUniformWeightInitializer:
    def init(self, dimension: Tuple) -> Tensor:
        return np.random.uniform(-1, 1, dimension)
       

class GlorotUniformWeightInitializer:
    def init(self, dimension: Tuple) -> Tensor:
        l = 0
        for d in dimension:
            l += d
        sd = np.sqrt(6.0 / l)
        return np.random.uniform(-sd, sd, dimension)

class GlorotNormalWeightInitializer:
    def init(self, dimension: Tuple) -> Tensor:
        l = 0
        for d in dimension:
            l += d
        sd = np.sqrt(2.0 / l)
        return np.random.normal(0, sd, dimension)

class ZerosWeightInitializer:
    def init(self, dimension: Tuple) -> Tensor:
        return np.zeros(dimension)

class OnesWeightInitializer:
    def init(self, dimension: Tuple) -> Tensor:
        return np.ones(dimension)


# TO DO: He
