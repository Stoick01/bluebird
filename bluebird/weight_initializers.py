"""
Set of classes for weight inititalization
"""

from typing import Tuple

import numpy as np

from .tensor import Tensor

class WeightInitializer:
    """
    Base class that every weight inititualization inherits
    """

    def init(self, dimension: Tuple) -> Tensor:
        """
        Function that initializes weights

        dimension - Tuple
            - shape should be the dimensions of weights or biases in a layer

        returns:
            - Tensor which shape is identical to the dimension argmuent
        """

        raise NotImplementedError



class RandomWeightInitializer(WeightInitializer):
    """
    Initializer random weights
    """

    def init(self, dimension: Tuple) -> Tensor:
        return np.random.randn(*dimension)

class RandomUniformWeightInitializer(WeightInitializer):
    """
    Initializes uniform weights between -1 and 1
    """

    def init(self, dimension: Tuple) -> Tensor:
        return np.random.uniform(-1, 1, dimension)
       

class GlorotUniformWeightInitializer(WeightInitializer):
    """
    Uses Glorot uniform initialization, weights are uniform
    Glorot ensures small variance
    """

    def init(self, dimension: Tuple) -> Tensor:
        l = 0
        for d in dimension:
            l += d
        sd = np.sqrt(6.0 / l)
        return np.random.uniform(-sd, sd, dimension)

class GlorotNormalWeightInitializer(WeightInitializer):
    """
    Uses Glorot uniform initialization, weights are normaly distributed
    Glorot ensures small variance
    """

    def init(self, dimension: Tuple) -> Tensor:
        l = 0
        for d in dimension:
            l += d
        sd = np.sqrt(2.0 / l)
        return np.random.normal(0, sd, dimension)

class ZerosWeightInitializer(WeightInitializer):
    """
    Initializes a Tensor with every elements value 0
    """

    def init(self, dimension: Tuple) -> Tensor:
        return np.zeros(dimension)

class OnesWeightInitializer(WeightInitializer):
    """
    Initializes a Tensor with every elements value 0
    """

    def init(self, dimension: Tuple) -> Tensor:
        return np.ones(dimension)


# TO DO: He
