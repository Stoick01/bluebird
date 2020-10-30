"""
Set of classes for weight inititalization
"""

from typing import Tuple

import numpy as np

from .tensor import Tensor

class WeightInitializer:
    """
    Base class that every weight inititualization inherits

    Example:
        class CustomWeightInitializer(WeightInitializer):
            def init(self, dimension: Tuple) -> Tensor:
                x = ... x must be a Tensor of dimensions given to the function
                return x
    """

    def init(self, dimension: Tuple) -> Tensor:
        raise NotImplementedError



class RandomWeightInitializer(WeightInitializer):
    """
    Initializes random values for tensor
    """

    def init(self, dimension: Tuple) -> Tensor:
        """
        Args:
            dimension: dimensions of Tensor that init returns, Type: Tuple

        Example:
            >>> weight_init = RandomWeightInitializer()
            >>> w = weight_init.init((10, 20))
        """

        return np.random.randn(*dimension)

class RandomUniformWeightInitializer(WeightInitializer):
    """
    Initializes random uniform values between -1 and 1
    """

    def init(self, dimension: Tuple) -> Tensor:
        """
        Args:
            dimension: dimensions of Tensor that init returns, Type: Tuple

        Example:
            >>> weight_init = RandomUniformWeightInitializer()
            >>> w = weight_init.init((10, 20))
        """

        return np.random.uniform(-1, 1, dimension)
       

class XavierUniformWeightInitializer(WeightInitializer):
    """
    Uses Xavier weight initialization
    weights have small variance, with uniform distribution
    """

    def init(self, dimension: Tuple) -> Tensor:
        """
        Args:
            dimension: dimensions of Tensor that init returns, Type: Tuple

        Example:
            >>> weight_init = XavierUniformWeightInitializer()
            >>> w = weight_init.init((10, 20))
        """

        l = 0
        for d in dimension:
            l += d
        sd = np.sqrt(6.0 / l)
        return np.random.uniform(-sd, sd, dimension)

class XavierNormalWeightInitializer(WeightInitializer):
    """
    Uses Xavier uniform initialization, weights are normaly distributed
    Xavier ensures small variance
    """

    def init(self, dimension: Tuple) -> Tensor:
        """
        Args:
            dimension: dimensions of Tensor that init returns, Type: Tuple

        Example:
            >>> weight_init = XavierNormalWeightInitializer()
            >>> w = weight_init.init((10, 20))
        """

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
        """
        Args:
            dimension: dimensions of Tensor that init returns, Type: Tuple

        Example:
            >>> weight_init = ZerosWeightInitializer()
            >>> w = weight_init.init((10, 20))
        """
        
        return np.zeros(dimension)

class OnesWeightInitializer(WeightInitializer):
    """
    Initializes a Tensor with every elements value 1
    """

    def init(self, dimension: Tuple) -> Tensor:
        """
        Args:
            dimension: dimensions of Tensor that init returns, Type: Tuple

        Example:
            >>> weight_init = OnesWeightInitializer()
            >>> w = weight_init.init((10, 20))
        """

        return np.ones(dimension)


class HeWeightInitializer(WeightInitializer):
    """
    He initializer, initializes weights with small
    """

    def init(self, dimension: Tuple) -> Tensor:
        """
        Args:
            dimension: dimensions of Tensor that init returns, Type: Tuple

        Example:
            >>> weight_init = HeWeightInitializer()
            >>> w = weight_init.init((10, 20))
        """
        return np.random.randn(dimension[0], dimension[1]) * np.sqrt(2.0 / dimension[0])