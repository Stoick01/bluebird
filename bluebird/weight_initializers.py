"""
Weight initializers
===================

Set of classes for weight inititalization.
"""

from typing import Tuple

import numpy as np

from .tensor import Tensor

class WeightInitializer:
    """
    Base class that every weight inititualization.

    This is the base class that all other weight initializers use, funciton init should be implemented that it returns a Tensor with the initialized weights.

    Example::

        class CustomWeightInitializer(WeightInitializer):
            def init(self, dimension: Tuple) -> Tensor:
                x = ... x must be a Tensor of dimensions given to the function
                return x
        
    """

    def init(self, dimension: Tuple) -> Tensor:
        """
        Initializes the weights.

        Args:
            dimension (Tuple): dimensions of weights in a network (input, output)

        Returns:
            :obj:`Tensor`: Tensor with dimensions of (input, output), when implemented

        Raises:
            NotImplementedError

        """
        raise NotImplementedError



class RandomWeightInitializer(WeightInitializer):
    """
    Initializes weights with random values.

    Example::

        weight_init = RandomWeightInitializer()
        w = weight_init.init((10, 20))

    """

    def init(self, dimension: Tuple) -> Tensor:
        """
        Initializes the weights.

        Args:
            dimension (Tuple): dimensions of weights in a network (input, output)

        Returns:
            :obj:`Tensor`: Tensor with dimensions of (input, output)

        """

        return np.random.randn(*dimension)

class RandomUniformWeightInitializer(WeightInitializer):
    """
    Initializes random uniform values between -1 and 1.

    Example::
        
        weight_init = RandomUniformWeightInitializer()
        w = weight_init.init((10, 20))

    """

    def init(self, dimension: Tuple) -> Tensor:
        """
        Initializes the weights.

        Args:
            dimension (Tuple): dimensions of weights in a network (input, output)

        Returns:
            :obj:`Tensor`: Tensor with dimensions of (input, output)

        """

        return np.random.uniform(-1, 1, dimension)
       

class XavierUniformWeightInitializer(WeightInitializer):
    """
    Uses Xavier weight initialization.
    Weights have small variance, with uniform distribution.

    Example::

        weight_init = XavierUniformWeightInitializer()
        w = weight_init.init((10, 20))
        
    """

    def init(self, dimension: Tuple) -> Tensor:
        """
        Initializes the weights.

        Args:
            dimension (Tuple): dimensions of weights in a network (input, output)

        Returns:
            :obj:`Tensor`: Tensor with dimensions of (input, output)

        """

        l = 0
        for d in dimension:
            l += d
        sd = np.sqrt(6.0 / l)
        return np.random.uniform(-sd, sd, dimension)

class XavierNormalWeightInitializer(WeightInitializer):
    """
    Uses Xavier uniform initialization, weights are normaly distributed.
    Xavier ensures small variance.

    Example::

        weight_init = XavierNormalWeightInitializer()
        w = weight_init.init((10, 20))

    """

    def init(self, dimension: Tuple) -> Tensor:
        """
        Initializes the weights.

        Args:
            dimension (Tuple): dimensions of weights in a network (input, output)

        Returns:
            :obj:`Tensor`: Tensor with dimensions of (input, output)

        """

        l = 0
        for d in dimension:
            l += d
        sd = np.sqrt(2.0 / l)
        return np.random.normal(0, sd, dimension)

class ZerosWeightInitializer(WeightInitializer):
    """
    Initializes a Tensor with all zeors.

    Example::

        weight_init = ZerosWeightInitializer()
        w = weight_init.init((10, 20))

    """

    def init(self, dimension: Tuple) -> Tensor:
        """
        Initializes the weights.

        Args:
            dimension (Tuple): dimensions of weights in a network (input, output)

        Returns:
            :obj:`Tensor`: Tensor with dimensions of (input, output)

        """
        
        return np.zeros(dimension)

class OnesWeightInitializer(WeightInitializer):
    """
    Initializes a Tensor with all ones.

    Example::

        weight_init = OnesWeightInitializer()
        w = weight_init.init((10, 20))

    """

    def init(self, dimension: Tuple) -> Tensor:
        """
        Initializes the weights.

        Args:
            dimension (Tuple): dimensions of weights in a network (input, output)

        Returns:
            :obj:`Tensor`: Tensor with dimensions of (input, output)

        """

        return np.ones(dimension)


class HeWeightInitializer(WeightInitializer):
    """
    He initializer, initializes weights with small variance.

    Example::
        
        weight_init = HeWeightInitializer()
        w = weight_init.init((10, 20))

    """

    def init(self, dimension: Tuple) -> Tensor:
        """
        Initializes the weights.

        Args:
            dimension (Tuple): dimensions of weights in a network (input, output)

        Returns:
            :obj:`Tensor`: Tensor with dimensions of (input, output)

        """
        return np.random.randn(dimension[0], dimension[1]) * np.sqrt(2.0 / dimension[0])