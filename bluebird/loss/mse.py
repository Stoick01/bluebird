"""
Mean Squared Error
==================

Measures the averages of the squares of the errors.
"""

import numpy as np

from bluebird.tensor import Tensor
from .loss import Loss

class MSE(Loss):
    """
    Mean squared error

    Example::

        loss = MSE()
        net.build(loss=losss)
    
    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        """
        Calculates the loss function.

        Args:
            predicted (:obj:`Tensor`): models output
            actual (:obj:`Tensor`): expected output

        Returns:
            float: loss

        """

        return np.sum((predicted - actual) ** 2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        """
        Calculates the loss function.

        Args:
            predicted (:obj:`Tensor`): models output
            actual (:obj:`Tensor`): expected output

        Returns:
            float: gradient

        """

        return  (predicted - actual) * 2
