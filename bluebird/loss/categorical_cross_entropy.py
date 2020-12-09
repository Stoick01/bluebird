"""
Categorical Cross Entropy
=========================

Used for multi class classification.

"""

import time

import numpy as np

from bluebird.tensor import Tensor

import bluebird.utils as utl

from .loss import Loss

class CategoricalCrossEntropy(Loss):
    """
    Categorical cross entropy.

    Example::

        loss = CategoricalCrossEntropy()
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

        predicted = utl.clip(predicted)
        return - np.sum(actual * utl.fix_overflow(np.log(predicted)))
            

    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        """
        Calculates the loss function.

        Args:
            predicted (:obj:`Tensor`): models output
            actual (:obj:`Tensor`): expected output

        Returns:
            float: gradient

        """

        predicted = utl.clip(predicted)
        return predicted - actual