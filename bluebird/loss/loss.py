"""
Loss
====

Base loss class that all other losses inherit.
"""

import numpy as np

from bluebird.tensor import Tensor

class Loss:
    """
    Base loss class that all other losses inherit.

    Example::

        class CustomLoss(Loss):
            def loss(self, predicted: Tensor, actual: Tensor) -> float:
                ... loss function
            
            def grad(self, predicted: Tensor, actual: Tensor) -> float:
                ... gradient of loss function

    """

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        """
        Calculates the loss function.

        Args:
            predicted (:obj:`Tensor`): models output
            actual (:obj:`Tensor`): expected output

        Raises:
            NotImplementedError

        """

        raise NotImplementedError
    
    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        """
        Calculates the loss function.

        Args:
            predicted (:obj:`Tensor`): models output
            actual (:obj:`Tensor`): expected output

        Raises:
            NotImplementedError

        """

        raise NotImplementedError


# TO DO: Accuracy, BinaryAccuracy, CategoricalAccuracy
# SparseCategoricalCrossentropy, 
# RMSE, MeanAbsErr, MeanSquaredLogErr
# ACU, Precision, TruePositive, TrueNegative, FalsePositive,
# FalseNegative, Hinge, Squared and Categorical Hinge,
# Kullback-Leibler