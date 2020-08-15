"""
Loss functions for measuring accuracy
"""

import numpy as np

from .tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

class MSE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        return 2 * (predicted - actual)

# TO DO: Accuracy, BinaryAccuracy, CategoricalAccuracy,
# BinaryCrossentropy, CategoricalCrossentropy, 
# SparseCategoricalCrossentropy, 
# MSE, RMSE, MeanAbsErr, MeanSquaredLogErr
# ACU, Precision, TruePositive, TrueNegative, FalsePositive,
# FalseNegative, Hinge, Squared and Categorical Hinge