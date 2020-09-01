"""
Basic loss function defenition
"""

import numpy as np

from bluebird.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError


# TO DO: Accuracy, BinaryAccuracy, CategoricalAccuracy
# SparseCategoricalCrossentropy, 
# RMSE, MeanAbsErr, MeanSquaredLogErr
# ACU, Precision, TruePositive, TrueNegative, FalsePositive,
# FalseNegative, Hinge, Squared and Categorical Hinge,
# Kullback-Leibler