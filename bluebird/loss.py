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