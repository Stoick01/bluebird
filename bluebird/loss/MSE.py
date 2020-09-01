"""
Cross entropy function
"""

import numpy as np

from bluebird.tensor import Tensor
from .loss import Loss

class MSE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2) / len(actual)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        return  (predicted - actual) * 2
