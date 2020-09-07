"""
Cross entropy function
"""

import time

import numpy as np

from bluebird.tensor import Tensor

import bluebird.utils as utl

from .loss import Loss

class CategoricalCrossEntropy(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        predicted = utl.clip(predicted)
        return - np.sum(actual * utl.fix_overflow(np.log(predicted)))
            

    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        predicted = utl.clip(predicted)
        return predicted - actual