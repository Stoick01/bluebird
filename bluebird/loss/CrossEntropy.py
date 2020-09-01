"""
Cross entropy function
"""

import time

import numpy as np

from bluebird.tensor import Tensor
from .loss import Loss

class CrossEntropy(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        # print(predicted)
        # print("--")
        # print(actual)
        # print("--")
        # print(actual * np.log(predicted))
        # print("---")
        # print(- np.sum(actual * np.log(predicted)) / len(actual))
        # print("----------------")
        # time.sleep(5)
        return - np.sum(actual * np.log(predicted)) / len(actual)
            

    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        return predicted - actual