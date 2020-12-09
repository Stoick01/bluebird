"""
Loss
====

They measure how good your model performs, and are crucial in model optimization.

Each optimization step fist starts by calculating the loss.

"""

from .loss import Loss
from .categorical_cross_entropy import CategoricalCrossEntropy
from .mse import MSE
