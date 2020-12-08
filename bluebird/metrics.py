"""
Metrics
=======

Functions designed to help you check how good is your model performing.
"""

import numpy as np

from .tensor import Tensor

def accuracy(x: Tensor, y: Tensor) -> float:
    """
    Checks how accurate your model is.

    Compares true and guessed values, and returns the precentage of how many values match.

    Args:
        x (:obj:`Tensor`): true values
        y (:obj:`Tensor`): guessed values

    Returns:
        float: Percentige of equal elements between two arrays
    """

    return np.sum(x == y) / len(x)