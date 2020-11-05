import numpy as np

from .tensor import Tensor

def accuracy(x: Tensor, y: Tensor) -> float:
    """
    Compare number of equal elements in two arrays

    Args:
        x: type: Tensor 
        y: type: Tensor

    Returns:
        Percentige of equal elements between two arrays
    """

    return np.sum(x == y) / len(x)