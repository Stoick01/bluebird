"""
Utility functions
"""

import numpy as np

from .tensor import Tensor


def fix_overflow(x:Tensor) -> Tensor:
    """
    Ensures to fix infinite and not a number values
    
    Args:
        x: Tensor with nan and inf values, Type: Tensor

    Example:
        >>> x = fix_overflow(x)
    """

    return np.nan_to_num(x)

def clip(x:Tensor) -> Tensor:
    """
    Clips very small values from tensor

    Args:
        x: Type: Tensor

    Example:
        >>> x = clip(x)
    """
    x[x>0.9999999] = 0.9999999
    x[x<1e-7] = 1e-7
    return x

def grad_clip(x:Tensor) -> Tensor:
    """
    Clips too big and too small gradients

    Args:
        x: Gradient with too large or small values, Type: Tensor

    Example:
        >>> grad = grad_clip(grad)
    """
    x[x>5] = 5
    x[x<-5] = -5
    return x