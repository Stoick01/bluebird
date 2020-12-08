"""
Utility functions
=================

Small utility functions.
"""

import numpy as np

from .tensor import Tensor


def fix_overflow(x:Tensor) -> Tensor:
    """
    Ensures to fix infinite and not a number values.

    Example::

        x = fix_overflow(x)
    
    Args:
        x (:obj:`Tensor`): Tensor with nan and inf values

    Returns:
        :obj:`Tensor`: Tensor without inf and nan values

    """

    return np.nan_to_num(x)

def clip(x:Tensor) -> Tensor:
    """
    Clips very small values from tensor, used in cross entropy.

    Example::

        x = clip(x)

    Args:
        x (:obj:`Tensor`): Tensor with too small values

    Returns:
        :obj:`Tensor`: Cliped Tensor
        
    """
    x[x>0.9999999] = 0.9999999
    x[x<1e-7] = 1e-7
    return x

def grad_clip(x:Tensor) -> Tensor:
    """
    Clips too big and too small gradients.

    Example::
    
        grad = grad_clip(grad)

    Args:
        x(:obj:`Tensor`): Gradient with too large or small values

    Returns:
        :obj:`Tensor`: Cliped Gradient
        
    """
    x[x>5] = 5
    x[x<-5] = -5
    return x