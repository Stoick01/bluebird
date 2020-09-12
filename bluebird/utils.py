"""
Utility functions
"""

import numpy as np

from .tensor import Tensor


def fix_overflow(x:Tensor) -> Tensor:
    return np.nan_to_num(x)

def clip(x:Tensor) -> Tensor:
    x[x>0.9999999] = 0.9999999
    x[x<1e-7] = 1e-7
    return x

def grad_clip(x:Tensor) -> Tensor:
    x[x>5] = 5
    x[x<-5] = -5
    return x