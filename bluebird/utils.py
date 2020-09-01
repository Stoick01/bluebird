"""
Utility functions
"""

import numpy as np

from .tensor import Tensor

def scale(x:Tensor) -> Tensor:
    x[x>1e8] = 1e8
    x[x<-1e8] = -1e8
    return x