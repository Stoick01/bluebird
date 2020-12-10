"""
Layers
======

All layers inherit from base Layer class. 
Important to note is that activation functions are also implemented as layers.
"""

from .layer import Layer
from .input import Input
from .linear import Linear
from .dense import Dense
from .flatten import Flatten
from .dropout import Dropout