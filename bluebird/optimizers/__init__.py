"""
Optimizers
==========

Optimizers are used to update weights.

Each optimizer is inherited from base optimizer class, and you pass them to your model during build step.

``net.build(optimizer=AdaGrad(lr=0.003))``

See bellow information on all other optimizers.
"""

from .optimizer import Optimizer
from .nestov_momentum import NestovMomentum
from .sgd import SGD
from .adagrad import AdaGrad
from .adam import Adam