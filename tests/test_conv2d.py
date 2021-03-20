import unittest

import numpy as np

from bluebird.layers import *
from bluebird.nn import NeuralNet
from bluebird.activations import *

from .test_helpers import grad_calc_layers

class TestConv2D(unittest.TestCase):

    def test_forward(self):
        """Test forward propagation for Conv2D"""

        conv = Conv2D(3)
        conv.build(1)
        
        x = np.random.randn(5, 4, 3, 1)
        a = conv.forward(x)

        assert a.shape == (5, 6, 5, 3)
    
    def test_weight_grad(self):
        """Tests the input grad for Conv2D layer"""

        net = NeuralNet([
            Input(1),
            Conv2D(3),
            MaxPool2D(kernel_size=3),
            Tanh(),
            Flatten((2, 2, 3)),
            Dense(1, activation=Tanh())
        ])
        net.build()

        x = np.random.randn(1, 4, 4, 1)
        y = np.random.randn(1, 1)

        diff = grad_calc_layers(x, y, net)

        for key, val in diff.items():
            assert val < 1e-7, f"Gradient of {key} not calculated properly"