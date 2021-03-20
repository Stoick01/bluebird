import unittest

import numpy as np

from bluebird.layers import *
from bluebird.nn import NeuralNet
from bluebird.activations import *


class TestMaxPool2D(unittest.TestCase):

    def test_forward(self):
        """Test forward propagation for MaxPool2D"""

        pool = MaxPool2D(kernel_size=3)
        
        x = np.random.randn(5, 6, 9, 3)
        a = pool.forward(x)

        assert a.shape == (5, 2, 3, 3)
    
    # def test_weight_grad(self):
    #     """Tests the input grad for MaxPool2D layer"""

    #     layer = MaxPool2D(kernel_size=3)

    #     x = np.random.randn(5, 6, 9, 3)

    #     diff = grad_calc(x, layer)
    #     print(diff)

    #     for key, val in diff.items():
    #         assert val < 1e-7, "Gradient of " + key + " not calculated properly"