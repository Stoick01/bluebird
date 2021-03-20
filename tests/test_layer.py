import unittest

import numpy as np

from bluebird.layers import *
from bluebird.nn import NeuralNet
from bluebird.activations import *

from .test_helpers import grad_calc_layers

class TestLayers(unittest.TestCase):
    
    def test_layer(self):
        """Tests implementation of functions in layer"""

        layer = Layer()

        with self.assertRaises(NotImplementedError) as context:
            layer.build(20)
        
        self.assertEqual(context.exception.__class__, NotImplementedError, "Should raise NotImplementedError")

        with self.assertRaises(NotImplementedError) as context:
            layer.forward(np.array([1, 2, 3, 4]))
        
        self.assertEqual(context.exception.__class__, NotImplementedError, "Should raise NotImplementedError")

        with self.assertRaises(NotImplementedError) as context:
            layer.backward(np.array([1, 2, 3, 4]))
        
        self.assertEqual(context.exception.__class__, NotImplementedError, "Should raise NotImplementedError")

class TestLinear(unittest.TestCase):
    
    def test_weight_grad(self):
        """Tests the grad for linear layer"""

        net = NeuralNet([
            Input(5),
            Linear(1),
            Sigmoid()
        ])
        net.build()

        x = np.random.uniform(-1, 1, (1, 5))
        y = np.array([1.0])

        diff = grad_calc_layers(x, y, net)

        for key, val in diff.items():
            assert val < 1e-7, "Gradient of " + key + " not calculated properly"



        