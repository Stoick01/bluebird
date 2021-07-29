import unittest

import numpy as np

from bluebird.activations import *

from .test_helpers import grad_calc_activ

class TestRelu(unittest.TestCase):
    
    def test_input_grad(self):

        relu = Relu()

        x = np.random.randn(5, 5)

        diff = grad_calc_activ(x, relu)

        assert diff < 1e-8, "Gradient not calculated properly"

class TestSigmoid(unittest.TestCase):
    
    def test_input_grad(self):

        sigmoid = Sigmoid()

        x = np.random.randn(5, 5)

        diff = grad_calc_activ(x, sigmoid)

        assert diff < 1e-8, "Gradient not calculated properly"


class TestTanh(unittest.TestCase):
    
    def test_input_grad(self):

        tanh = Tanh()

        x = np.random.randn(5, 5)

        diff = grad_calc_activ(x, tanh)

        assert diff < 1e-8, "Gradient not calculated properly"

class TestLeakyRely(unittest.TestCase):
    
    def test_input_grad(self):

        leaky = LeakyRelu(0.02)

        x = np.random.randn(5, 5)

        diff = grad_calc_activ(x, leaky, 1e-8, 0.02)

        assert diff < 1e-8, "Gradient not calculated properly"