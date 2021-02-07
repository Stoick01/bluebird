import unittest

from bluebird.weight_initializers import *

class TestWeight(unittest.TestCase):
    def test_weight_initializer(self):
        """
        Tests not implemented od base weights class
        """
        with self.assertRaises(NotImplementedError) as context:
            init = WeightInitializer()
            init.init((2, 2))
        
        self.assertEqual(context.exception.__class__, NotImplementedError, "Should raise not implemented on init()")

    def test_random_init(self):
        """
        Tests initialization of random weights
        """

        init = RandomWeightInitializer()

        self.assertEqual(init.init((5, 5)).shape, (5, 5), "Should have dimensions of (5, 5)")

    def test_random_uniform(self):
        """
        Tests initialization of random uniform weights
        """

        init = RandomUniformWeightInitializer()

        self.assertEqual(init.init((5, 5)).shape, (5, 5), "Should have dimensions of (5, 5)")
        self.assertLessEqual(init.init((5, 5)).max(), 1, "Should be 1")
        self.assertGreaterEqual(init.init((5, 5)).min(), -1, "Should be -1")

    def test_xavier_uniform(self):
        """Tests initialization of xavier uniform weights"""

        init = XavierUniformWeightInitializer()
        init = init.init((5, 5))

        self.assertEqual(init.shape, (5, 5), "Should have dimesnions of (5, 5)")

