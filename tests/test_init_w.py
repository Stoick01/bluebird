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
        
        self.assertEqual(context.exception.__class__, NotImplementedError)

    def test_random_init(self):
        """
        Tests initialization of random weights
        """

        init = RandomWeightInitializer()

        self.assertEqual(init.init((5, 5)).shape, (5, 5))

    def test_random_uniform(self):
        """
        Tests initialization of random uniform weights
        """

        init = RandomUniformWeightInitializer()

        self.assertEqual(init.init((5, 5)).shape, (5, 5))
        self.assertLessEqual(init.init((5, 5)).max(), 1)
        self.assertGreaterEqual(init.init((5, 5)).min(), -1)


# class GlorotUniformWeightInitializer:
#     def init(self, dimension: Tuple) -> Tensor:
#         l = 0
#         for d in dimension:
#             l += d
#         sd = np.sqrt(6.0 / l)
#         return np.random.uniform(-sd, sd, dimension)

# class GlorotNormalWeightInitializer:
#     def init(self, dimension: Tuple) -> Tensor:
#         l = 0
#         for d in dimension:
#             l += d
#         sd = np.sqrt(2.0 / l)
#         return np.random.normal(0, sd, dimension)

# class ZerosWeightInitializer:
#     def init(self, dimension: Tuple) -> Tensor:
#         return np.zeros(dimension)

# class OnesWeightInitializer:
#     def init(self, dimension: Tuple) -> Tensor:
#         return np.ones(dimension)
