"""
Basic layer arhitecture
All other layers and activations inherit this basic arhitecture
"""

from typing import Dict

import numpy as np

from bluebird.tensor import Tensor

class Layer:
    """
    Default layer, that all other layers and activations inherit

    Example:

       class CustomLayer(Layer):
            def build(self, input_size) -> None:
                super().__init__()
                ... Make sure you do initialize input_size(it's inherited from previous layer)

            def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
                ... when data is passing forward

            def backward(self, grad: Tensor) -> Tensor:
                ... Make shoure you calculate gradients for weights and biases if needed
                ... (self.grads['w'] and self.grads['b'])

    """

    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        self.train = True
        self.test = True

    def build(self, input_size) -> None:
        """
        Used to finalize building layers
        """
        raise NotImplementedError

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        """
        Produces output for inputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagates the gradinet through the layer
        """
        raise NotImplementedError

# TO DO: Masking, Lambda, Convolution (1D, 2D, 3D, Seperable, 
# Depthwise, Transpose), Pooling (Max, Average, GlobalMax, 
# GlobalAverage), Recurrent (LSTM, GRU, RNN), BatchNormalization,
#  LayerNormalization, SpatialDropout, GaussianDropout,  
# 