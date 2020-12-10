"""
Layer
=====

All other layers and activations inherit base Layer class.
"""

from typing import Dict

import numpy as np

from bluebird.tensor import Tensor

class Layer:
    """
    Default layer, that all other layers and activations inherit.

    Example::

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
        Used to finalize building layers.

        Important to note, you should set the input_size for the layer in here.
        Does not apply to the activation functions, you don't need implement it in them.

        Args:
            input_size (int): output size from previous layer

        Raises:
            NotImplementedError
        """

        raise NotImplementedError

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        """
        Called each time the data passes throughout the nework.

        Args:
            inputs (:obj:`Tensor`): output from the previous layer
            training (bool, optional): set to true during training, and is false when network predicts

        Raises:
            NotImplementedError
        
        """

        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        Used to calculate the gradients of weights and biases.

        Args:
            grad (:obj:`Tensor`): gradient from previous layer or loss function.

        Raises:
            NotImplementedError

        """

        raise NotImplementedError

# TO DO: Masking, Lambda, Convolution (1D, 2D, 3D, Seperable, 
# Depthwise, Transpose), Pooling (Max, Average, GlobalMax, 
# GlobalAverage), Recurrent (LSTM, GRU, RNN), BatchNormalization,
#  LayerNormalization, SpatialDropout, GaussianDropout,  
# 