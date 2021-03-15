"""
2D Max Pooling 
==============

Applies a 2D max pooling over an input signal.

Commonly used used in combination with 2D convolution layer.
"""

import numpy as np

from bluebird.tensor import Tensor
import bluebird.utils as utl

from .layer import Layer

class MaxPool2D(Layer):
    """
    Applies a 2D max pooling.

    Example::

        conv = MaxPool2D(kernel_size=5)
        net = NeuralNet([
                ...
                conv,
                ...
            ])
    """

    def __init__(self, kernel_size: int, stride: int = None):
        """
        Initalize the object.

        Args:
            kernel_size (int): size of the window
            stride (int): defines how much the window moves, defaults to kernel_size
        """

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        """
        Called each time the data passes throughout the nework.

        Args:
            inputs (:obj:`Tensor`): output from the previous layer
            training (bool, optional): set to true during training, and is false when network predicts

        Returns:
            :obj:`Tensor`: processed input data
        
        """

        (n, height, width, channels) = inputs.shape
        f = self.kernel_size
        
        new_height = int((height - f)/stride) + 1
        new_width = int((width - f)/stride) + 1
        out_channels = channels

        Z = np.zeros([n, new_height, new_width, out_channels])

        # batch
        for i in range(n):
            # height
            for h in range(new_height):
                # width
                for w in range(new_width):
                    # channel
                    for c in range(out_channels):
                        v_start = h * self.stride
                        v_end = h * self.stride + f
                        h_start = w * self.stride
                        h_end = w * self.stride + f

                        slic = inputs[i: v_start:v_end, h_start:h_end, c]

                        Z[i, h, w, c] =  np.max(slic)

        return Z