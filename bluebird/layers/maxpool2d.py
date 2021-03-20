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

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        if self.stride == None:
            self.stride = self.kernel_size
    

    def forward(self, inputs: Tensor, training: bool = False) -> Tensor:
        """
        Called each time the data passes throughout the nework.

        Args:
            inputs (:obj:`Tensor`): output from the previous layer
            training (bool, optional): set to true during training, and is false when network predicts

        Returns:
            :obj:`Tensor`: processed input data
        
        """

        self.inputs = inputs

        (n, height, width, channels) = inputs.shape
        f = self.kernel_size
        
        new_height = int((height - f)/self.stride) + 1
        new_width = int((width - f)/self.stride) + 1
        out_channels = channels

        Z = np.zeros((n, new_height, new_width, out_channels))

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

                        slic = inputs[i, v_start:v_end, h_start:h_end, c]
                        
                        Z[i, h, w, c] =  np.max(slic)

        return Z

    def create_mask(self, a: Tensor) -> Tensor:
        """
        Creates the one hot max mask.

        Args:
            a (:obj:`Tensor`): tensor you wish to create the mask from

        Returns:
            :obj:`Tensor`: mask
        """
        
        return a.max() == a


    def backward(self, grad: Tensor) -> Tensor:
        """
        Used to calculate the gradients of weights and biases.

        Args:
            grad (:obj:`Tensor`): gradient from previous layer or loss function.

        Returns:
            :obj:`Tensor`: Gradient

        """

        self.grads['in'] = np.zeros(self.inputs.shape)
        f = self.kernel_size

        (n, height_prev, width_prev, channels_prev) = self.inputs.shape
        (n, height, width, channels) = grad.shape

        for i in range(n):
            a = self.inputs[i]
            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        v_start = h
                        v_end = v_start + f
                        h_start = w
                        h_end = h_start + f

                        slic = a[v_start:v_end, h_start:h_end, c]
                        mask = self.create_mask(slic)

                        self.grads['in'][i, v_start:v_end, h_start:h_end, c] += np.multiply(mask, grad[i, h, w, c])

        return self.grads['in']





