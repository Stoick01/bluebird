"""
2D Convolution layer
====================

Convolution layers are used to develop convolutional neural netowrks that are most commonly used to analyze images.
"""

import numpy as np

from bluebird.tensor import Tensor
from bluebird.weight_initializers import  ZerosWeightInitializer, OnesWeightInitializer
import bluebird.utils as utl

from .layer import Layer

class Conv2D(Layer):
    """
    Applies a 2D convolution over and input.

    It has 3 dimensions, width, height and depth.

    Example::

        conv = Conv2D(32, kernel_size=5)
        net = NeuralNet([
                ...
                conv,
                ...
            ])
    """

    def __init__(self, out_channels, kernel_size=3, stride=1, padding=True):
        """
        Initializes the object.

        Args:
            out_channels (int): number of output channels
            kernel_size (int, optional): size of kernel matrix, defaults to 3
            stride (int, optional): determens how much the filter moves, defaults to 1
            padding (bool, optional): True if you want to add padding to the input image, defualts to True 
        """

        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def build(self, in_channels):
        """
        Builds the layer.

        Args:
            in_channels (int): number of input channels
        """

        self.in_channels = in_channels

        # self.params["w"] = self.weight_initializer.init((input_size, self.output_size))
        # self.params["b"] = self.bias_initializer.init((self.output_size,))

    def zero_padding(self, inp):
        """
        Add zero padding to the input Tensor.

        Args:
            inputs (:obj:`Tensor`): input to the layer

        Returns
            :obj:`Tensor`: padded input

        """

        pad_len = self.kernel_size - 1
        n, w, h, c = inp.shape
        padded = np.zeros((n, w+2*pad_len, h+2*pad_len, c), dtype=inp.dtype)
        padded[:, pad_len:-pad_len, pad_len:-pad_len, :] = inp

        return padded

    def step(self, inp, w, b):
        """
        Perform single step in convolution.

        Args:
            inp (:obj:`Tensor`): slice of input
            w (:obj:`Tensor`): weights
            b (:obj:`Tensor`): bias
        """

        return np.sum(inp @ w + b)

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
        (f, f, channels, out_channels) = self.params['w'].shape
        
        new_height = int((height + 2*padding - f)/stride) + 1
        new_width = int((width + 2*padding - f)/stride) + 1

        Z = np.zeros([n, new_height, new_width, out_channels])

        padded = inputs

        if self.padding:
            padded = self.zero_padding(inputs)

        # batch
        for i in range(n):
            inp = padded[i, :, :, :]
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

                        slic = inp[v_start:v_end, h_start:h_end, :]

                        Z[i, h, w, c] = self.step(slic, self.params['w'][:, :, :, c], self.params['b'][:, :, :, c])

        return Z

