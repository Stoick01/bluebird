"""
2D Convolution layer
====================

Convolution layers are used to develop convolutional neural netowrks that are most commonly used to analyze images.
"""

import numpy as np

from bluebird.tensor import Tensor
from bluebird.weight_initializers import  ZerosWeightInitializer, HeWeightInitializer
import bluebird.utils as utl

from .layer import Layer

class Conv2D(Layer):
    """
    Applies a 2D convolution over and input.

    Example::

        conv = Conv2D(32, kernel_size=5)
        net = NeuralNet([
                ...
                conv,
                ...
            ])
    """

    def __init__(self, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: bool = True):
        """
        Initializes the object.

        Args:
            out_channels (int): number of output channels
            kernel_size (int, optional): size of the window, defaults to 3
            stride (int, optional): determens how much the filter moves, defaults to 1
            padding (bool, optional): True if you want to add padding to the input image, defualts to True 
        """

        super().__init__()
        self.output_size = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def build(self, in_channels: int):
        """
        Builds the layer.

        Args:
            in_channels (int): number of input channels
        """

        self.input_size = in_channels

        weight_initializer = HeWeightInitializer()
        bias_initializer = ZerosWeightInitializer()

        self.params["w"] = weight_initializer.init((self.kernel_size, self.kernel_size, self.input_size, self.output_size))
        self.params["b"] = bias_initializer.init((1, 1, 1, self.output_size))

    def zero_padding(self, inp: Tensor) -> Tensor:
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

    def remove_zero_padding(self, inp: Tensor) -> Tensor:
        """
        Add zero padding to the input Tensor.

        Args:
            inputs (:obj:`Tensor`): input to the layer

        Returns
            :obj:`Tensor`: padded input

        """
        pad = self.kernel_size - 1
        return inp[pad:-pad, pad:-pad, :]


    def step(self, inp: Tensor, w: Tensor, b: Tensor) -> Tensor:
        """
        Perform single step in convolution.

        Args:
            inp (:obj:`Tensor`): slice of input
            w (:obj:`Tensor`): weights
            b (:obj:`Tensor`): bias
        """

        return np.sum(inp * w) + b

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
        (f, f, channels, out_channels) = self.params['w'].shape

        padding = self.kernel_size - 1
        
        new_height = int((height + 2*padding - f)/self.stride) + 1
        new_width = int((width + 2*padding - f)/self.stride) + 1

        Z = np.zeros((n, new_height, new_width, out_channels))

        padded = inputs

        if self.padding:
            padded = self.zero_padding(inputs)

        self.padded = padded


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

    def backward(self, grad: Tensor) -> Tensor:
        """
        Used to calculate the gradients of weights and biases.

        Args:
            grad (:obj:`Tensor`): gradient from previous layer or loss function.

        Returns:
            :obj:`Tensor`: Gradient

        """
        
        (n, height_prev, width_prev, channels_prev) = self.inputs.shape

        (f, f, channels_prev, channels) = self.params['w'].shape

        (n, height, width, channels) = grad.shape

        self.grads['in'] = np.zeros((n, height_prev, width_prev, channels_prev))
        self.grads['w'] = np.zeros((f, f, channels_prev, channels))
        self.grads['b'] = np.zeros((1, 1, 1, channels))

        da_pad = np.zeros(self.padded.shape)

        for i in range(n):
            a = self.padded[i]
            da = da_pad[i]

            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        v_start = h
                        v_end = v_start + f
                        h_start = w
                        h_end = h_start + f

                        slic = a[v_start:v_end, h_start:h_end, :]

                        da[v_start:v_end, h_start:h_end, :] += self.params['w'][:, :, :, c] * grad[i, h, w, c]
                        self.grads['w'][:, :, :, c] += slic * grad[i, h, w, c]
                        self.grads['b'][:, :, :, c] += grad[i, h, w, c]

            self.grads['in'][i, :, :, :] = self.remove_zero_padding(da)

        return self.grads['in']

