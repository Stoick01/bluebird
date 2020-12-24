"""
Batch Normalization
===================

Batch normalization makes network less dependent on initialization strategy.
It also enables us to use higher learning rate.
"""

import numpy as np

from bluebird.tensor import Tensor
from bluebird.weight_initializers import  ZerosWeightInitializer, OnesWeightInitializer
import bluebird.utils as utl

from .layer import Layer

class BatchNormalization(Layer):
    """
    Normalizes the data and applies linear transformation.

    Gamma and beta are reprisented with weights and biases foe ease of implementation with optimizers.
    
    input = (input - mean) / variance
    output = input * weights + bias

    Example::

        batch = BatchNormalization()
        net = NeuralNet([
                ...
                batch,
                ...
            ])
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initializes the object.
        
        Args:
            eps (float, optional): prevents division by zero, defaults to 1e-8
        """
        
        super().__init__()
        self.eps = eps


    def build(self, input_size: int) -> None:
        """
        Called by the model, before its training step.

        It sets the input size and initializes the weights.

        Args:
            input_size (int): output size from previous layer

        """

        self.input_size = input_size
        self.output_size = input_size

        gama_init = OnesWeightInitializer()
        beta_init = ZerosWeightInitializer()

        self.params["w"] = gama_init.init((self.input_size,))
        self.params["b"] = beta_init.init((self.output_size,))

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

        self.mean = np.mean(inputs, axis=0)
        self.var = np.var(inputs, axis=0)

        self.norm = (inputs - self.mean) / np.sqrt(self.var + self.eps)

        return self.params['w'] * inputs + self.params['b']

    def backward(self, grad: Tensor) -> Tensor:
        """
        Used to calculate the gradients of weights and biases.

        Args:
            grad (:obj:`Tensor`): gradient from previous layer or loss function.

        Returns:
            :obj:`Tensor`: Gradient

        """

        m = self.inputs.shape[0]
        mu = self.inputs - self.mean
        std_inv = 1.0 / np.sqrt(self.var + self.eps)

        d_norm = grad * self.params['w']
        d_var = np.sum(d_norm * mu, axis=0) * (-0.5) * std_inv**3
        d_mu = np.sum(d_norm * (-std_inv), axis=0) + d_var * np.mean(-2 * mu, axis=0)

        self.grads['w'] = np.sum(grad * self.norm, axis=0)
        self.grads['b'] = np.sum(grad, axis=0)

        return d_norm * std_inv + d_var * 2 * mu / m + d_mu / m

