"""
Linear layer
============

Linear layer is one of the core layers, it calculates the ouptut based on weights and bias.

Outuput is equal to input multiplied by weights plus bias.
"""

import numpy as np

from bluebird.tensor import Tensor
from bluebird.weight_initializers import WeightInitializer, ZerosWeightInitializer, HeWeightInitializer
import bluebird.utils as utl

from .layer import Layer

class Linear(Layer):
    """
    It calculates the output based on formula:

    output = input @ weights + bias

    Example::

        linear = Lainear(50)
        net = NeuralNet([
                ...
                linear,
                ...
            ])
    """

    def __init__(self, output_size: int,
                 weight_initializer: WeightInitializer = HeWeightInitializer(),
                 bias_initializer: WeightInitializer = ZerosWeightInitializer()) -> None:
        """
        Initializes the object.

        Args:
            output_size (int): dimension of the output
            activation (:obj:`Activation`): activation function
            weight_initializer (:obj:`WeightInitializer`, optional): defines how weights are initialized, defaults to HeWeightInitializer
            bias_initializer (:obj:`WeightInitializer`, optional): defines how weights are initialized, defaults to ZerosWeightInitializer
        """

        if not isinstance(output_size, int):
            raise TypeException("output_size", "int")

        if not isinstance(weight_initializer, WeightInitializer):
            raise TypeException("weight_initializer", "WeightInitializer")

        if not isinstance(bias_initializer, WeightInitializer):
            raise TypeException("bias_initializer", "WeightInitializer")

        super().__init__()
        self.output_size = output_size

        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_size: int):
        """
        Called by the model, before its training step.

        It sets the input size and initializes the weights.

        Args:
            input_size (int): output size from previous layer

        """

        self.input_size = input_size

        self.params["w"] = self.weight_initializer.init((input_size, self.output_size))
        self.params["b"] = self.bias_initializer.init((self.output_size,))
        

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
        return utl.fix_overflow(inputs @ self.params["w"] + self.params["b"])

    def backward(self, grad: Tensor) -> Tensor:
        """
        Used to calculate the gradients of weights and biases.

        Args:
            grad (:obj:`Tensor`): gradient from previous layer or loss function.

        Returns:
            :obj:`Tensor`: Gradient

        """

        self.grads["b"] = utl.fix_overflow(np.sum(grad, axis=0))
        self.grads["w"] = utl.fix_overflow(self.inputs.T @ grad)
        return utl.fix_overflow(grad @ self.params["w"].T)