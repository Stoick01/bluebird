"""
Dense layer
===========

Combination of linear and activation layer.

It simeplefies the creation of your model.

"""

from typing import Dict

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bluebird.activations import Activation

import numpy as np

from .layer import Layer
from .linear import Linear

import bluebird as bb
from bluebird.tensor import Tensor
from bluebird.weight_initializers import WeightInitializer, HeWeightInitializer, ZerosWeightInitializer

from bluebird.exceptions import TypeException


class Dense(Layer):
    """
    Combination of linear and activation layer.

    It takes activation later, and output dimension for an argument, so that you don't have to specify them seperably.

    Also ihnerits the base Layer class.

    Example::

        dense = Dense(50, activation=Relu())
        net = NeuralNet([
                ...
                dense,
                ...
            ])

    """

    def __init__(self, output_size: int, activation: 'Activation', 
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

        # if not isinstance(activation, bb.activations.Activation):
        #     raise TypeException("activation", "Activation")

        if not isinstance(weight_initializer, WeightInitializer):
            raise TypeException("weight_initializer", "WeightInitializer")

        if not isinstance(bias_initializer, WeightInitializer):
            raise TypeException("bias_initializer", "WeightInitializer")

        self.output_size = output_size
        self.hidden = activation

        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_size: int) -> None:
        """
        Called by the model, before its training step.

        It sets the input size and initializes the weights.

        Args:
            input_size (int): output size from previous layer

        """

        self.layer = Linear(self.output_size, self.weight_initializer, self.bias_initializer)
        self.layer.build(input_size)
        
        self.input_size = input_size
        self.params = self.layer.params
        self.grads = self.layer.grads

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

        self.outputs = self.layer.forward(inputs, training)

        if self.hidden != None:
            self.outputs = self.hidden.forward(self.outputs, training)

        return self.outputs

    def backward(self, grad: Tensor) -> Tensor:
        """
        Used to calculate the gradients of weights and biases.

        Args:
            grad (:obj:`Tensor`): gradient from previous layer or loss function.

        Returns:
            :obj:`Tensor`: Gradient

        """

        self.grad = grad

        if self.hidden != None:
            self.grad = self.hidden.backward(self.grad)

        self.grad = self.layer.backward(self.grad)

        return self.grad