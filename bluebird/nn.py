"""
Neural Net
==========

Default class for creating feed forward neural networs
"""

# TO DO: Model from multiple models, base model class

from typing import Sequence, Iterator, Tuple

from .tensor import Tensor
from .layers import Layer, Input
from .loss import Loss, MSE
from .data import DataIterator, BatchIterator
from .optimizers import Optimizer, SGD
from .activations import Activation
from .weight_initializers import WeightInitializer, RandomWeightInitializer

from .exceptions import TypeException
from .progress_tracker import ProgressBar

import bluebird.utils as utl


class Model():
    """
    Base model class.

    All other models inherit from it.

    Example::

        class CustomModel(Model):
            def __init__(self, layers: Sequence[Layer]):
                super.__init__()

            def build(self, 
                    iterator: DataIterator = BatchIterator(),
                    loss: Loss = MSE(),
                    optimizer: Optimizer = SGD()) -> None:
                super.build()

            def step(self, batch) -> float:
                ...

                return self.loss.loss(predicted, batch.targets) 
    """


    def __init__(self, layers: Sequence[Layer]) -> None:
        """
        Initalizes the object.

        Args:
            layers (Sequence[Layer]): sequence of layers of the newtwork

        """
        if not isinstance(layers, Sequence):
            raise TypeException("layers", "Sequence[Layer]")

        self.layers = layers

    def build(self, 
            iterator: DataIterator = BatchIterator(),
            loss: Loss = MSE(),
            optimizer: Optimizer = SGD()) -> None:
        """
        Used to build the model.

        Defines aditional parameters and initializes the weights.

        Args:
            iterator (:obj:`DataIterator`, otpional): defines the way you want to iteratre over the data,
                Defaults to BatchIterator(32)
            loss (:obj:`Loss`, otpional): defines loss function
                Defaults to MSE()
            optimizer (:obj:`Optimizer`, optional): defines how the weights should be updated
                Defaults to SGD()

        """
        

        if not isinstance(iterator, DataIterator):
            raise TypeException("iterator", "DataIterator")

        if not isinstance(loss, Loss):
            raise TypeException("loss", "Loss")

        if not isinstance(optimizer, Optimizer):
            raise TypeException("optimizer", "Optimizer")

        self.iterator = iterator
        self.loss = loss
        self.optimizer = optimizer

        dimension = 0
        for layer in self.layers:
            if isinstance(layer, Activation):
                continue

            if isinstance(layer, Input):
                layer.build()
                dimension = layer.output_size
                continue

            layer.build(dimension)
            dimension = layer.output_size

        self.optimizer.build(self)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward propagates through the network.

        Args:
            inputs (:obj:`Tensor`): Network input

        Returns:
            :obj:`Tensor`: Network output

        """

        for layer in self.layers:
            inputs = layer.forward(inputs, training=True)
        return inputs


    def backward(self, grad:Tensor) -> Tensor:
        """
        Backward propagates through the network.

        Args:
            inputs (:obj:`Tensor`): Gradient that you get from the loss function

        Returns:
            :obj:`Tensor`: Gradient from the first layer

        """

        for layer in self.get_layers():
            grad = utl.grad_clip(grad)
            grad = layer.backward(grad)
        return grad

    def get_layers(self) -> Iterator[Layer]:
        """
        Returns reversed layers.

        Returns:
            Iterator[:obj:`Tensor`]: List of layer objects

        """

        for layer in reversed(self.layers):
            yield layer

    def predict(self, inputs: Tensor) -> Tensor:
        """
        Used to predict values after you finished the training.

        Args:
            inputs (:obj:`Tensor`): values you wish to predict

        Returns: 
            :obj:`Tensor`: Predicted values
        
        """

        for layer in self.layers:
            inputs = layer.forward(inputs, training=False)
        return inputs

    def get_params_and_grads(self) -> Iterator[Tensor]:
        """
        Returns parameters and gradients for each layer.

        Returns:
            Iterator[:obj:`Tensor`]: List of parameter, gradient pairs

        """

        for layer in self.get_layers():
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def step(self, batch) -> float:
        """
        Step function is called during each training step.

        It calculates the loss and updates weights and biases.

        You must return loss.

        Args:
            batch (:obj:`Batch`): recives batch object, each batch object has inputs and targets

        Raises:
            NotImplementedError

        """

        raise NotImplementedError


    def fit(self, 
            inputs: Tensor,
            targets: Tensor,
            num_epochs: int) -> None:
        """
        Used to train the model.

        Args:
            inputs (:obj:`Tensor`): values used for training
            targets (:obj:`Tensor`): tarets
            num_epochs (int): number of epochs you want to train
            
        """
        

        if not isinstance(inputs, Tensor):
            raise TypeException("inputs", "Tensor")

        if not isinstance(targets, Tensor):
            raise TypeException("targets", "Tensor")

        if not isinstance(num_epochs, int):
            raise TypeException("num_epochs", "int")

        n = len(inputs)

        epoch_loss = 0.0
        items = 0
        for epoch in range(num_epochs):
            bar = ProgressBar(n, num_epochs)

            for batch in self.iterator(inputs, targets):
                items += len(batch.inputs)
                epoch_loss += self.step(batch)
                bar.print_bar(items - n*epoch, epoch+1, epoch_loss/items)



class NeuralNet(Model):
    """
    Creates a neural network model.

    Example::
        
        net = NeuralNet([
                Flatten(input_size=(28, 28)),
                Dense(300, activation=Relu()),
                Dense(100, activation=Relu()),
                Dense(50, activation=Relu()),
                Dense(10, activation=Softmax())
            ])
        net.build(optimizer=AdaGrad(lr=0.003), loss=CategoricalCrossEntropy())

        net.fit(X_train, y_train, num_epochs=20)

        net.predict(X_test)

    """


    def step(self, batch) -> float:
        """
        Step function is called during each training step.

        It calculates the loss and updates weights and biases.

        Args:
            batch (:obj:`Batch`): recives batch object, each batch object has inputs and targets

        Returns:
            float: returns calculated loss

        """
        predicted = self.predict(batch.inputs)

        loss = self.loss.loss(predicted, batch.targets) 
        grad = self.loss.grad(predicted, batch.targets)

        self.backward(grad)
        self.optimizer.step()

        return loss

                