"""
Neural Net
"""

# TO DO: Model from multiple models

# import warnings
# warnings.filterwarnings('ignore')

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

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        if not isinstance(layers, Sequence):
            raise TypeException("layers", "Sequence[Layer]")

        self.layers = layers

    def build(self, 
            iterator: DataIterator = BatchIterator(),
            loss: Loss = MSE(),
            optimizer: Optimizer = SGD()) -> None:

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
        for layer in self.layers:
            inputs = layer.forward(inputs, training=True)
        return inputs

    def backward(self, grad:Tensor) -> Tensor:
        for layer in self.get_layers():
            grad = utl.grad_clip(grad)
            grad = layer.backward(grad)
        return grad

    def get_layers(self) -> Iterator[Layer]:
        for layer in reversed(self.layers):
            yield layer

    def get_params_and_grads(self) -> Iterator[Tensor]:
        for layer in self.get_layers():
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

    def predict(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs, training=False)
        return inputs

    def fit(self, 
            inputs: Tensor,
            targets: Tensor,
            num_epochs: int = 5000) -> None:

        if not isinstance(inputs, Tensor):
            raise TypeException("inputs", "Tensor")

        if not isinstance(targets, Tensor):
            raise TypeException("targets", "Tensor")

        if not isinstance(num_epochs, int):
            raise TypeException("num_epochs", "int")

        n = len(inputs)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            items = 0

            bar = ProgressBar(n, num_epochs)

            for batch in self.iterator(inputs, targets):
                items += len(batch.inputs)
                predicted = self.predict(batch.inputs)
                epoch_loss += self.loss.loss(predicted, batch.targets) 
                grad = self.loss.grad(predicted, batch.targets)
                self.backward(grad)
                self.optimizer.step()
                bar.print_bar(items, epoch, epoch_loss/items)
            # print(epoch, epoch_loss/n)