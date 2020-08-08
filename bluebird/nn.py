"""
Neural Net
"""

from typing import Sequence, Iterator, Tuple

from .tensor import Tensor
from .layers import Layer, Input
from .loss import Loss, MSE
from .data import DataIterator, BatchIterator
from .optimizers import Optimizer, SGD
from .activations import Activation

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def build(self, 
            iterator: DataIterator = BatchIterator(),
            loss: Loss = MSE(),
            optimizer: Optimizer = SGD()) -> None:
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
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad:Tensor) -> Tensor:
        for layer in self.get_layers():
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
        return self.forward(inputs)

    def fit(self, 
            inputs: Tensor,
            targets: Tensor,
            num_epochs: int = 5000) -> None:

        n = len(inputs)

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for batch in self.iterator(inputs, targets):
                predicted = self.predict(batch.inputs)
                epoch_loss += self.loss.loss(predicted, batch.targets) * (len(batch.targets) / n)
                grad = self.loss.grad(predicted, batch.targets)
                self.backward(grad)
                self.optimizer.step()
            print(epoch, epoch_loss)