"""
Neural Net
"""

from typing import Sequence, Iterator, Tuple

from .tensor import Tensor
from .layers import Layer
from .loss import Loss, MSE
from .data import DataIterator, BatchIterator
from .optimizers import Optimizer, SGD
from .activation import Activation

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def build(self, 
            iterator: DataIterator = BatchIterator(),
            loss: Loss = MSE(),
            optimizer: Optimizer = SGD()):
        self.iterator = iterator
        self.loss = loss
        self.optimizer = optimizer

        shape = 0
        for layer in self.layers:
            if isinstance(layer, Activation):
                continue

            layer.build(shape)
            shape = layer.shape

        self.optimizer.build(self)

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def get_layers(self) -> Iterator[Layer]:
        for layer in reversed(self.layers):
            yield layer

    def predict(self, inputs: Tensor) -> Tensor:
        return self.forward(inputs)

    def fit(self, 
            inputs: Tensor,
            targets: Tensor,
            num_epochs: int = 5000) -> None:

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for batch in self.iterator(inputs, targets):
                predicted = self.predict(batch.inputs)
                epoch_loss += self.loss.loss(predicted, batch.targets)
                self.optimizer.step(predicted, batch.targets)
            print(epoch, epoch_loss / 100)