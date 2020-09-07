from typing import List

import numpy as np

from bluebird.nn import NeuralNet
from bluebird.layers import Linear, Input, Dense, Dropout
from bluebird.activations import Tanh, Relu, Softmax, Sigmoid
from bluebird.optimizers import SGD, NestovMomentum, AdaGrad
from bluebird.data import BatchIterator
from bluebird.loss import CategoricalCrossEntropy

def fizz_buzz_encode(x: int) -> List[int]:
    if x % 15 == 0:
        return [0, 0, 0, 1]
    if x % 5 == 0:
        return [0, 0, 1, 0]
    if x % 3 == 0:
        return [0, 1, 0, 0]
    return [1, 0, 0, 0]

def binary_encode(x: int) -> List[int]:
    """
    10 digit binary encoding of x
    """

    return [x >> i & 1 for i in range(10)]

inputs = np.array([
    binary_encode(x)
    for x in range(101, 1024)
])

targets = np.array([
    fizz_buzz_encode(x)
    for x in range(101, 1024)
])

net = NeuralNet([
    Input(10),
    Dense(50, activation=Relu()),
    Dense(4, activation=Softmax())
])


net.build(optimizer=NestovMomentum(lr=0.003), loss=CategoricalCrossEntropy())

net.fit(inputs, targets, num_epochs=5000)

for x in range(1, 101):
    predicted = net.predict(np.array([binary_encode(x)]))
    predicted_idx = np.argmax(predicted)
    actual_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), "fizz", "buzz", "fizzbuzz"]
    print(x, labels[predicted_idx], labels[actual_idx])