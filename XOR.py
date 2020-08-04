import numpy as np

from bluebird.nn import NeuralNet
from bluebird.layers import Linear, Input
from bluebird.activations import Tanh

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]    
])

net = NeuralNet([
    Input(2),
    Linear(2),
    Tanh(),
    Linear(2)
])

net.build()
net.fit(inputs, targets)

for x, y in zip(inputs, targets):
    predicted = net.predict(x)
    print(x, predicted, y)