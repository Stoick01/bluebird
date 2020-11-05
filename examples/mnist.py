import os
import sys

sys.path.append(os.path.abspath('../'))

import numpy as np

from bluebird.datasets.mnist import load_data

import bluebird
from bluebird.nn import NeuralNet
from bluebird.activations import Relu, Sigmoid, Softmax, Tanh
from bluebird.layers import Flatten, Dropout, Dense, Linear
from bluebird.loss import CategoricalCrossEntropy
from bluebird.metrics import accuracy

(X_train, y_train), (X_test, y_test) = load_data()

def convert_labels(labels: np.ndarray) -> np.ndarray:
    ar = np.zeros((labels.shape[0], 10))

    for (i, j) in zip(range(ar.shape[0]), labels):
        ar[i][j] = 1

    return ar

y_train = convert_labels(y_train)
y_test = convert_labels(y_test)
X_train = X_train / 255.0
X_test = X_test / 255.0

net = NeuralNet([
    Flatten(input_size=(28, 28)),
    Dense(300, activation=Relu()),
    Dense(100, activation=Relu()),
    Dense(50, activation=Relu()),
    Dense(10, activation=Softmax())
])

net.build(optimizer=bluebird.optimizers.Adam(lr=0.00001), loss=CategoricalCrossEntropy())

net.fit(X_train, y_train, num_epochs=20)

print("Pred, True")

for (im, tst) in zip(X_test[:10], y_test[:10]):
    im = np.array([im])
    pred = net.predict(np.array(im))

    print("{b: > 4}, {c: > 4}".format(b=np.argmax(pred), c=np.argmax(tst)))

p = net.predict(X_test)

p = p.argmax(axis=1)
y_test = y_test.argmax(axis=1)

print(accuracy(p, y_test))