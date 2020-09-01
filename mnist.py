import os

import numpy as np

from bluebird.datasets.mnist import load_data

import bluebird
from bluebird.nn import NeuralNet
from bluebird.activations import Relu, Sigmoid, Softmax
from bluebird.layers import Flatten, Dropout, Dense
from bluebird.data import BatchIterator
from bluebird.loss import CrossEntropy

if os.path.isdir('./mnist'):
    print("Loading data...")
    y_train = np.loadtxt(open('./mnist/training_labels.csv', 'rb'), delimiter=",")
    y_train = y_train.astype("int")
    y_test = np.loadtxt(open('./mnist/test_labels.csv', 'rb'), delimiter=",")
    y_test = y_test.astype("int")

    X_train = np.loadtxt(open('./mnist/training_images.csv', 'rb'), delimiter=",")
    X_train = X_train.reshape(-1, 28, 28)

    X_test = np.loadtxt(open('./mnist/test_images.csv', 'rb'), delimiter=",")
    X_test = X_test.reshape(-1, 28, 28)
else:
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
    Dense(10, activation=Softmax())
])

net.build(optimizer=bluebird.optimizers.SGD(lr=0.01), loss=CrossEntropy())

net.fit(X_train, y_train, num_epochs=100)

print("Pred, True")

for (im, tst) in zip(X_test[:10], y_test[:10]):
    im = np.array([im])
    pred = net.predict(np.array(im))

    print("{b: > 4}, {c: > 4}".format(b=np.argmax(pred), c=np.argmax(tst)))