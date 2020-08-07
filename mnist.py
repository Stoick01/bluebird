import numpy as np

from bluebird.datasets.mnist import load_data

import bluebird
from bluebird.nn import NeuralNet

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
    bluebird.layers.Flatten(input_size=(28, 28)),
    bluebird.layers.Dense(200, activation=bluebird.activations.Relu()),
    bluebird.layers.Dense(200, activation=bluebird.activations.Relu()),
    bluebird.layers.Dense(200, activation=bluebird.activations.Relu()),
    bluebird.layers.Dense(10, activation=bluebird.activations.Softmax())
])

net.build(optimizer=bluebird.optimizers.SGD(lr=0.03), iterator=bluebird.data.BatchIterator(batch_size=32))

net.fit(X_train, y_train, num_epochs=100)

print("Pred, True")

for (im, tst) in zip(X_test[:10], y_test[:10]):
    im = np.array([im])
    pred = net.predict(np.array(im))

    print("{b: > 4}, {c: > 4}".format(b=np.argmax(pred), c=np.argmax(tst)))