import numpy as np

from bluebird.datasets.mnist import load_data

(X_train, y_train), (X_test, y_test) = load_data()

print(X_train.shape, X_test.shape)