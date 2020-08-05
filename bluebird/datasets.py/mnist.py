"""
MNIST dataset
60 000 28*28 grayscale images of 10 digits
10 000 test set of images
"""

import os
import urllib
import gzip

def load_data(path="mnist"):
    PROJECT_ROOT_DIR = "."
    DOWNLOAD_ROOT = "http://yann.lecun.com/exdb/mnist/"
    FILENAMES = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    data_path = os.path.join(PROJECT_ROOT_DIR, path)
    os.makedirs(data_path, exist_ok=True)

    for filename in FILENAMES:
        url == DOWNLOAD_ROOT + filename
        urllib.request.urlretrieve(url, os.path.join(data_path, filename))

    