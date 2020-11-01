"""
MNIST dataset
60 000 28*28 grayscale images of 10 digits
10 000 test set of images
"""

import os
import urllib.request as urllib
import gzip
import pickle

from typing import Tuple

import numpy as np

def load_data(path: str = "mnist", save: bool = False) -> Tuple:
    PROJECT_ROOT_DIR = "."
    DOWNLOAD_ROOT = "http://yann.lecun.com/exdb/mnist/"
    FILENAMES = [
        ["training_images","train-images-idx3-ubyte.gz"],
        ["test_images","t10k-images-idx3-ubyte.gz"],
        ["training_labels","train-labels-idx1-ubyte.gz"],
        ["test_labels","t10k-labels-idx1-ubyte.gz"]
    ]

    data = {}

    data_path = os.path.join(PROJECT_ROOT_DIR, path)
    if save:
        os.makedirs(data_path, exist_ok=True)

    for filename in FILENAMES:
        url = DOWNLOAD_ROOT + filename[1]
        print("Downloading " + filename[1])
        with urllib.urlopen(url) as response:
            with gzip.GzipFile(fileobj=response) as f:
                if "labels" in filename[0]:
                    data[filename[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
                else:
                    data[filename[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)

                if save:
                    np.savetxt(os.path.join(data_path, filename[0] + '.csv'), data[filename[0]], fmt='%i', delimiter=',')
                
    data["training_images"] = np.reshape(data["training_images"], (data["training_images"].shape[0], 28, 28))

    data["test_images"] = np.reshape(data["test_images"], (data["test_images"].shape[0], 28, 28))

    return (data["training_images"], data["training_labels"]), (data["test_images"], data["test_labels"])

        