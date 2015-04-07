import sys
import gzip
import pickle
import os
import numpy as np


def load_raw_mnist():
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'

    PY2 = sys.version_info[0] == 2

    if PY2:
        from urllib import urlretrieve
        pickle_load = lambda f, encoding: pickle.load(f)
    else:
        from urllib.request import urlretrieve
        pickle_load = lambda f, encoding: pickle.load(f, encoding=encoding)

    if not os.path.exists(filename):
        print("Downloading MNIST")
        urlretrieve(url, filename)

    with gzip.open(filename, 'rb') as f:
        train_valid_test = pickle_load(f, encoding='latin-1')

    X, y = [np.concatenate(arrs) for arrs in zip(*train_valid_test)]
    return X, y


def binary_mnist(class0, class1):
    X, y = load_raw_mnist()
    idxs = (y == class0) | (y == class1)
    X_new = X[idxs]
    y_new = y[idxs] == class1
    return X_new, y_new
