import gzip
import pickle
import os
import sys
import functools

import numpy as np
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics

import pal


DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
DATA_FILENAME = 'mnist.pkl.gz'


def load_data(url=DATA_URL, filename=DATA_FILENAME):
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
        return pickle_load(f, encoding='latin-1')


def predict_model(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X_test)[:, 1]
    else:
        return clf.predict(X_test)


def accuracy_fn(y_true, preds):
    return sklearn.metrics.accuracy_score(y_true, preds >= 0.5)


def least_certain(X_train, train_idxs, y, predict_fn, objective_fn):
    X_tmp = X_train[train_idxs]
    preds = predict_fn(X_tmp, y, X_train)
    idxs = set(train_idxs)
    best_idx = None
    lowest_certainty = 1e5
    for idx, pred in enumerate(preds):
        if idx not in idxs:
            certainty = abs(pred - 0.5)
            if lowest_certainty > certainty:
                best_idx = idx
                lowest_certainty = certainty
    assert best_idx is not None
    return best_idx


def most_variance(X_train, train_idxs, y, predict_fn, objective_fn):
    X_tmp = X_train[train_idxs]
    all_preds = []
    for _ in range(10):
        idxs = np.random.randint(0, len(train_idxs), len(train_idxs))
        preds = predict_fn(X_tmp[idxs], y[idxs], X_train)
        all_preds.append(preds)
    stds = np.std(all_preds, axis=0)
    idxs = set(train_idxs)
    best_idx = None
    highest_variance = 0
    for idx, std in enumerate(stds):
        if idx not in idxs:
            if highest_variance < std:
                best_idx = idx
                highest_variance = std
    assert best_idx is not None
    return best_idx

if __name__ == "__main__":
    X_raw, y_raw = load_data()[0]
    mnist_class0 = 1
    mnist_class1 = 7
    idxs = (y_raw == mnist_class0) | (y_raw == mnist_class1)
    X = X_raw[idxs]
    y = (y_raw[idxs] == mnist_class1) + 0.0
    clf = sklearn.linear_model.LogisticRegression()
    scores1 = pal.simulate_sequential_binary(
        X,
        y,
        predict_fn=functools.partial(predict_model, clf),
        active_learning_fn=least_certain,
        objective_fn=accuracy_fn,
        num_initial_samples=10,
        num_final_samples=50,
        test_size=0.5)
    scores2 = pal.simulate_sequential_binary(
        X,
        y,
        predict_fn=functools.partial(predict_model, clf),
        active_learning_fn=most_variance,
        objective_fn=accuracy_fn,
        num_initial_samples=10,
        num_final_samples=50,
        test_size=0.5)
    import pylab
    p1, = pylab.plot(scores1, label="least_certain")
    p2, = pylab.plot(scores2, label="most_variance")
    pylab.legend(handles=[p1, p2])
    pylab.savefig("mnist.png")
