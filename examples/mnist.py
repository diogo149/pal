import gzip
import pickle
import os
import sys
import functools

import numpy as np
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics

import pylab
try:
    # optionally use seaborn if available
    import seaborn as sns
except ImportError:
    pass

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


if __name__ == "__main__":
    X_raw, y_raw = load_data()[0]
    mnist_class0 = 1
    mnist_class1 = 7
    model = "lr"
    idxs = (y_raw == mnist_class0) | (y_raw == mnist_class1)
    X = X_raw[idxs]
    y = (y_raw[idxs] == mnist_class1) + 0.0

    if model == "lr":
        clf = sklearn.linear_model.LogisticRegression()
    elif model == "sgd":
        # using SGD because NoBootstrapPredictionMostVariance only makes sense
        # with a stochastic model
        clf = sklearn.linear_model.SGDClassifier(shuffle=True, n_iter=5)
    elif model == "extratrees":
        clf = sklearn.ensemble.ExtraTreesClassifier()

    predict_fn = functools.partial(predict_model, clf)
    kwargs = dict(
        X=X,
        y=y,
        num_initial_samples=10,
        num_final_samples=100,
        predict_fn=predict_fn,
        test_size=0.5,
        seed=51,
        objective_fn=accuracy_fn,
    )

    rng = np.random.RandomState(kwargs["seed"])
    strategies = [
        ("random",
         pal.strategy.Random(rng)),
        ("pred_closest_to_0.5",
         pal.strategy.PredictionClosestToValue(0.5, predict_fn)),
        ("boostrap_most_variance",
         pal.strategy.BootstrapPredictionMostVariance(predict_fn)),
        ("pred_closest_to_0.5_once",
         pal.strategy.PredictionClosestToValueComputedOnce(0.5, predict_fn)),
        ("no_bootstrap_most_variance",
         pal.strategy.NoBootstrapPredictionMostVariance(predict_fn)),
    ]
    all_scores = []
    lines = []
    for label, strategy in strategies:
        scores = pal.simulate_sequential_binary(
            active_learning_fn=strategy,
            **kwargs
        )
        all_scores.append(scores)
        line, = pylab.plot(scores, label=label)
        lines.append(line)
    pylab.legend(handles=lines)
    pylab.savefig("mnist.png")
    try:
        pylab.show()
    except:
        pass
    pylab.clf()
