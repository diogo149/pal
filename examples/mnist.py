import functools
import time

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


def predict_model(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X_test)[:, 1]
    else:
        return clf.predict(X_test)


def accuracy_fn(y_true, preds):
    return sklearn.metrics.accuracy_score(y_true, preds >= 0.5)


if __name__ == "__main__":
    X_raw, y_raw = pal.data.load_raw_mnist()
    mnist_class0 = 1
    mnist_class1 = 7
    model = "lr"
    seed = 51

    rng = np.random.RandomState(seed)
    idxs = (y_raw == mnist_class0) | (y_raw == mnist_class1)
    X = X_raw[idxs]
    y = y_raw[idxs] == mnist_class1
    # bernoulli noise
    y = ((rng.rand(*y.shape) < 0.1) ^ y)
    y += 0.0

    if model == "lr":
        clf = sklearn.linear_model.LogisticRegression(random_state=rng)
    elif model == "sgd":
        # using SGD because NoBootstrapPredictionMostVariance only makes sense
        # with a stochastic model
        clf = sklearn.linear_model.SGDClassifier(shuffle=True,
                                                 n_iter=100,
                                                 random_state=rng)
    elif model == "extratrees":
        clf = sklearn.ensemble.ExtraTreesClassifier(random_state=rng)

    predict_fn = functools.partial(predict_model, clf)
    kwargs = dict(
        X=X,
        y=y,
        num_initial_samples=10,
        num_final_samples=100,
        predict_fn=predict_fn,
        test_size=0.5,
        seed=seed,
        # objective_fn=accuracy_fn,
        objective_fn=sklearn.metrics.roc_auc_score,
    )

    strategies = [
        ("random",
         pal.strategy.Random(rng)),
        ("pred_closest_to_0.5",
         pal.strategy.PredictionClosestToValue(0.5, predict_fn)),
        ("boostrap_most_variance",
         pal.strategy.BootstrapPredictionMostVariance(predict_fn, rng)),
        ("pred_closest_to_0.5_once",
         pal.strategy.PredictionClosestToValueComputedOnce(0.5, predict_fn)),
    ]
    strategies = [
        ("2",
         pal.strategy.BootstrapPredictionMostVariance(predict_fn,
                                                      rng,
                                                      num_predictions=2)),
        ("3",
         pal.strategy.BootstrapPredictionMostVariance(predict_fn,
                                                      rng,
                                                      num_predictions=3)),
        ("4",
         pal.strategy.BootstrapPredictionMostVariance(predict_fn,
                                                      rng,
                                                      num_predictions=4)),
    ]
    all_scores = []
    lines = []
    for label, strategy in strategies:
        start_time = time.time()
        scores = pal.simulate.sequential_binary(
            active_learning_fn=strategy,
            **kwargs
        )
        print("%s took %f" % (label, time.time() - start_time))
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
