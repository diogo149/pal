import functools
import time

import numpy as np
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics

import pylab
try:
    # optionally use seaborn if available
    import seaborn
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
    model = "lr"
    seed = 51

    X, y = pal.data.binary_mnist(class0=1, class1=7)

    rng = np.random.RandomState(seed)
    # bernoulli noise
    # y = ((rng.rand(*y.shape) < 0.1) ^ y)
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
    elif model == "rf":
        clf = sklearn.ensemble.RandomForestClassifier(random_state=rng)
    elif model == "gbm":
        clf = sklearn.ensemble.GradientBoostingClassifier(random_state=rng)
    elif model == "ridge":
        clf = sklearn.linear_model.Ridge()

    # X = np.random.randn(1000, 50)
    # y = np.random.randn(1000, 3)
    predict_fn = functools.partial(predict_model, clf)
    kwargs = dict(
        X=X,
        y=y,
        num_initial_samples=10,
        num_final_samples=100,
        predict_fn=predict_fn,
        test_size=0.5,
        seed=seed,
        # FIXME
        objective_fn=accuracy_fn,
        # objective_fn=sklearn.metrics.roc_auc_score,
        # objective_fn=sklearn.metrics.mean_squared_error,
        stratified_sample=True,
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

    def foo(X_train, y_train, X_test):
        preds = predict_fn(X_train, y_train, X_test)
        return -np.abs(preds - 0.5)

    def foo2(X_train, y_train, X_test):
        all_preds = []
        for _ in range(3):
            # bootstrap sampling
            idxs = rng.randint(0, len(X_train), len(X_train))
            # getting prediction
            preds = predict_fn(X_train[idxs], y_train[idxs], X_test)
            all_preds.append(preds)
        all_preds = np.array(all_preds)
        return all_preds.swapaxes(0, 1).reshape(len(X_test), -1).std(axis=1)

    for label, strategy in strategies:
        start_time = time.time()
        result = pal.simulate.simulate_sequential(
            # active_learning_fn=strategy,
            score_fn=foo2,  # FIXME
            **kwargs
        )
        scores = result["test_scores"]
        print("%s took %f" % (label, time.time() - start_time))
        all_scores.append(scores)
        line, = pylab.plot(scores, label=label)
        lines.append(line)
        # FIXME delete
        break
    pylab.legend(handles=lines)
    pylab.savefig("mnist.png")
    try:
        pylab.show()
    except:
        pass
    pylab.clf()

    # FIXME
    # lsl = result["learning_scores_labeled"]
    # lsu = result["learning_scores_unlabeled"]
    # all_mins = min(ys.min()
    #                for xs in [lsl, lsu]
    #                for ys in xs)
    # all_maxs = max(ys.max()
    #                for xs in [lsl, lsu]
    #                for ys in xs)

    # def animate(nframe):
    #     pylab.clf()
    #     pylab.subplot(211)
    #     pylab.xlim(all_mins, all_maxs)
    #     pylab.hist(lsl[nframe], alpha=0.5, color="red", normed=True)
    #     pylab.hist(lsu[nframe], alpha=0.5, color="blue", normed=True)
    #     pylab.title('Label %d' % nframe)
    #     pylab.subplot(212)
    #     pylab.plot(scores[:nframe + 1])

    # fig = pylab.figure(figsize=(5, 4))
    # from matplotlib import animation
    # anim = animation.FuncAnimation(fig,
    #                                animate,
    #                                frames=len(lsl),
    #                                # 100ms per frame
    #                                interval=100,
    #                                repeat=True,
    #                                # 1 second
    #                                repeat_delay=1000)
    # anim.save('mnist.gif', writer='imagemagick')
    # pylab.clf()

    # pylab.hist(lsl[0], alpha=0.5, color="red", normed=True)
    # pylab.hist(lsu[0], alpha=0.5, color="blue", normed=True)
    # pylab.show()
    # pylab.hist(lsl[-1], alpha=0.5, color="red", normed=True)
    # pylab.hist(lsu[-1], alpha=0.5, color="blue", normed=True)
    # pylab.show()
