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


def accuracy(y_true, preds):
    return sklearn.metrics.accuracy_score(
        np.argmax(y_true, axis=1),
        np.argmax(preds, axis=1))


if __name__ == "__main__":
    model = "lr"
    seed = 51
    kwargs = dict(
        num_initial_samples=10,
        num_final_samples=100,
        samples_per_step=5,
        test_size=0.5,
        stratified_sample=True,
        objective_fn=accuracy,
    )

    # X, y = pal.data.binary_mnist(class0=1, class1=7)
    X, y = pal.data.multiclass_mnist(range(10))

    rng = np.random.RandomState(seed)
    # bernoulli noise
    # y = ((rng.rand(*y.shape) < 0.1) ^ y)
    y += 0.0

    if model == "lr":
        clf = sklearn.linear_model.LogisticRegression(random_state=seed)
    elif model == "sgd":
        # using SGD because NoBootstrapPredictionMostVariance only makes sense
        # with a stochastic model
        clf = sklearn.linear_model.SGDClassifier(shuffle=True,
                                                 n_iter=100,
                                                 random_state=seed)
    elif model == "extratrees":
        clf = sklearn.ensemble.ExtraTreesClassifier(random_state=seed)
    elif model == "rf":
        clf = sklearn.ensemble.RandomForestClassifier(random_state=seed)
    elif model == "gbm":
        clf = sklearn.ensemble.GradientBoostingClassifier(random_state=seed)
    elif model == "ridge":
        clf = sklearn.linear_model.Ridge()

    predict_fn = functools.partial(pal.utils.predict_model, clf)
    strategies = [
        ("random",
         pal.strategy.Random(rng)),
        ("pred_closest_to_0.5",
         pal.strategy.PredictionClosestToValue(0.5,
                                               predict_fn)),
        ("boostrap_most_variance",
         pal.strategy.BootstrapPredictionMostVariance(predict_fn,
                                                      rng,
                                                      num_samples=4)),
        ("pred_closest_to_0.5_once",
         pal.strategy.PredictionClosestToValueComputedOnce(0.5,
                                                           predict_fn)),
    ]

    kwargs.update(dict(
        X=X,
        y=y,
        predict_fn=predict_fn,
        seed=seed,
    ))
    all_scores = []
    lines = []
    for label, strategy in strategies:
        start_time = time.time()
        result = pal.simulate.simulate_sequential(
            score_fn=strategy,
            **kwargs
        )
        scores = result["test_scores"]
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
    #                                blit=True,
    #                                # 1 second (doesn't seem to work)
    #                                repeat_delay=1000)
    # anim.save('mnist.gif', writer='imagemagick')
    # pylab.clf()

    # pylab.hist(lsl[0], alpha=0.5, color="red", normed=True)
    # pylab.hist(lsu[0], alpha=0.5, color="blue", normed=True)
    # pylab.show()
    # pylab.hist(lsl[-1], alpha=0.5, color="red", normed=True)
    # pylab.hist(lsu[-1], alpha=0.5, color="blue", normed=True)
    # pylab.show()
