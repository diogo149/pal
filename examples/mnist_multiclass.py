import functools
import time

import numpy as np
import sklearn.linear_model
import sklearn.ensemble

import pal


if __name__ == "__main__":
    model = "lr"
    seed = 51
    num_initial_samples = 10
    num_final_samples = 100
    samples_per_step = 5
    test_size = 0.5
    objective_fn = pal.ml_utils.accuracy_2d

    X, y = pal.data.multiclass_mnist(range(10))

    rng = np.random.RandomState(seed)
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

    predict_fn = functools.partial(pal.ml_utils.predict_model, clf)
    strategies = [
        ("random",
         pal.strategy.Random(rng)),
        ("pred_closest_to_0.5",
         pal.strategy.PredictionClosestToValue(0.5,
                                               predict_fn)),
        ("pred_closest_to_0.5_euclidean",
         pal.strategy.PredictionClosestToValue(0.5,
                                               predict_fn,
                                               distance_fn="euclidean")),
        ("entropy",
         pal.strategy.Entropy(predict_fn)),
        ("boostrap_most_variance",
         pal.strategy.BootstrapPredictionMostVariance(predict_fn,
                                                      rng,
                                                      num_samples=4)),
        ("pred_closest_to_0.5_once",
         pal.strategy.PredictionClosestToValueComputedOnce(0.5,
                                                           predict_fn)),
    ]
    labels = [x[0] for x in strategies]

    all_scores = []
    lines = []
    states = []
    for label, strategy in strategies:
        start_time = time.time()
        state = pal.analysis.initial_state(X, y, seed)
        state = pal.analysis.train_test_split(state, test_size=test_size)
        state = pal.analysis.simulate_indices(state,
                                              strategy,
                                              num_initial_samples,
                                              num_final_samples,
                                              samples_per_step,
                                              stratified_sample=True)
        state = pal.analysis.calculate_indices_objectives(state,
                                                          predict_fn,
                                                          objective_fn)
        state = pal.analysis.labeled_and_unlabeled_scores(state)
        print("%s took %f" % (label, time.time() - start_time))
        states.append(state)

    # only want to calculate once, because it's expensive
    states[0] = pal.analysis.objective_upper_bound(states[0],
                                                   predict_fn,
                                                   objective_fn)

    with pal.viz.plot_to("mnist_multiclass_obj.png"):
        pal.viz.plot_objective_values(states, labels)

    for label, state in zip(labels, states):
        # show initial and final states
        with pal.viz.plot_to("mnist_multiclass_distribution_%s.png" % label):
            pal.viz.plot_labeled_unlabeled_score_distibutions(state)
        # create a gif as well
        pal.viz.animate_labeled_unlabeled_score_distibutions(
            state, "mnist_multiclass_distribution_%s.gif" % label)
