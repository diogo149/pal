import collections

import numpy as np
from sklearn import cross_validation

from . import ml_utils


def initial_state(X, y, seed):
    return dict(
        X=X,
        y=y,
        seed=seed,
    )


def train_test_split(state, test_size=0.5):
    """
    merges in a train and test split for X, y to the state
    """
    X = state["X"]
    y = state["y"]
    seed = state["seed"]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed
    )
    return dict(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        **state
    )


def simulate_indices(state,
                     score_fn,
                     num_initial_samples,
                     num_final_samples,
                     samples_per_step=1,
                     stratified_sample=False):
    """
    uses a score function to simulate choosing indices to label, and merges
    those indices into the state

    score_fn:
    takes in training features, training labels, and test features and
    returns a score of how good the test observation would be to label
    (higher is better)

    samples_per_step:
    how many points to label at each step of the algorithm
    """
    X_train = state["X_train"]
    y_train = state["y_train"]
    seed = state["seed"]

    labeled_idxs = list(
        ml_utils.train_test_split_indexes(
            np.argmax(y_train, axis=1),
            test_size=num_initial_samples,
            random_state=seed,
            stratified=stratified_sample
        )[1])
    assert len(labeled_idxs) == num_initial_samples

    idx_labeling_order = [labeled_idxs]
    labeling_scores = []
    labeled_idxs_history = [labeled_idxs]
    num_labeled = [num_initial_samples]
    while len(labeled_idxs) < num_final_samples:
        labeling_score = score_fn(X_train[labeled_idxs],
                                  y_train[labeled_idxs],
                                  X_train)

        sorted_idxs = [x[1] for x in list(sorted(zip(labeling_score,
                                                     range(len(X_train))),
                                                 reverse=True))]

        labeled_idxs_set = set(labeled_idxs)
        next_idxs = [idx for idx in sorted_idxs
                     if idx not in labeled_idxs_set][:samples_per_step]
        labeled_idxs = labeled_idxs + next_idxs

        labeling_scores.append(labeling_score)
        idx_labeling_order.append(next_idxs)
        labeled_idxs_history.append(labeled_idxs)
        num_labeled.append(len(labeled_idxs))
    return dict(
        num_labeled=num_labeled,
        labeling_scores=labeling_scores,
        idx_labeling_order=idx_labeling_order,
        labeled_idxs_history=labeled_idxs_history,
        **state
    )


def objective_upper_bound(state, predict_fn, objective_fn):
    """
    calculates an upper bound for a model and merges into the state
    """
    X_train = state["X_train"]
    y_train = state["y_train"]
    X_test = state["X_test"]
    y_test = state["y_test"]
    preds = predict_fn(X_train, y_train, X_test)
    obj = objective_fn(y_test, preds)
    return dict(
        objective_upper_bound=obj,
        **state
    )


def calculate_indices_objectives(state, predict_fn, objective_fn):
    X_train = state["X_train"]
    y_train = state["y_train"]
    X_test = state["X_test"]
    y_test = state["y_test"]
    labeled_idxs_history = state["labeled_idxs_history"]
    objs = []
    for labeled_idxs in labeled_idxs_history:
        preds = predict_fn(X_train[labeled_idxs],
                           y_train[labeled_idxs],
                           X_test)
        obj = objective_fn(y_test, preds)
        objs.append(obj)
    return dict(
        objective_values=objs,
        ** state
    )


def labeled_and_unlabeled_scores(state):
    """
    calculates for each time step all of the scores for the labeled and
    unlabeled points
    """
    y_train = state["y_train"]
    labeled_idxs_history = state["labeled_idxs_history"]
    labeling_scores = state["labeling_scores"]

    num_points = len(y_train)
    all_points = set(range(num_points))
    unlabeled_idxs_history = [list(all_points - set(labeled_idxs))
                              for labeled_idxs in labeled_idxs_history]

    labeled_scores = []
    unlabeled_scores = []
    assert len(labeled_idxs_history) - 1 == len(labeling_scores)
    for labeled_idxs, unlabeled_idxs, score in zip(labeled_idxs_history[1:],
                                                   unlabeled_idxs_history[1:],
                                                   labeling_scores):
        labeled_scores.append(score[labeled_idxs])
        unlabeled_scores.append(score[unlabeled_idxs])
    return dict(
        unlabeled_idxs_history=unlabeled_idxs_history,
        labeled_scores=labeled_scores,
        unlabeled_scores=unlabeled_scores,
        **state
    )


def oracle_goodness_of_indices(state,
                               predict_fn,
                               objective_fn,
                               rng,
                               sampled_points=10):
    """
    computes an approximation of "true" goodness of indices, ie.
    the percentile of the chosen points at increasing test accuracy

    sampled_points:
    how many points to sample to determine the percentile of the points we're
    chosen
    """
    X_train = state["X_train"]
    y_train = state["y_train"]
    X_test = state["X_test"]
    y_test = state["y_test"]
    labeled_idxs_history = state["labeled_idxs_history"]
    unlabeled_idxs_history = state["unlabeled_idxs_history"]
    idx_labeling_order = state["idx_labeling_order"]

    assert (len(idx_labeling_order)
            == len(labeled_idxs_history)
            == len(unlabeled_idxs_history))

    # start with 0 so list has same length as other lists
    oracle_goodness = [0.0]
    for (chosen_idxs,
         labeled_idxs,
         unlabeled_idxs) in zip(idx_labeling_order[1:],
                                labeled_idxs_history,
                                unlabeled_idxs_history):
        # note that the chosen_idxs are for the next time step, since
        # labeled and unlabeled idxs exist for the initial step
        for idx in chosen_idxs:
            assert idx in unlabeled_idxs

        def score_idx(idx):
            idxs = labeled_idxs + [idx]
            preds = predict_fn(X_train[idxs], y_train[idxs], X_test)
            return objective_fn(y_test, preds)

        sampled_idxs = rng.choice(np.arange(len(unlabeled_idxs)),
                                  size=sampled_points,
                                  replace=False)
        reference_scores = np.array(
            [score_idx(idx) for idx in sampled_idxs]
        )
        avg_goodness = np.mean(
            [(score_idx(idx) > reference_scores).mean()
             for idx in chosen_idxs]
        )
        oracle_goodness.append(avg_goodness)
    return dict(
        oracle_goodness=oracle_goodness,
        **state
    )


def before_and_after_scores(state):
    """
    for every observation that starts unlabeled but eventually is labeled,
    computes the average

    rationale: a potentially good litmus test for an algorithm is whether
    or not the scores get lower once they are labeled (presumably, they
    would be less good to label because they are already labeled)

    analyzing by observation may be necessary because some observations may
    just be hard
    """
    labeled_idxs_history = state["labeled_idxs_history"]
    unlabeled_idxs_history = state["unlabeled_idxs_history"]
    labeling_scores = state["labeling_scores"]
    assert (len(labeling_scores) + 1
            == len(labeled_idxs_history)
            == len(unlabeled_idxs_history))

    # intersection of points that have received scores when both unlabeled
    # and labeled
    unlabeled_then_labeled = list(
        # observations with scores when labeled
        set(labeled_idxs_history[-2])
        # and observations with scores when unlabeled
        & set(unlabeled_idxs_history[0]))

    before_and_after_diffs = []
    # add initial item to be aligned with other fields
    mean_before_and_after_diffs = [0.0]
    unlabeled = collections.defaultdict(list)
    labeled = collections.defaultdict(list)
    for scores, labeled_idxs in zip(labeling_scores,
                                    map(set, labeled_idxs_history)):
        for idx in unlabeled_then_labeled:
            score = scores[idx]
            if idx in labeled_idxs:
                labeled[idx].append(score)
            else:
                unlabeled[idx].append(score)

        diffs = []
        for idx in labeled.keys():
            assert labeled[idx]
            assert unlabeled[idx]
            diffs.append(np.mean(labeled[idx]) - np.mean(unlabeled[idx]))
        mean_diff = np.mean(diffs) if diffs else 0
        assert not np.isnan(mean_diff)

        before_and_after_diffs.append(diffs)
        mean_before_and_after_diffs.append(mean_diff)

    return dict(
        before_and_after_diffs=before_and_after_diffs,
        mean_before_and_after_diffs=mean_before_and_after_diffs,
        **state
    )
