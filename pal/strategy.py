"""
when writing new strategies, make sure that they work for both y as a vector
and y as a matrix
"""

import numpy as np
import scipy.stats


def aggregate_columns(vals, aggregator):
    if aggregator == "mean":
        return np.mean(vals, axis=1)
    elif aggregator == "std":
        return np.std(vals, axis=1)
    else:
        raise ValueError


def flexible_int_input(N, sample_size):
    if isinstance(sample_size, float):
        return int(np.round(N * sample_size))
    elif isinstance(sample_size, int):
        return sample_size
    elif isinstance(sample_size, str):
        if sample_size == "sqrt":
            return int(np.round(np.sqrt(N)))
        elif sample_size == "log2":
            return int(np.round(np.log2(N)))
        else:
            raise ValueError
    else:
        raise ValueError


class ScoreFunction(object):

    def __call__(self, X_train, y_train, X_test):
        raise NotImplementedError


class Random(ScoreFunction):

    """
    gives a random score to each row
    """

    def __init__(self, rng):
        """
        rng: np.random.RandomState object
        """
        self.rng = rng

    def __call__(self, X_train, y_train, X_test):
        return self.rng.randn(len(X_test))


class Bagging(ScoreFunction):

    """
    performs bootstrap sampling on an input score_fn, and aggregates scores
    """

    def __init__(self,
                 score_fn,
                 rng,
                 num_samples,
                 aggregator="mean",
                 sample_size=1.0,
                 replace=True):
        self.score_fn = score_fn
        self.rng = rng
        self.num_samples = num_samples
        self.aggregator = aggregator
        self.sample_size = sample_size
        self.replace = replace

    def __call__(self, X_train, y_train, X_test):
        all_scores = []
        N = len(X_train)
        all_idxs = np.arange(N)
        sample_size = flexible_int_input(N, self.sample_size)
        for _ in range(self.num_samples):
            idxs = self.rng.choice(all_idxs,
                                   sample_size,
                                   replace=self.replace)
            # getting scores
            scores = self.score_fn(X_train[idxs], y_train[idxs], X_test)
            all_scores.append(scores)
        all_scores = np.array(  # convert to numpy array
            all_scores
        ).swapaxes(  # put original rows as rows
            0, 1
        )
        return aggregate_columns(all_scores, self.aggregator)


class RepeatAndAggregate(ScoreFunction):

    """
    repeatedly applies input score_fn, and aggregates scores

    NOTE: only makes sense with stochastic score_fn
    """

    def __init__(self,
                 score_fn,
                 num_samples,
                 aggregator="mean"):
        self.score_fn = score_fn
        self.num_samples = num_samples
        self.aggregator = aggregator

    def __call__(self, X_train, y_train, X_test):
        all_scores = []
        for _ in range(self.num_samples):
            scores = self.score_fn(X_train, y_train, X_test)
            all_scores.append(scores)
        all_scores = np.array(  # convert to numpy array
            all_scores
        ).swapaxes(  # put original rows as rows
            0, 1
        )
        return aggregate_columns(all_scores, self.aggregator)


class Cached(ScoreFunction):

    """
    a stateful score function that uses the same scores computed for a given
    test set

    rationale: testing the effects of using stale scores
    """

    def __init__(self, score_fn):
        self.score_fn = score_fn
        self.last_X_test_ = None

    def __call__(self, X_train, y_train, X_test):
        if self.last_X_test_ is not X_test:
            self.last_X_test_ = X_test
            self.scores_ = self.score_fn(X_train, y_train, X_test)
        assert len(self.scores_) == len(X_test)
        return self.scores_


class PredictionClosestToValue(ScoreFunction):

    """
    scores observations based on how close their prediction is to a given value
    """

    def __init__(self,
                 target_value,
                 predict_fn,
                 aggregator="mean",
                 distance_fn="cityblock"):
        """
        target_value: value for best_idx's prediction to be close to

        predict_fn: function that takes in X_train, y_train, X_test and
        returns predictions for y_test
        """
        self.target_value = target_value
        self.predict_fn = predict_fn
        self.aggregator = aggregator
        self.distance_fn = distance_fn

    def __call__(self, X_train, y_train, X_test):
        preds = self.predict_fn(X_train, y_train, X_test)
        if self.distance_fn == "cityblock":
            distance_fn = lambda x, y: np.abs(x - y)
        elif self.distance_fn == "euclidean":
            distance_fn = lambda x, y: (x - y) ** 2
        else:
            distance_fn = self.distance_fn
        diff_from_target = -distance_fn(preds, self.target_value)
        return aggregate_columns(diff_from_target, self.aggregator)


class BootstrapPredictionMostVariance(ScoreFunction):

    """
    score observations by which have the most variance in predictions from
    bootstrapped models trained on the data

    rationale: observations with high variance should have low certainty
    """

    def __init__(self,
                 predict_fn,
                 rng,
                 num_samples,
                 aggregator="mean",
                 sample_size=1.0):
        self.predict_fn = predict_fn
        self.rng = rng
        self.num_samples = num_samples
        self.aggregator = aggregator
        self.sample_size = sample_size

    def __call__(self, X_train, y_train, X_test):
        std_fn = Bagging(
            self.predict_fn,
            self.rng,
            self.num_samples,
            sample_size=self.sample_size,
            aggregator="std",
        )
        # aggregate std for each column
        stds = std_fn(X_train, y_train, X_test)
        # take mean of std for each column
        return aggregate_columns(stds, self.aggregator)


class NoBootstrapPredictionMostVariance(ScoreFunction):

    """
    score observations by which have the most variance in predictions from
    models trained on the same data
    """

    def __init__(self,
                 predict_fn,
                 rng,
                 num_samples,
                 aggregator="mean"):
        self.predict_fn = predict_fn
        self.num_samples = num_samples
        self.aggregator = aggregator

    def __call__(self, X_train, y_train, X_test):
        std_fn = RepeatAndAggregate(
            self.predict_fn,
            self.num_samples,
            aggregator="std",
        )
        # aggregate std for each column
        stds = std_fn(X_train, y_train, X_test)
        # take mean of std for each column
        return aggregate_columns(stds, self.aggregator)


class Entropy(ScoreFunction):

    """
    score points based on their entropy

    assumes classification task and score function returns probabilities

    should be the same as prioritizing the probability that is closest to 0.5
    for binary classification
    """

    def __init__(self, predict_fn):
        self.predict_fn = predict_fn

    def __call__(self, X_train, y_train, X_test):
        probs = self.predict_fn(X_train, y_train, X_test)
        # need to transpose because entropy reduces along axis=0
        return scipy.stats.entropy(probs.T)


def PredictionClosestToValueComputedOnce(target_value,
                                         predict_fn,
                                         aggregator="mean"):
    """
    like PredictionClosestToValue, but use the same scores as the ones
    initially computed
    """
    return Cached(
        score_fn=PredictionClosestToValue(
            target_value=target_value,
            predict_fn=predict_fn,
            aggregator=aggregator,
        )
    )
