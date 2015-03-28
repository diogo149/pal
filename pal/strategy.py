import numpy as np


class ActiveLearningStrategy(object):

    def __call__(self, X_train, train_idxs, y):
        raise NotImplementedError


class Random(ActiveLearningStrategy):

    """
    predicts a random next index that hasn't been chosen before
    """

    def __init__(self, rng):
        """
        rng: np.random.RandomState object
        """
        self.rng = rng

    def __call__(self, X_train, train_idxs, y):
        idxs = list(set(range(len(X_train))) - set(train_idxs))
        return self.rng.choice(idxs)


class PredictionClosestToValue(ActiveLearningStrategy):

    """
    returns the index of the observation whose predicted value is closest to
    the input target value
    """

    def __init__(self, target_value, predict_fn):
        """
        target_value: value for best_idx's prediction to be close to

        predict_fn: function that takes in X_train, y_train, X_test and
        returns predictions for y_test
        """
        self.target_value = target_value
        self.predict_fn = predict_fn

    def __call__(self, X_train, train_idxs, y):
        X_tmp = X_train[train_idxs]
        preds = self.predict_fn(X_tmp, y, X_train)
        idxs = set(train_idxs)
        best_idx = None
        lowest_diff = np.inf
        for idx, pred in enumerate(preds):
            if idx not in idxs:
                diff = abs(pred - self.target_value)
                if lowest_diff > diff:
                    best_idx = idx
                    lowest_diff = diff
        assert best_idx is not None
        return best_idx


class PredictionClosestToValueComputedOnce(ActiveLearningStrategy):

    """
    like PredictionClosestToValue, but the order of indexes chosen is
    selected once, before any feedback is received

    this is to test the necessity of the feedback by comparing to random or
    one with feedback
    """

    def __init__(self, target_value, predict_fn):
        """
        target_value: value for best_idx's prediction to be close to

        predict_fn: function that takes in X_train, y_train, X_test and
        returns predictions for y_test
        """
        self.target_value = target_value
        self.predict_fn = predict_fn
        self.computed = False

    def __call__(self, X_train, train_idxs, y):
        if not self.computed:
            self.computed = True
            X_tmp = X_train[train_idxs]
            preds = self.predict_fn(X_tmp, y, X_train)
            idxs = set(train_idxs)
            self.candidates = []
            for idx, pred in enumerate(preds):
                if idx not in idxs:
                    diff = abs(pred - self.target_value)
                    self.candidates.append((idx, diff))
            self.candidates.sort(key=lambda x: x[1], reverse=True)
        idx, diff = self.candidates.pop()
        return idx


class BootstrapPredictionMostVariance(ActiveLearningStrategy):

    """
    pick the index which has the most variance in predictions from
    bootstrapped models trained on the data

    WARNING: can be a bit slow
    """

    def __init__(self, predict_fn, num_predictions=10):
        """
        predict_fn: function that takes in X_train, y_train, X_test and
        returns predictions for y_test

        num_predictions: number of predictions to take the variance over
        """
        self.predict_fn = predict_fn
        self.num_predictions = num_predictions

    def __call__(self, X_train, train_idxs, y):
        X_tmp = X_train[train_idxs]
        all_preds = []
        for _ in range(self.num_predictions):
            # bootstrap sampling
            idxs = np.random.randint(0, len(train_idxs), len(train_idxs))
            # getting prediction
            preds = self.predict_fn(X_tmp[idxs], y[idxs], X_train)
            all_preds.append(preds)
        stds = np.std(all_preds, axis=0)
        idxs = set(train_idxs)
        best_idx = None
        highest_variance = -np.inf
        for idx, std in enumerate(stds):
            if idx not in idxs:
                if highest_variance < std:
                    best_idx = idx
                    highest_variance = std
        assert best_idx is not None
        return best_idx


class NoBootstrapPredictionMostVariance(ActiveLearningStrategy):

    """
    pick the index which has the most variance in predictions from models
    trained on the data

    WARNING: can be a bit slow
    WARNING: this uses the same data every time, so a stochastic model should
    be used
    """

    def __init__(self, predict_fn, num_predictions=10):
        """
        predict_fn: function that takes in X_train, y_train, X_test and
        returns predictions for y_test

        num_predictions: number of predictions to take the variance over
        """
        self.predict_fn = predict_fn
        self.num_predictions = num_predictions

    def __call__(self, X_train, train_idxs, y):
        X_tmp = X_train[train_idxs]
        all_preds = []
        for _ in range(self.num_predictions):
            preds = self.predict_fn(X_tmp, y, X_train)
            all_preds.append(preds)
        stds = np.std(all_preds, axis=0)
        idxs = set(train_idxs)
        best_idx = None
        highest_variance = -np.inf
        for idx, std in enumerate(stds):
            if idx not in idxs:
                if highest_variance < std:
                    best_idx = idx
                    highest_variance = std
        assert best_idx is not None
        return best_idx
