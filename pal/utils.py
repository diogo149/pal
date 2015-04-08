import copy

import numpy as np
from sklearn import cross_validation


def train_test_split_indexes(y,
                             test_size=0.1,
                             train_size=None,
                             stratified=False,
                             random_state=None):
    kwargs = dict(
        train_size=train_size,
        test_size=test_size,
        random_state=random_state
    )
    if stratified:
        splitter = cross_validation.StratifiedShuffleSplit(y, **kwargs)
    else:
        splitter = cross_validation.ShuffleSplit(len(y), **kwargs)
    return next(iter(splitter))


def predict_model(clf, X_train, y_train, X_test):
    """
    takes in a sklearn model and 3 matrices (2D arrays) corresponding to
    training features, training targets, and testing features and returns
    a matrix of predictions
    """
    # deepcopy model so that random state isn't changed over time
    clf = copy.deepcopy(clf)
    if hasattr(clf, "predict_proba"):
        # classification
        num_points = X_test.shape[0]
        num_classes = y_train.shape[1]
        clf.fit(X_train, np.argmax(y_train, axis=1))
        preds = clf.predict_proba(X_test)
        all_preds = np.zeros((num_points, num_classes), dtype=preds.dtype)
        all_preds[:, clf.classes_] = preds
        return all_preds
    else:
        return clf.predict(X_test)
