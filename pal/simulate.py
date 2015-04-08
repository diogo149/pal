import numpy as np
from sklearn import cross_validation


def train_test_split_indexes(y,
                             test_size=0.1,
                             train_size=None,
                             stratified=False,
                             random_state=None):
    # TODO move to utils?
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


def simulate_sequential(X,
                        y,
                        predict_fn,
                        score_fn,
                        objective_fn,
                        num_initial_samples,
                        num_final_samples,
                        test_size=0.33,
                        seed=42,
                        stratified_sample=False):
    """
    predict_fn:
    takes in training features, training labels, and test features and
    returns predictions for each of the test features

    score_fn:
    takes in training features, training labels, and test features and
    returns a score of how good the test observation would be to label
    (higher is better)

    objective_fn:
    function to maximize
    takes in y_true as 1st argument, predictions as 2nd argument
    """
    assert len(X.shape) == len(y.shape) == 2
    assert len(X) == len(y)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed
    )

    def calculate_test_score(labeled_idxs):
        X_labeled = X_train[labeled_idxs]
        y_labeled = y_train[labeled_idxs]
        preds = predict_fn(X_labeled, y_labeled, X_test)
        return objective_fn(y_test, preds)

    labeled_idxs = list(
        train_test_split_indexes(
            y_train,
            test_size=num_initial_samples,
            random_state=seed,
            stratified=stratified_sample
        )[1])
    assert len(labeled_idxs) == num_initial_samples
    init_score = calculate_test_score(labeled_idxs)
    test_scores = [init_score]
    learning_scores_labeled = []
    learning_scores_unlabeled = []
    while len(labeled_idxs) < num_final_samples:
        learning_scores = score_fn(X_train[labeled_idxs],
                                   y_train[labeled_idxs],
                                   X_train)

        ls_labeled = learning_scores[labeled_idxs]
        is_unlabeled = ~np.in1d(np.arange(len(X_train)), labeled_idxs)
        ls_unlabeled = learning_scores[is_unlabeled]
        learning_scores_labeled.append(ls_labeled)
        learning_scores_unlabeled.append(ls_unlabeled)

        # use learning_scores to determine next index to label
        next_idx = np.where(is_unlabeled
                            & (learning_scores == np.max(ls_unlabeled)))[0][0]

        # FIXME delete
        # next_idx = active_learning_fn(X_train,
        #                               labeled_idxs,
        #                               y_train[labeled_idxs])
        assert next_idx not in labeled_idxs
        labeled_idxs.append(next_idx)
        test_score = calculate_test_score(labeled_idxs)
        test_scores.append(test_score)
    return dict(
        test_scores=test_scores,
        learning_scores_labeled=learning_scores_labeled,
        learning_scores_unlabeled=learning_scores_unlabeled,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
