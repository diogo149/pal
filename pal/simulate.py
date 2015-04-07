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


def sequential_binary(X,
                      y,
                      predict_fn,
                      active_learning_fn,
                      objective_fn,
                      num_initial_samples,
                      num_final_samples,
                      test_size=0.33,
                      seed=42):
    """

    objective_fn:
    function to maximize
    takes in y_true as 1st argument, predictions as 2nd argument
    """
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed
    )

    def calculate_test_score(train_idxs):
        X_tmp = X_train[train_idxs]
        y_tmp = y_train[train_idxs]
        preds = predict_fn(X_tmp, y_tmp, X_test)
        return objective_fn(y_test, preds)

    train_idxs = list(train_test_split_indexes(y_train,
                                               test_size=num_initial_samples,
                                               random_state=seed,
                                               stratified=True)[1])
    assert len(train_idxs) == num_initial_samples
    init_score = calculate_test_score(train_idxs)
    scores = [init_score]
    while len(train_idxs) < num_final_samples:
        next_idx = active_learning_fn(X_train, train_idxs, y_train[train_idxs])
        assert next_idx not in train_idxs
        train_idxs.append(next_idx)
        score = calculate_test_score(train_idxs)
        scores.append(score)
    return scores
