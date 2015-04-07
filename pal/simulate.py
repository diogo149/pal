from sklearn import cross_validation


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

    def calc_score(train_idxs):
        X_tmp = X_train[train_idxs]
        y_tmp = y_train[train_idxs]
        preds = predict_fn(X_tmp, y_tmp, X_test)
        return objective_fn(y_test, preds)

    sss = cross_validation.StratifiedShuffleSplit(
        y_train,
        test_size=num_initial_samples,
        random_state=seed)
    train_idxs = list(sss.__iter__().next()[1])
    init_score = calc_score(train_idxs)
    scores = [init_score]
    while len(train_idxs) < num_final_samples:
        next_idx = active_learning_fn(X_train, train_idxs, y_train[train_idxs])
        assert next_idx not in train_idxs
        train_idxs.append(next_idx)
        score = calc_score(train_idxs)
        scores.append(score)
    return scores
