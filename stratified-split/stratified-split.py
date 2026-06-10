import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    X = np.asarray(X)
    y = np.asarray(y)

    if rng is None:
        rng = np.random.RandomState(42)

    train_idx = []
    test_idx = []

    classes, counts = np.unique(y, return_counts=True)

    for cls, n_cls in zip(classes, counts):
        idx = np.where(y == cls)[0].copy()

        if hasattr(rng, "shuffle"):
            rng.shuffle(idx)
        else:
            np.random.shuffle(idx)

        n_test = int(round(n_cls * test_size))

        if n_cls > 1:
            n_test = min(n_test, n_cls - 1)
        else:
            n_test = 0

        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    # Preserve original dataset ordering
    train_idx = np.sort(np.array(train_idx, dtype=int))
    test_idx = np.sort(np.array(test_idx, dtype=int))

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test