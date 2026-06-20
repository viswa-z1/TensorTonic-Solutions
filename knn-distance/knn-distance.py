import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise Euclidean distances and return k nearest neighbor indices.

    Parameters
    ----------
    X_train : array-like, shape (n_train, d) or (n_train,)
    X_test : array-like, shape (n_test, d) or (n_test,)
    k : int

    Returns
    -------
    neighbors : ndarray, shape (n_test, k)
        Indices of k nearest training points, padded with -1 if k > n_train.
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    # Handle 1D input
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Compute pairwise Euclidean distances via broadcasting
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))

    # Number of actual neighbors available
    m = min(k, n_train)

    # Indices sorted by distance
    nearest = np.argsort(distances, axis=1)[:, :m]

    # Pad with -1 if k > n_train
    if k > n_train:
        result = np.full((n_test, k), -1, dtype=int)
        result[:, :n_train] = nearest
    else:
        result = nearest.astype(int)

    return result