import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)

    Parameters
    ----------
    N : int
        Number of samples.
    k : int
        Number of folds.
    shuffle : bool
        Whether to shuffle indices before splitting.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    folds : list of tuples
        Each tuple is (train_idx, val_idx), both 1D int arrays.
    """
    if not (2 <= k <= N):
        raise ValueError("Require 2 <= k <= N")

    indices = np.arange(N)

    if shuffle:
        if rng is not None:
            indices = rng.permutation(indices)
        else:
            indices = np.random.permutation(indices)

    # Split into k folds with sizes differing by at most 1
    val_folds = np.array_split(indices, k)

    folds = []

    for i in range(k):
        val_idx = val_folds[i]

        # Concatenate all other folds to form training indices
        if k == 2:
            train_idx = val_folds[1 - i]
        else:
            train_idx = np.concatenate(val_folds[:i] + val_folds[i+1:])

        folds.append((train_idx.astype(int), val_idx.astype(int)))

    return folds