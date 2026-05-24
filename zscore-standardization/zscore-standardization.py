import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X using z-score normalization.

    Parameters:
        X    : 1D or 2D NumPy array
        axis : axis for computing mean/std
        eps  : small value to avoid divide-by-zero

    Returns:
        Standardized NumPy array (float)
    """

    # Convert to float array
    X = np.asarray(X, dtype=float)

    # Compute mean and std
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)

    # Standardize
    X_standardized = (X - mean) / (std + eps)

    return X_standardized