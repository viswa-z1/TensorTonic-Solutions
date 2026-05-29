import numpy as np

def covariance_matrix(X):
    """
    Compute sample covariance matrix without using np.cov.

    Parameters:
        X : array-like of shape (N, D)

    Returns:
        np.ndarray of shape (D, D)
        or None for invalid input
    """

    try:
        X = np.asarray(X, dtype=float)

        # Must be 2D
        if X.ndim != 2:
            return None

        N, D = X.shape

        # Need at least 2 samples
        if N < 2:
            return None

        # Compute feature means
        mean = np.mean(X, axis=0)

        # Center the data
        X_centered = X - mean

        # Sample covariance
        cov = (X_centered.T @ X_centered) / (N - 1)

        return cov

    except Exception:
        return None