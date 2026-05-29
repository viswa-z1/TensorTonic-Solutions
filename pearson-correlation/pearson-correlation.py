import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.

    Returns:
        np.ndarray of shape (D, D)
        or None for invalid input.
    """

    try:
        # Convert input
        X = np.asarray(X, dtype=float)

        # Must be 2D
        if X.ndim != 2:
            return None

        N, D = X.shape

        # Need at least 2 samples
        if N < 2:
            return None

        # Center data
        mean = np.mean(X, axis=0)
        X_centered = X - mean

        # Sample covariance matrix
        cov = (X_centered.T @ X_centered) / (N - 1)

        # Standard deviations from covariance diagonal
        std = np.sqrt(np.diag(cov))

        # Denominator matrix σσᵀ
        denom = np.outer(std, std)

        # Initialize correlation matrix with NaNs
        corr = np.full((D, D), np.nan, dtype=float)

        # Compute correlations only where denominator is non-zero
        valid = denom != 0
        corr[valid] = cov[valid] / denom[valid]

        # Diagonal:
        # 1.0 for non-constant features
        # NaN for zero-variance features
        diag_idx = np.arange(D)
        corr[diag_idx, diag_idx] = np.where(std > 0, 1.0, np.nan)

        return corr

    except Exception:
        return None