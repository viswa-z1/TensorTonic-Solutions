import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    X = np.asarray(X, dtype=float)
    X_imp = X.copy()

    if strategy not in ('mean', 'median'):
        raise ValueError("strategy must be 'mean' or 'median'")

    # 1D case
    if X_imp.ndim == 1:
        nan_mask = np.isnan(X_imp)

        if np.any(nan_mask):
            valid = X_imp[~nan_mask]

            if valid.size == 0:
                fill_value = 0.0
            else:
                fill_value = (
                    np.mean(valid)
                    if strategy == 'mean'
                    else np.median(valid)
                )

            X_imp[nan_mask] = fill_value

        return X_imp

    # 2D case
    if X_imp.ndim == 2:
        n_cols = X_imp.shape[1]

        for j in range(n_cols):
            col = X_imp[:, j]
            nan_mask = np.isnan(col)

            if not np.any(nan_mask):
                continue

            valid = col[~nan_mask]

            if valid.size == 0:
                fill_value = 0.0
            else:
                fill_value = (
                    np.mean(valid)
                    if strategy == 'mean'
                    else np.median(valid)
                )

            col[nan_mask] = fill_value
            X_imp[:, j] = col

        return X_imp

    # Handle higher dimensions if needed
    raise ValueError("X must be 1D or 2D")