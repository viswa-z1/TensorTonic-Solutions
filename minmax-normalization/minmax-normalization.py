import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1].

    Parameters:
        X    : 1D or 2D array
        axis : axis along which scaling is applied
        eps  : small value to avoid divide-by-zero

    Returns:
        Scaled NumPy array (float)
    """

    # Convert to float array
    X = np.asarray(X, dtype=float)

    # Compute min and max
    x_min = np.min(X, axis=axis, keepdims=True)
    x_max = np.max(X, axis=axis, keepdims=True)

    # Denominator with numerical stability
    denom = np.maximum(x_max - x_min, eps)

    # Min-max scaling
    X_scaled = (X - x_min) / denom

    return X_scaled
    