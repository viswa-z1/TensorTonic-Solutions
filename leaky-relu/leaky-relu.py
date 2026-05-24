import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.

    Parameters:
        x      : scalar, list, or NumPy array
        alpha  : slope for negative values

    Returns:
        NumPy array
    """

    # Convert input to NumPy array
    x = np.asarray(x, dtype=float)

    # Apply Leaky ReLU
    return np.where(x >= 0, x, alpha * x)