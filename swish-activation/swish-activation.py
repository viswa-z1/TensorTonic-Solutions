import numpy as np

def swish(x):
    """
    Implement Swish activation function.

    Parameters:
        x : scalar, list, or np.ndarray

    Returns:
        np.ndarray of floats
    """

    x = np.asarray(x, dtype=float)

    # For scalar input, return shape (1,)
    if x.ndim == 0:
        x = x.reshape(1)

    # Clip to avoid overflow in exp()
    x_clip = np.clip(x, -500, 500)

    # Sigmoid
    sigmoid = 1.0 / (1.0 + np.exp(-x_clip))

    # Swish
    return x * sigmoid