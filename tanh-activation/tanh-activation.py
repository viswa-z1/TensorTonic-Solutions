import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.

    Parameters:
        x : scalar, list, or NumPy array

    Returns:
        np.ndarray of floats
    """
    x = np.asarray(x, dtype=float)

    # Special handling for scalar input:
    # expected output is shape (1,)
    if x.ndim == 0:
        x = x.reshape(1)

    return np.tanh(x)