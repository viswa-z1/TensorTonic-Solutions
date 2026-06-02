import numpy as np

def relu(x):
    """
    Implement ReLU activation function.

    Parameters:
        x : scalar, list, or NumPy array

    Returns:
        NumPy array with same shape as input
    """
    x = np.asarray(x, dtype=float)
    return np.maximum(0.0, x)