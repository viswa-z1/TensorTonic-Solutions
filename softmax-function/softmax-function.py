import numpy as np

def softmax(x):
    """
    Compute softmax for 1D or 2D NumPy arrays.

    For 1D:
        returns shape (N,)

    For 2D:
        computes row-wise softmax
        returns shape (M, N)
    """

    x = np.asarray(x, dtype=float)

    # 1D case
    if x.ndim == 1:

        x_shifted = x - np.max(x)

        exp_x = np.exp(x_shifted)

        return exp_x / np.sum(exp_x)

    # 2D case
    elif x.ndim == 2:

        x_shifted = x - np.max(x, axis=1, keepdims=True)

        exp_x = np.exp(x_shifted)

        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    else:
        raise ValueError("Input must be 1D or 2D")