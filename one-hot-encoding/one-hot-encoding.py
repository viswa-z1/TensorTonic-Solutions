import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("y must be a 1D array")

    if y.size == 0:
        raise ValueError("y must not be empty")

    if np.any(y < 0):
        raise ValueError("labels must be non-negative")

    if num_classes is None:
        num_classes = int(np.max(y)) + 1

    if num_classes < 1:
        raise ValueError("num_classes must be >= 1")

    if np.any(y >= num_classes):
        raise ValueError("label out of range for num_classes")

    Y = np.zeros((y.shape[0], num_classes), dtype=float)
    Y[np.arange(y.shape[0]), y.astype(int)] = 1.0

    return Y