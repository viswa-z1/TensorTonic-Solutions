import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays.

    Parameters:
        x, y : lists or NumPy arrays

    Returns:
        float
    """

    # Convert to NumPy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Validate dimensions
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")

    # Validate equal lengths
    if x.shape[0] != y.shape[0]:
        raise ValueError("Vectors must have the same length")

    # Compute dot product
    return float(np.dot(x, y))