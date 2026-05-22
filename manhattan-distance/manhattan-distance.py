import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """

    # Convert to NumPy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Compute Manhattan distance
    distance = np.sum(np.abs(x - y))

    return float(distance)