import numpy as np
import math

def gelu(x):
    """
    Compute the exact GELU activation using erf.

    Parameters:
        x : scalar, list, or np.ndarray

    Returns:
        np.ndarray of floats
    """

    # Convert input to NumPy array
    x = np.asarray(x, dtype=float)

    # Vectorized erf
    erf_vec = np.vectorize(math.erf)

    # GELU formula
    return 0.5 * x * (
        1.0 + erf_vec(x / np.sqrt(2.0))
    )