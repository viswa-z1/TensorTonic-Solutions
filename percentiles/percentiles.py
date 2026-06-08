import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.

    Parameters
    ----------
    x : list or array-like
        Numeric data
    q : list, array-like, or scalar
        Percentiles in [0, 100]

    Returns
    -------
    np.ndarray
        Percentile values
    """
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)

    try:
        return np.percentile(x, q, method="linear")
    except TypeError:
        # For older NumPy versions
        return np.percentile(x, q, interpolation="linear")