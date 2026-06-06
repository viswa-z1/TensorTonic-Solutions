import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.

    Parameters
    ----------
    k : list or array-like of integers (k >= 1)
    p : float, success probability (0 < p <= 1)

    Returns
    -------
    (pmf, mean)
        pmf : np.ndarray
        mean : float
    """
    k = np.asarray(k, dtype=float)

    pmf = ((1.0 - p) ** (k - 1)) * p
    mean = float(1.0 / p)

    return pmf, mean