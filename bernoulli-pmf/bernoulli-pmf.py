import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.

    Parameters:
        x : array-like containing 0s and 1s
        p : probability of success

    Returns:
        (pmf, mean, var)
    """

    x = np.asarray(x)

    # PMF: P(X=1)=p, P(X=0)=1-p
    pmf = np.where(x == 1, p, 1.0 - p).astype(float)

    # Moments
    mean = float(p)
    var = float(p * (1.0 - p))

    return pmf, mean, var