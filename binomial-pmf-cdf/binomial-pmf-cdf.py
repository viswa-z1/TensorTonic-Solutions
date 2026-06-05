import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.

    Parameters:
        n : int
            Number of trials
        p : float
            Success probability
        k : int
            Number of successes

    Returns:
        (pmf, cdf)
    """

    # PMF
    pmf = comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))

    # CDF
    cdf = 0.0
    for i in range(k + 1):
        cdf += comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))

    return float(pmf), float(cdf)