import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.

    Parameters
    ----------
    lam : float
        Rate parameter (lambda > 0)
    k : int
        Number of events (k >= 0)

    Returns
    -------
    (pmf, cdf) : tuple of floats
    """

    # log(k!)
    if k <= 1:
        log_fact_k = 0.0
    else:
        log_fact_k = np.sum(np.log(np.arange(1, k + 1)))

    # PMF
    log_pmf = -lam + k * np.log(lam) - log_fact_k
    pmf = np.exp(log_pmf)

    # CDF
    cdf = 0.0
    for i in range(k + 1):

        if i <= 1:
            log_fact_i = 0.0
        else:
            log_fact_i = np.sum(np.log(np.arange(1, i + 1)))

        log_term = -lam + i * np.log(lam) - log_fact_i
        cdf += np.exp(log_term)

    return float(pmf), float(cdf)
    