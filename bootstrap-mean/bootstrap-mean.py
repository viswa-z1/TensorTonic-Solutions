import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)

    Parameters
    ----------
    x : array-like, shape (N,)
        Input observations.
    n_bootstrap : int
        Number of bootstrap samples.
    ci : float
        Confidence level (e.g. 0.95).
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    boot_means : ndarray, shape (n_bootstrap,)
        Mean of each bootstrap sample.
    lower : float
        Lower confidence bound.
    upper : float
        Upper confidence bound.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    if rng is None:
        indices = np.random.randint(0, n, size=(n_bootstrap, n))
    else:
        indices = rng.integers(0, n, size=(n_bootstrap, n))

    # Bootstrap samples and their means
    boot_samples = x[indices]
    boot_means = boot_samples.mean(axis=1)

    # Confidence interval
    alpha = (1.0 - ci) / 2.0
    lower, upper = np.quantile(boot_means, [alpha, 1.0 - alpha])

    return boot_means, float(lower), float(upper)