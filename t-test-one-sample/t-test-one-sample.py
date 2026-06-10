import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    x = np.asarray(x, dtype=float)

    n = x.size
    mean_x = np.mean(x)

    # Sample standard deviation (Bessel correction)
    s = np.sqrt(np.sum((x - mean_x) ** 2) / (n - 1))

    # Standard error
    se = s / np.sqrt(n)

    # Handle zero-variance case
    if se == 0:
        return float(0.0 if mean_x == mu0 else np.sign(mean_x - mu0) * np.inf)

    t_stat = (mean_x - mu0) / se

    return float(t_stat)