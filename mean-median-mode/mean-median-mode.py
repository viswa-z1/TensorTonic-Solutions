import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    Returns: (mean, median, mode)
    """

    x = np.asarray(x, dtype=float)

    mean = float(np.mean(x))
    median = float(np.median(x))

    counts = Counter(x.tolist())
    max_freq = max(counts.values())

    # Smallest value among those with maximum frequency
    mode = float(min(val for val, freq in counts.items() if freq == max_freq))

    return mean, median, mode