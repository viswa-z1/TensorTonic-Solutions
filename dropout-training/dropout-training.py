import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.

    Parameters:
        x   : input array
        p   : dropout probability
        rng : optional random generator

    Returns:
        (output, dropout_pattern)
    """

    x = np.asarray(x, dtype=float)

    # Special case: no dropout
    if p == 0.0:
        pattern = np.ones_like(x, dtype=float)
        return x.copy(), pattern

    # Keep probability
    keep_prob = 1.0 - p

    # Random generator
    if rng is not None:
        random_vals = rng.random(x.shape)
    else:
        random_vals = np.random.random(x.shape)

    # Create dropout mask/pattern
    pattern = np.where(random_vals < keep_prob,
                       1.0 / keep_prob,
                       0.0)

    # Apply dropout
    output = x * pattern

    return output, pattern