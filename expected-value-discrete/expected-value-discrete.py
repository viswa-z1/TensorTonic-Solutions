import numpy as np

def expected_value_discrete(x, p):
    """
    Compute expected value of a discrete random variable.

    Parameters:
        x : values
        p : probabilities

    Returns:
        float expected value
    """

    # Convert to numpy arrays
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    # Validate shapes
    if x.shape != p.shape:
        raise ValueError("x and p must have the same shape")

    # Validate probabilities sum to 1
    if not np.allclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1")

    # Compute expected value
    expected = np.sum(x * p)

    return float(expected)