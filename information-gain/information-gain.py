import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    y = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)

    if y.shape[0] != split_mask.shape[0]:
        raise ValueError("y and split_mask must have the same length")

    y_left = y[split_mask]
    y_right = y[~split_mask]

    n_left = y_left.size
    n_right = y_right.size
    n_total = y.size

    # Empty side => no information gained
    if n_left == 0 or n_right == 0:
        return 0.0

    parent_entropy = _entropy(y)
    left_entropy = _entropy(y_left)
    right_entropy = _entropy(y_right)

    weighted_child_entropy = (
        (n_left / n_total) * left_entropy +
        (n_right / n_total) * right_entropy
    )

    return float(parent_entropy - weighted_child_entropy)