import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.

    Parameters
    ----------
    y_left : array-like
        Labels in the left child node.
    y_right : array-like
        Labels in the right child node.

    Returns
    -------
    float
        Weighted Gini impurity of the split.
    """
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)

    def node_gini(y):
        n = len(y)
        if n == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / n
        return 1.0 - np.sum(probs ** 2)

    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    if n_total == 0:
        return 0.0

    gini_left = node_gini(y_left)
    gini_right = node_gini(y_right)

    weighted_gini = (
        (n_left / n_total) * gini_left +
        (n_right / n_total) * gini_right
    )

    return float(weighted_gini)