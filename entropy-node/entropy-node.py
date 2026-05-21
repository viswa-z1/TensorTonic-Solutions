import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.

    Parameters:
        y : array-like of class labels

    Returns:
        float entropy
    """

    # Convert to numpy array
    y = np.asarray(y)

    # Empty node -> zero entropy
    if y.size == 0:
        return 0.0

    # Get class counts
    _, counts = np.unique(y, return_counts=True)

    # Convert counts to probabilities
    probs = counts / counts.sum()

    # Stable computation: ignore zero probabilities
    probs = probs[probs > 0]

    # Compute entropy
    entropy = -np.sum(probs * np.log2(probs))

    return float(entropy)