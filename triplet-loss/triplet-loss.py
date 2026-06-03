import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss using squared Euclidean distance.

    Parameters:
        anchor   : (N,D) or (D,)
        positive : (N,D) or (D,)
        negative : (N,D) or (D,)
        margin   : float >= 0

    Returns:
        float (mean loss over batch)
    """

    anchor = np.asarray(anchor, dtype=float)
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)

    # Handle single vectors
    if anchor.ndim == 1:
        anchor = anchor.reshape(1, -1)

    if positive.ndim == 1:
        positive = positive.reshape(1, -1)

    if negative.ndim == 1:
        negative = negative.reshape(1, -1)

    # Validate shapes
    if anchor.shape != positive.shape or anchor.shape != negative.shape:
        raise ValueError("anchor, positive, and negative must have the same shape")

    # Squared Euclidean distances
    d_ap = np.sum((anchor - positive) ** 2, axis=1)
    d_an = np.sum((anchor - negative) ** 2, axis=1)

    # Triplet loss per sample
    losses = np.maximum(0.0, d_ap - d_an + margin)

    # Mean batch loss
    return float(np.mean(losses))