import numpy as np

def contrastive_loss(a, b, y,
                     margin=1.0,
                     reduction="mean") -> float:
    """
    Compute contrastive loss for Siamese networks.

    Parameters:
        a, b      : shape (D,) or (N,D)
        y         : shape (N,), values in {0,1}
        margin    : margin for negative pairs
        reduction : "mean" or "sum"

    Returns:
        float
    """

    # Convert to arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)

    # Convert single vectors to batch
    if a.ndim == 1:
        a = a.reshape(1, -1)

    if b.ndim == 1:
        b = b.reshape(1, -1)

    # Validate shapes
    if a.shape != b.shape:
        raise ValueError("a and b must have same shape")

    if y.shape[0] != a.shape[0]:
        raise ValueError("y must match batch size")

    # Validate labels
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y must contain only 0 or 1")

    # Euclidean distances
    d = np.sqrt(np.sum((a - b) ** 2, axis=1))

    # Contrastive loss
    losses = (
        y * (d ** 2)
        + (1 - y) * np.maximum(0.0, margin - d) ** 2
    )

    # Reduction
    if reduction == "mean":
        return float(np.mean(losses))

    elif reduction == "sum":
        return float(np.sum(losses))

    else:
        raise ValueError("reduction must be 'mean' or 'sum'")