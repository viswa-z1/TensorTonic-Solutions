import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    Compute binary hinge loss.

    Parameters:
        y_true     : 1D array of {-1, +1}
        y_score    : 1D array of prediction scores
        margin     : margin value
        reduction  : "mean" or "sum"

    Returns:
        float
    """

    # Convert to arrays
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    # Validate shapes
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have same shape")

    # Validate labels
    valid = np.all(np.isin(y_true, [-1, 1]))

    if not valid:
        raise ValueError("y_true must contain only -1 and +1")

    # Vectorized hinge loss
    losses = np.maximum(0.0, margin - y_true * y_score)

    # Reduction
    if reduction == "mean":
        return float(np.mean(losses))

    elif reduction == "sum":
        return float(np.sum(losses))

    else:
        raise ValueError("reduction must be 'mean' or 'sum'")