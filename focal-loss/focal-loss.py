import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.

    Parameters:
        p : np.ndarray of predicted probabilities, shape (N,)
        y : np.ndarray of binary labels {0,1}, shape (N,)
        gamma : focusing parameter

    Returns:
        float : mean focal loss
    """

    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    # Numerical stability
    p = np.clip(p, 1e-15, 1.0 - 1e-15)

    # Positive-class term
    term1 = ((1.0 - p) ** gamma) * y * np.log(p)

    # Negative-class term
    term2 = (p ** gamma) * (1.0 - y) * np.log(1.0 - p)

    # Element-wise focal loss
    loss = -(term1 + term2)

    return float(np.mean(loss))