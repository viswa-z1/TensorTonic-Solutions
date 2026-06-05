import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.

    Parameters:
        p : array-like
            Predicted probabilities (1D or 2D)
        y : array-like
            Binary ground-truth mask, same shape as p
        eps : float
            Numerical stability constant

    Returns:
        float
    """

    p = np.asarray(p, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()

    if p.shape != y.shape:
        raise ValueError("p and y must have the same shape")

    # Intersection
    intersection = np.sum(p * y)

    # Sums
    sum_p = np.sum(p)
    sum_y = np.sum(y)

    # Dice coefficient
    dice = (2.0 * intersection + eps) / (sum_p + sum_y + eps)

    # Dice loss
    return float(1.0 - dice)