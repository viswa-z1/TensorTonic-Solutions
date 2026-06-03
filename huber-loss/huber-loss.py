import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute mean Huber Loss.

    Parameters:
        y_true : array-like
        y_pred : array-like
        delta  : positive float

    Returns:
        float
    """

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Prediction error
    e = y_true - y_pred
    abs_e = np.abs(e)

    # Piecewise Huber loss
    loss = np.where(
        abs_e <= delta,
        0.5 * e**2,
        delta * (abs_e - 0.5 * delta)
    )

    return float(np.mean(loss))