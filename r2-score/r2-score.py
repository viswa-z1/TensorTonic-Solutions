import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² score for 1D regression.

    Handles constant-target edge case:
      - return 1.0 if predictions match exactly
      - else return 0.0
    """

    # Convert to NumPy arrays
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Mean of targets
    y_mean = np.mean(y_true)

    # Total sum of squares
    sst = np.sum((y_true - y_mean) ** 2)

    # Constant target edge case
    if sst == 0:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0

    # Residual sum of squares
    sse = np.sum((y_true - y_pred) ** 2)

    # R² score
    r2 = 1.0 - (sse / sst)

    return float(r2)