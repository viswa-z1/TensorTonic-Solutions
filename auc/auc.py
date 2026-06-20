import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.

    Parameters
    ----------
    fpr : array-like, shape (M,)
        False Positive Rate values (must be increasing)
    tpr : array-like, shape (M,)
        True Positive Rate values

    Returns
    -------
    float
        Area under the ROC curve in [0, 1]
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)

    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape")

    if fpr.size < 2:
        raise ValueError("At least two points are required")

    return float(np.trapezoid(tpr, fpr))