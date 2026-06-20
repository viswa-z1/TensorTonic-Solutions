import numpy as np

def roc_curve(y_true, y_score):
    """
    Compute ROC curve from binary labels and scores.

    Parameters
    ----------
    y_true : array-like of shape (N,)
        Binary labels {0,1}
    y_score : array-like of shape (N,)
        Prediction scores

    Returns
    -------
    fpr : np.ndarray
    tpr : np.ndarray
    thresholds : np.ndarray
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Sort by descending score
    desc_idx = np.argsort(-y_score, kind='mergesort')
    y_score = y_score[desc_idx]
    y_true = y_true[desc_idx]

    # Cumulative true positives and false positives
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)

    # Indices where score changes (last occurrence of each unique score)
    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_indices, len(y_score) - 1]

    # Extract TP, FP, and thresholds at those indices
    tps = tps[threshold_idxs]
    fps = fps[threshold_idxs]
    thresholds = y_score[threshold_idxs]

    # Total positives and negatives
    P = tps[-1]
    N = fps[-1]

    # Compute rates
    tpr = tps / P if P > 0 else np.zeros_like(tps, dtype=float)
    fpr = fps / N if N > 0 else np.zeros_like(fps, dtype=float)

    # Add starting point (0,0) with threshold = inf
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    thresholds = np.r_[np.inf, thresholds]

    return fpr, tpr, thresholds