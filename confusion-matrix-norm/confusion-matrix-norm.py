import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.

    Parameters
    ----------
    y_true : array-like, shape (N,)
        True labels.
    y_pred : array-like, shape (N,)
        Predicted labels.
    num_classes : int or None
        Number of classes. If None, inferred from data.
    normalize : {'none', 'true', 'pred', 'all'}
        Normalization mode.

    Returns
    -------
    cm : ndarray of shape (K, K)
        Confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if y_true.size == 0:
        K = num_classes if num_classes is not None else 0
        dtype = int if normalize == 'none' else float
        return np.zeros((K, K), dtype=dtype)

    if num_classes is None:
        K = int(max(y_true.max(), y_pred.max()) + 1)
    else:
        K = int(num_classes)

    # Validate labels
    if np.any(y_true < 0) or np.any(y_true >= K):
        raise ValueError("y_true contains invalid labels")
    if np.any(y_pred < 0) or np.any(y_pred >= K):
        raise ValueError("y_pred contains invalid labels")

    # Vectorized confusion matrix using bincount
    indices = y_true * K + y_pred
    cm = np.bincount(indices, minlength=K * K).reshape(K, K)

    if normalize == 'none':
        return cm

    cm = cm.astype(float)

    if normalize == 'true':
        denom = cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        denom = cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        denom = cm.sum()
    else:
        raise ValueError("normalize must be one of {'none', 'true', 'pred', 'all'}")

    # Avoid division by zero
    eps = 1e-12

    if normalize == 'all':
        denom = denom if denom > 0 else 1.0
        cm /= (denom + eps)
    else:
        denom = np.where(denom == 0, 1.0, denom)
        cm /= (denom + eps)

    return cm