import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """

    # Convert to NumPy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Validate lengths
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    # True positives = correct predictions
    tp = np.sum(y_true == y_pred)

    total = len(y_true)

    # False positives and false negatives
    fp = total - tp
    fn = total - tp

    denominator = 2 * tp + fp + fn

    if denominator == 0:
        return 0.0

    f1 = (2 * tp) / denominator

    return float(f1)