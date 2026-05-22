import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.

    Parameters:
        y_true : array-like of shape (N,)
                 Correct class labels

        y_pred : array-like of shape (N, K)
                 Predicted probabilities

    Returns:
        float : average cross-entropy loss
    """

    # Convert to NumPy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)

    # Validate shapes
    if len(y_true) != y_pred.shape[0]:
        raise ValueError("Mismatch between y_true and y_pred")

    # Extract probabilities of correct classes
    correct_probs = y_pred[np.arange(len(y_true)), y_true]

    # Compute cross-entropy
    loss = -np.mean(np.log(correct_probs))

    return float(loss)