import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.

    Parameters
    ----------
    y_train : array-like
        Training labels.
    X_test : array-like
        Test features (only the number of samples matters).

    Returns
    -------
    np.ndarray
        Predictions for all test samples.
    """
    y_train = np.asarray(y_train)

    # Handle empty test set
    n_test = len(X_test)
    if n_test == 0:
        return np.array([], dtype=int)

    # Find unique labels and their counts
    labels, counts = np.unique(y_train, return_counts=True)

    # Majority class (np.unique returns sorted labels, ensuring stable tie-breaking)
    majority_class = labels[np.argmax(counts)]

    # Predict the majority class for all test samples
    return np.full(n_test, majority_class, dtype=y_train.dtype)