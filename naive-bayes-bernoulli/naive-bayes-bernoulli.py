import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute unnormalized log posteriors for Bernoulli Naive Bayes.

    Parameters
    ----------
    X_train : array-like, shape (n_train, d)
        Binary training features {0,1}
    y_train : array-like, shape (n_train,)
        Training labels
    X_test : array-like, shape (n_test, d)
        Binary test features {0,1}

    Returns
    -------
    log_posteriors : ndarray, shape (n_test, n_classes)
        Unnormalized log posterior scores for each test sample and class.
        Classes are ordered in ascending order.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=np.float64)

    classes = np.unique(y_train)
    n_classes = len(classes)
    n_train, d = X_train.shape
    n_test = X_test.shape[0]

    # Log priors
    log_prior = np.empty(n_classes)
    # Log P(x_i=1|y)
    log_theta = np.empty((n_classes, d))
    # Log P(x_i=0|y)
    log_one_minus_theta = np.empty((n_classes, d))

    for idx, c in enumerate(classes):
        mask = (y_train == c)
        X_c = X_train[mask]
        n_c = X_c.shape[0]

        # Prior P(y)
        log_prior[idx] = np.log(n_c / n_train)

        # Laplace smoothing (alpha = 1)
        theta = (X_c.sum(axis=0) + 1.0) / (n_c + 2.0)

        log_theta[idx] = np.log(theta)
        log_one_minus_theta[idx] = np.log(1.0 - theta)

    # Compute log posteriors
    # shape: (n_test, n_classes)
    log_posteriors = (
        X_test[:, None, :] * log_theta[None, :, :]
        + (1.0 - X_test[:, None, :]) * log_one_minus_theta[None, :, :]
    ).sum(axis=2)

    log_posteriors += log_prior

    return log_posteriors