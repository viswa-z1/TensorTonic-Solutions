import numpy as np

def silhouette_score(X, labels):
    """
    Compute mean Silhouette Score.

    Parameters:
        X      : shape (n_samples, n_features)
        labels : shape (n_samples,)

    Returns:
        float
    """

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    n = X.shape[0]

    # Pairwise Euclidean distance matrix
    diff = X[:, None, :] - X[None, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))

    unique_labels = np.unique(labels)

    # Intra-cluster distance
    a = np.zeros(n, dtype=float)

    # Nearest-cluster distance
    b = np.full(n, np.inf, dtype=float)

    for cluster in unique_labels:

        same_mask = (labels == cluster)
        same_idx = np.where(same_mask)[0]

        # ----- Compute a(i) -----
        if len(same_idx) > 1:

            intra = distances[np.ix_(same_idx, same_idx)]

            # subtract self-distance
            a[same_idx] = (
                np.sum(intra, axis=1)
                / (len(same_idx) - 1)
            )
        else:
            a[same_idx] = 0.0

        # ----- Compute b(i) -----
        for other_cluster in unique_labels:

            if other_cluster == cluster:
                continue

            other_mask = (labels == other_cluster)
            other_idx = np.where(other_mask)[0]

            inter = distances[np.ix_(same_idx, other_idx)]

            inter_mean = np.mean(inter, axis=1)

            b[same_idx] = np.minimum(
                b[same_idx],
                inter_mean
            )

    # Silhouette score per sample
    denom = np.maximum(a, b)

    s = np.where(
        denom > 0,
        (b - a) / denom,
        0.0
    )

    return float(np.mean(s))