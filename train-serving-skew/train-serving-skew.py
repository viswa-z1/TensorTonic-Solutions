import numpy as np

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    """
    Detect train-serving skew using PSI.
    """
    result = {}

    # Process features present in both dictionaries
    for feature in train_dist:
        if feature not in serving_dist:
            continue

        train = np.asarray(train_dist[feature], dtype=float)
        serving = np.asarray(serving_dist[feature], dtype=float)

        # Prevent log(0) / division by zero
        train = train + eps
        serving = serving + eps

        psi = np.sum((serving - train) * np.log(serving / train))

        result[feature] = {
            "psi": float(psi),
            "skewed": bool(psi >= threshold)
        }

    return result