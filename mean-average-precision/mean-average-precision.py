import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    """
    Compute Mean Average Precision (mAP) for multiple retrieval queries.
    """
    if len(y_true_list) != len(y_score_list):
        raise ValueError("y_true_list and y_score_list must have the same length")

    ap_per_query = []

    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        # Sort by descending score
        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]

        # Total relevant items in the full query
        total_relevant = y_true_sorted.sum()

        if total_relevant == 0:
            ap_per_query.append(0.0)
            continue

        # Apply cutoff k
        if k is not None:
            y_true_sorted = y_true_sorted[:k]

        # Cumulative relevant count
        cum_rel = np.cumsum(y_true_sorted)

        # Precision at each rank
        ranks = np.arange(1, len(y_true_sorted) + 1)
        precision = cum_rel / ranks

        # AP: average precision over relevant positions, normalized by total relevant items
        ap = np.sum(precision * y_true_sorted) / total_relevant
        ap_per_query.append(float(ap))

    map_value = float(np.mean(ap_per_query)) if ap_per_query else 0.0

    return map_value, ap_per_query