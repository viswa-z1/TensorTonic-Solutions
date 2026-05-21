def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k.

    Parameters:
        recommended : ranked recommendation list
        relevant    : ground-truth relevant items
        k           : cutoff

    Returns:
        [precision, recall]
    """

    # Take top-k recommendations
    top_k = recommended[:k]

    # Convert relevant items to set for fast lookup
    relevant_set = set(relevant)

    # Count hits
    hits = sum(1 for item in top_k if item in relevant_set)

    # Compute metrics
    precision = hits / k
    recall = hits / len(relevant_set)

    return [float(precision), float(recall)]
    