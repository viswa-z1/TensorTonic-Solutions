import math

def ndcg(relevance_scores, k):
    """
    Compute NDCG@k.
    """
    k = min(k, len(relevance_scores))

    def dcg(scores):
        total = 0.0

        for i, rel in enumerate(scores[:k]):
            gain = (2 ** rel) - 1
            discount = math.log2(i + 2)
            total += gain / discount

        return total

    # DCG for the current ranking
    dcg_score = dcg(relevance_scores)

    # DCG for the ideal ranking
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg_score = dcg(ideal_scores)

    if idcg_score == 0:
        return 0.0

    return float(dcg_score / idcg_score)