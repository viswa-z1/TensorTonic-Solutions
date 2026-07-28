import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient between two raters.
    """
    n = len(rater1)
    assert n == len(rater2), "Rater lists must be the same length"

    # Observed agreement p_o: proportion of positions where labels match
    agreements = sum(1 for a, b in zip(rater1, rater2) if a == b)
    p_o = agreements / n

    # Collect all distinct labels from both raters
    labels = set(rater1) | set(rater2)

    # Count label frequencies for each rater
    counts_rater1 = {label: 0 for label in labels}
    counts_rater2 = {label: 0 for label in labels}

    for label in rater1:
        counts_rater1[label] += 1
    for label in rater2:
        counts_rater2[label] += 1

    # Compute expected agreement p_e
    p_e = 0.0
    for label in labels:
        p1 = counts_rater1[label] / n
        p2 = counts_rater2[label] / n
        p_e += p1 * p2

    # Handle degenerate case where denominator is zero (perfect agreement)
    if p_e == 1.0:
        return 1.0

    # Compute Cohen's kappa
    kappa = (p_o - p_e) / (1 - p_e)
    return kappa
