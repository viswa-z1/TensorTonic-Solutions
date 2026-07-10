def detect_drift(reference_counts, production_counts, threshold):
    """
    Compare reference and production distributions to detect data drift.
    """
    ref_total = sum(reference_counts)
    prod_total = sum(production_counts)

    ref_probs = [x / ref_total for x in reference_counts]
    prod_probs = [x / prod_total for x in production_counts]

    score = 0.5 * sum(abs(r - p) for r, p in zip(ref_probs, prod_probs))

    return {
        "score": float(score),
        "drift_detected": score > threshold
    }