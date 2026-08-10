def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    # Initialize bins
    bin_counts = [0] * n_bins
    bin_sums = [0.0] * n_bins
    bin_correct = [0] * n_bins

    # Assign each prediction to a bin
    for y, p in zip(y_true, y_pred):
        # Handle p = 1.0 separately (last bin)
        if p == 1.0:
            bin_idx = n_bins - 1
        else:
            bin_idx = int(p * n_bins)

        bin_counts[bin_idx] += 1
        bin_sums[bin_idx] += p
        bin_correct[bin_idx] += y

    # Compute ECE
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        if bin_counts[i] == 0:
            continue  # Skip empty bins

        # Compute accuracy and confidence for the bin
        acc = bin_correct[i] / bin_counts[i]
        conf = bin_sums[i] / bin_counts[i]

        # Weighted contribution to ECE
        ece += (bin_counts[i] / total_samples) * abs(acc - conf)

    return ece