def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    """
    Apply isotonic regression calibration.
    """
    # Sort calibration data by predicted probability
    pairs = sorted(zip(cal_probs, cal_labels))
    probs = [p for p, _ in pairs]
    labels = [y for _, y in pairs]

    # Pool Adjacent Violators Algorithm (PAVA)
    # Each block stores: [sum_labels, count, start, end]
    blocks = []

    for i, label in enumerate(labels):
        blocks.append([float(label), 1, i, i])

        # Merge blocks while monotonicity is violated
        while len(blocks) >= 2:
            prev = blocks[-2]
            curr = blocks[-1]

            prev_mean = prev[0] / prev[1]
            curr_mean = curr[0] / curr[1]

            if prev_mean <= curr_mean:
                break

            # Merge the two violating blocks
            merged = [
                prev[0] + curr[0],
                prev[1] + curr[1],
                prev[2],
                curr[3]
            ]

            blocks.pop()
            blocks.pop()
            blocks.append(merged)

    # Expand block means back to calibrated values
    calibrated = [0.0] * len(labels)

    for block_sum, count, start, end in blocks:
        mean = block_sum / count
        for i in range(start, end + 1):
            calibrated[i] = mean

    # Linear interpolation for new predictions
    result = []

    for q in new_probs:
        if q <= probs[0]:
            result.append(float(calibrated[0]))
            continue

        if q >= probs[-1]:
            result.append(float(calibrated[-1]))
            continue

        # Find interval containing q
        lo = 0
        hi = len(probs) - 1

        while lo < hi:
            mid = (lo + hi) // 2
            if probs[mid] < q:
                lo = mid + 1
            else:
                hi = mid

        i = lo

        if probs[i] == q:
            result.append(float(calibrated[i]))
        else:
            p1 = probs[i - 1]
            p2 = probs[i]
            c1 = calibrated[i - 1]
            c2 = calibrated[i]

            # Linear interpolation
            if p2 == p1:
                value = c1
            else:
                value = c1 + (q - p1) / (p2 - p1) * (c2 - c1)

            result.append(float(value))

    return result