from bisect import bisect_right

def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    """
    Apply isotonic regression calibration.
    Returns a list of calibrated probabilities.
    """
    # 1) Sort calibration data by predicted probability
    pairs = sorted(zip(cal_probs, cal_labels))
    probs = [p for p, _ in pairs]
    labels = [float(y) for _, y in pairs]

    # 2) Fit isotonic regression with Pool Adjacent Violators (PAV)
    block_means = []
    block_counts = []

    for y in labels:
        block_means.append(y)
        block_counts.append(1)

        # Merge adjacent violating blocks
        while len(block_means) >= 2 and block_means[-2] > block_means[-1]:
            total_count = block_counts[-2] + block_counts[-1]
            total_sum = block_means[-2] * block_counts[-2] + block_means[-1] * block_counts[-1]
            merged_mean = total_sum / total_count

            block_means[-2] = merged_mean
            block_counts[-2] = total_count
            block_means.pop()
            block_counts.pop()

    # Expand block means back to pointwise calibrated values
    calibrated = []
    for mean, count in zip(block_means, block_counts):
        calibrated.extend([mean] * count)

    # 3) Interpolate new predictions
    result = []
    n = len(probs)

    for q in new_probs:
        # Clamp outside range
        if q <= probs[0]:
            result.append(calibrated[0])
            continue
        if q >= probs[-1]:
            result.append(calibrated[-1])
            continue

        # Find interval probs[i] <= q < probs[i+1]
        i = bisect_right(probs, q) - 1

        p0, p1 = probs[i], probs[i + 1]
        c0, c1 = calibrated[i], calibrated[i + 1]

        # Linear interpolation
        t = (q - p0) / (p1 - p0)
        result.append(c0 + t * (c1 - c0))

    return result