def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    mn = min(values)
    mx = max(values)

    if mn == mx:
        return [0] * len(values)

    width = (mx - mn) / num_bins
    result = []

    for v in values:
        bin_idx = int((v - mn) // width)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        result.append(bin_idx)

    return result