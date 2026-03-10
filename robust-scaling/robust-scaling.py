def robust_scaling(values):
    """
    Scale values using median and interquartile range.
    """
    def median(arr):
        n = len(arr)
        mid = n // 2
        if n % 2 == 1:
            return float(arr[mid])
        return (arr[mid - 1] + arr[mid]) / 2.0

    if len(values) == 1:
        return [0.0]

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    med = median(sorted_vals)

    mid = n // 2
    if n % 2 == 0:
        lower_half = sorted_vals[:mid]
        upper_half = sorted_vals[mid:]
    else:
        lower_half = sorted_vals[:mid]
        upper_half = sorted_vals[mid + 1:]

    q1 = median(lower_half)
    q3 = median(upper_half)
    iqr = q3 - q1

    if iqr == 0:
        return [float(v - med) for v in values]

    return [float((v - med) / iqr) for v in values]