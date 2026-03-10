def winsorize(values, lower_pct, upper_pct):
    """
    Clip values at the given percentile bounds.
    Return a list of floats.
    """
    arr = sorted(values)
    n = len(arr)

    def percentile(p):
        # index using linear interpolation
        k = (n - 1) * p / 100.0
        lo = int(k)
        hi = min(lo + 1, n - 1)
        frac = k - lo
        return arr[lo] + frac * (arr[hi] - arr[lo])

    lower_bound = percentile(lower_pct)
    upper_bound = percentile(upper_pct)

    result = []
    for v in values:
        if v < lower_bound:
            result.append(float(lower_bound))
        elif v > upper_bound:
            result.append(float(upper_bound))
        else:
            result.append(float(v))

    return result