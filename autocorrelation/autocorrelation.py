def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    n = len(series)
    mean = sum(series) / n

    # Total variance (autocovariance at lag 0 denominator)
    gamma0 = sum((x - mean) ** 2 for x in series)

    # Handle constant series
    if gamma0 == 0:
        return [1.0] + [0.0] * max_lag

    result = []
    for k in range(max_lag + 1):
        num = 0.0
        for t in range(n - k):
            num += (series[t] - mean) * (series[t + k] - mean)
        result.append(num / gamma0)

    return result