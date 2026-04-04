def seasonal_average(series, period):
    """
    Compute the average value for each position in the seasonal cycle.
    """
    result = []

    for p in range(period):
        values = []
        for i in range(p, len(series), period):
            values.append(series[i])
        result.append(sum(values) / len(values))

    return result