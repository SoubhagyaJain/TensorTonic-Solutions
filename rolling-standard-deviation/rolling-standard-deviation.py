import math

def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    result = []

    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        mean = sum(window) / window_size
        variance = sum((x - mean) ** 2 for x in window) / window_size
        result.append(math.sqrt(variance))

    return result