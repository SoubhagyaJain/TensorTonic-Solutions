import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    arr = np.array(X, dtype=float, copy=True)
    was_1d = (arr.ndim == 1)

    if was_1d:
        arr = arr.reshape(-1, 1)

    if strategy == 'mean':
        stats = np.nanmean(arr, axis=0)
    elif strategy == 'median':
        stats = np.nanmedian(arr, axis=0)
    else:
        raise ValueError("strategy must be 'mean' or 'median'")

    # Fully-NaN columns -> fill with 0
    stats = np.where(np.isnan(stats), 0.0, stats)

    nan_mask = np.isnan(arr)
    arr[nan_mask] = np.take(stats, np.where(nan_mask)[1])

    return arr.ravel() if was_1d else arr