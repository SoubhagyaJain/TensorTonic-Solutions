import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")

    n = len(X)
    if rng is None:
        rng = np.random.default_rng()

    indices = rng.permutation(n)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    for start in range(0, n, batch_size):
        end = start + batch_size

        if drop_last and end > n:
            break

        yield X_shuffled[start:end], y_shuffled[start:end]