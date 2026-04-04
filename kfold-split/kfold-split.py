import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    indices = np.arange(N, dtype=int)

    if shuffle:
        if rng is not None:
            indices = rng.permutation(indices)
        else:
            np.random.shuffle(indices)

    base_size = N // k
    remainder = N % k

    folds = []
    start = 0

    for i in range(k):
        fold_size = base_size + (1 if i < remainder else 0)
        end = start + fold_size

        val_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:])).astype(int)

        folds.append((train_idx, val_idx))
        start = end

    return folds 

    