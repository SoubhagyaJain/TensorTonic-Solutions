import numpy as np

def decision_tree_split(X, y):
    """
    Best split by max Gini information gain.
    Tie-break: smallest feature index, then smallest threshold.

    Returns: [best_feature_index, best_threshold]
    """
    X = np.asarray(X)
    y = np.asarray(y)

    N, D = X.shape
    _, y_enc = np.unique(y, return_inverse=True)
    K = int(y_enc.max()) + 1

    # Parent gini
    parent_counts = np.bincount(y_enc, minlength=K).astype(np.float64)
    p = parent_counts / parent_counts.sum()
    parent_gini = 1.0 - np.sum(p * p)

    best_gain = -np.inf
    best_f = 0
    best_thr = 0.0

    for f in range(D):
        col = X[:, f]
        order = np.argsort(col, kind="mergesort")
        x_sorted = col[order]
        y_sorted = y_enc[order]

        # valid split points where feature value changes
        diff = x_sorted[1:] != x_sorted[:-1]
        if not np.any(diff):
            continue

        # cumulative counts for left side per class
        left_counts = np.zeros(K, dtype=np.int64)
        right_counts = np.bincount(y_sorted, minlength=K).astype(np.int64)

        for i in range(N - 1):
            c = y_sorted[i]
            left_counts[c] += 1
            right_counts[c] -= 1

            if not diff[i]:
                continue  # can't split between equal feature values

            n_left = i + 1
            n_right = N - n_left

            # Gini(left)
            pl = left_counts / n_left
            g_left = 1.0 - np.sum(pl * pl)

            # Gini(right)
            pr = right_counts / n_right
            g_right = 1.0 - np.sum(pr * pr)

            g_split = (n_left / N) * g_left + (n_right / N) * g_right
            gain = parent_gini - g_split

            thr = float((x_sorted[i] + x_sorted[i + 1]) / 2.0)

            # tie-break: max gain, then min feature, then min threshold
            if (gain > best_gain + 1e-15) or (
                abs(gain - best_gain) <= 1e-15 and (f < best_f or (f == best_f and thr < best_thr))
            ):
                best_gain = gain
                best_f = int(f)
                best_thr = thr

    return [best_f, best_thr]