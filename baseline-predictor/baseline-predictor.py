import numpy as np

def baseline_predict(ratings_matrix, target_pairs):
    """
    Baseline predictor: r_hat_ui = mu + b_u + b_i
    0 means unrated (excluded from means).

    Returns: list[float]
    """
    R = np.asarray(ratings_matrix, dtype=np.float64)
    pairs = np.asarray(target_pairs, dtype=np.int64)

    mask = (R != 0)
    vals = R[mask]
    if vals.size == 0:
        raise ValueError("ratings_matrix must contain at least one non-zero rating.")
    mu = float(vals.mean())

    # User bias
    user_sum = (R * mask).sum(axis=1)
    user_cnt = mask.sum(axis=1)
    user_mean = np.divide(user_sum, user_cnt, out=np.zeros_like(user_sum), where=user_cnt != 0)
    b_u = user_mean - mu

    # Item bias
    item_sum = (R * mask).sum(axis=0)
    item_cnt = mask.sum(axis=0)
    item_mean = np.divide(item_sum, item_cnt, out=np.zeros_like(item_sum), where=item_cnt != 0)
    b_i = item_mean - mu

    u = pairs[:, 0]
    i = pairs[:, 1]
    preds = mu + b_u[u] + b_i[i]

    return preds.tolist()