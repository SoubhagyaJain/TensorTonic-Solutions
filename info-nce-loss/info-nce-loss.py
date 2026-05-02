import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """

    # Convert to numpy arrays
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)

    # Step 1: Similarity matrix (N, N)
    S = (Z1 @ Z2.T) / temperature

    # Step 2: Numerical stability (row-wise max subtraction)
    S_max = np.max(S, axis=1, keepdims=True)
    S_stable = S - S_max

    # Step 3: log-softmax
    exp_S = np.exp(S_stable)
    log_sum_exp = np.log(np.sum(exp_S, axis=1))

    # Step 4: positive pairs (diagonal)
    pos = np.diag(S_stable)

    # Step 5: loss per sample
    loss = -pos + log_sum_exp

    # Step 6: mean loss
    return np.mean(loss)