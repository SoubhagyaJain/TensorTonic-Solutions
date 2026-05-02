import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    Return (w_new, m_new, v_new).
    """

    # Ensure numpy arrays
    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    # Step 1: First moment
    m_new = beta1 * m + (1 - beta1) * grad

    # Step 2: Second moment
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    # Step 3: Nesterov-adjusted update
    nesterov_term = beta1 * m_new + (1 - beta1) * grad

    w_new = w - lr * nesterov_term / (np.sqrt(v_new) + eps)

    return w_new, m_new, v_new