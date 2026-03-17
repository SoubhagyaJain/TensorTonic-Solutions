import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    new_m = beta1 * m + (1 - beta1) * grad
    new_v = beta2 * v + (1 - beta2) * (grad ** 2)

    nesterov_term = beta1 * new_m + (1 - beta1) * grad
    new_w = w - lr * nesterov_term / (np.sqrt(new_v) + eps)

    return new_w, new_m, new_v