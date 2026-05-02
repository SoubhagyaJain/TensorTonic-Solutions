import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    Return (w_new, v_new).
    """

    # Ensure numpy arrays
    w = np.asarray(w, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    # Step 2: Update velocity (grad is already at look-ahead)
    v_new = momentum * v + lr * grad

    # Step 3: Update parameters
    w_new = w - v_new

    return w_new, v_new