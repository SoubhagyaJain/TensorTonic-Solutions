import numpy as np

def rnn_step_backward(dh, cache):
    """
    Backward pass for a single vanilla RNN timestep.

    Forward:
        h_t = tanh(W x_t + U h_prev + b)

    Args:
        dh: (H,) upstream gradient dL/dh_t
        cache: [x_t, h_prev, h_t, W, U, b]

    Returns:
        (dx_t, dh_prev, dW, dU, db)
    """
    x_t, h_prev, h_t, W, U, b = cache

    dh = np.asarray(dh, dtype=np.float64).reshape(-1)
    x_t = np.asarray(x_t, dtype=np.float64).reshape(-1)
    h_prev = np.asarray(h_prev, dtype=np.float64).reshape(-1)
    h_t = np.asarray(h_t, dtype=np.float64).reshape(-1)
    W = np.asarray(W, dtype=np.float64)
    U = np.asarray(U, dtype=np.float64)

    # dz = dh * (1 - h_t^2)
    dz = dh * (1.0 - h_t * h_t)          # (H,)

    dx_t = W.T @ dz                       # (D,)
    dh_prev = U.T @ dz                    # (H,)
    dW = dz[:, None] * x_t[None, :]       # (H, D)
    dU = dz[:, None] * h_prev[None, :]    # (H, H)
    db = dz                                # (H,)

    return dx_t, dh_prev, dW, dU, db