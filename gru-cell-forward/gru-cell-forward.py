import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Single-step GRU forward pass.

    Supports:
      x      : (D,) or (N, D)
      h_prev : (H,) or (N, H)

    Returns:
      h_t    : (H,) or (N, H)
    """
    # Extract parameters as float arrays
    Wz = np.asarray(params["Wz"], dtype=float)
    Uz = np.asarray(params["Uz"], dtype=float)
    bz = np.asarray(params["bz"], dtype=float)

    Wr = np.asarray(params["Wr"], dtype=float)
    Ur = np.asarray(params["Ur"], dtype=float)
    br = np.asarray(params["br"], dtype=float)

    Wh = np.asarray(params["Wh"], dtype=float)
    Uh = np.asarray(params["Uh"], dtype=float)
    bh = np.asarray(params["bh"], dtype=float)

    # Infer dimensions
    D, H = Wz.shape

    # Convert inputs to 2D if needed
    x, x_was_1d = _as2d(x, D)
    h_prev, h_was_1d = _as2d(h_prev, H)

    # Gates
    z_t = _sigmoid(x @ Wz + h_prev @ Uz + bz)   # (N, H)
    r_t = _sigmoid(x @ Wr + h_prev @ Ur + br)   # (N, H)

    # Candidate hidden state
    h_tilde = np.tanh(x @ Wh + (r_t * h_prev) @ Uh + bh)  # (N, H)

    # Final hidden state
    h_t = (1.0 - z_t) * h_prev + z_t * h_tilde

    # Return same rank as input
    if x_was_1d and h_was_1d:
        return h_t[0]
    return h_t