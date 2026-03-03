import numpy as np

def _softmax(x, axis=-1):
    # stable softmax
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def multi_head_attention(Q, K, V, WQ, WK, WV, WO, num_heads):
    """
    Multi-Head Attention (NumPy only)

    Args:
        Q, K, V: np.ndarray, shape (B, T, d_model)
        WQ, WK, WV, WO: np.ndarray, shape (d_model, d_model)
        num_heads: int

    Returns:
        out: np.ndarray, shape (B, T, d_model)
    """
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    WQ = np.asarray(WQ, dtype=np.float64)
    WK = np.asarray(WK, dtype=np.float64)
    WV = np.asarray(WV, dtype=np.float64)
    WO = np.asarray(WO, dtype=np.float64)

    B, T, d_model = Q.shape
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    d_head = d_model // num_heads

    # 1) Linear projections
    Qp = Q @ WQ   # (B, T, d_model)
    Kp = K @ WK   # (B, T, d_model)
    Vp = V @ WV   # (B, T, d_model)

    # 2) Split into heads: (B, h, T, d_head)
    Qh = Qp.reshape(B, T, num_heads, d_head).transpose(0, 2, 1, 3)
    Kh = Kp.reshape(B, T, num_heads, d_head).transpose(0, 2, 1, 3)
    Vh = Vp.reshape(B, T, num_heads, d_head).transpose(0, 2, 1, 3)

    # 3) Scaled dot-product attention per head
    scale = 1.0 / np.sqrt(d_head)
    scores = (Qh @ Kh.transpose(0, 1, 3, 2)) * scale   # (B, h, T, T)
    attn = _softmax(scores, axis=-1)                   # (B, h, T, T)
    head_out = attn @ Vh                               # (B, h, T, d_head)

    # 4) Concat heads: (B, T, d_model)
    concat = head_out.transpose(0, 2, 1, 3).reshape(B, T, d_model)

    # 5) Output projection
    out = concat @ WO                                  # (B, T, d_model)
    return out