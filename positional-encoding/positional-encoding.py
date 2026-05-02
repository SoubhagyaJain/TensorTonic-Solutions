import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """

    # Positions: (seq_len, 1)
    pos = np.arange(seq_len, dtype=float)[:, np.newaxis]

    # Dimension indices: (d_model,)
    dim = np.arange(d_model, dtype=float)

    # Compute the exponent term: (2i / d_model)
    exponent = (2 * (dim // 2)) / d_model

    # Compute denominator: base^(2i/d_model)
    denom = base ** exponent

    # Compute angles: (seq_len, d_model)
    angles = pos / denom

    # Initialize PE
    pe = np.zeros((seq_len, d_model), dtype=float)

    # Apply sin to even indices
    pe[:, 0::2] = np.sin(angles[:, 0::2])

    # Apply cos to odd indices
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe