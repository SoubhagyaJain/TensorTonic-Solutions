import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if not seqs:
        return np.empty((0, 0), dtype=int)

    if max_len is None:
        max_len = max(len(seq) for seq in seqs) if seqs else 0

    n = len(seqs)
    result = np.full((n, max_len), pad_value, dtype=int)

    for i, seq in enumerate(seqs):
        trunc = seq[:max_len]
        result[i, :len(trunc)] = trunc

    return result