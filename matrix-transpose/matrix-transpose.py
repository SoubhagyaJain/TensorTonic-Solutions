import numpy as np

def matrix_transpose(A):
    """
    Manual transpose without using .T or np.transpose().
    Accepts: 2D NumPy array (or array-like)
    Returns: new NumPy array of shape (M, N)
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")

    n, m = A.shape
    out = np.empty((m, n), dtype=A.dtype)

    for i in range(n):
        for j in range(m):
            out[j, i] = A[i, j]

    return out