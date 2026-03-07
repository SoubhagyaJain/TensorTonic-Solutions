import numpy as np

def chi2_independence(C):
    C = np.asarray(C, dtype=float)

    row_sums = C.sum(axis=1, keepdims=True)
    col_sums = C.sum(axis=0, keepdims=True)
    total = C.sum()

    expected = (row_sums @ col_sums) / total
    chi2 = float(np.sum((C - expected) ** 2 / expected))

    return chi2, expected