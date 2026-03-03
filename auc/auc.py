import numpy as np

def auc(fpr, tpr) -> float:
    fpr = np.asarray(fpr, dtype=np.float64).ravel()
    tpr = np.asarray(tpr, dtype=np.float64).ravel()

    if fpr.size != tpr.size or fpr.size < 2:
        raise ValueError("fpr and tpr must have the same length and at least 2 points.")

    # Ensure FPR is increasing (sort to be safe)
    idx = np.argsort(fpr, kind="mergesort")
    fpr = fpr[idx]
    tpr = tpr[idx]

    # Trapezoidal rule
    dx = fpr[1:] - fpr[:-1]
    auc_val = np.sum(dx * (tpr[1:] + tpr[:-1]) * 0.5)

    # Clip for numerical safety
    return float(np.clip(auc_val, 0.0, 1.0))
    