import numpy as np

def streaming_minmax_init(D: int):
    # JSON-safe state (lists)
    return {"min": [float("inf")] * D, "max": [float("-inf")] * D}

def streaming_minmax_update(state: dict, X_batch, eps: float = 1e-8):
    """
    Update running min/max with X_batch, then return ONLY the normalized batch.
    Uses UPDATED stats (post-update). Avoids tiny eps drift by only using eps
    when a feature's range is 0.
    """
    X = np.asarray(X_batch, dtype=np.float64)
    D = len(state["min"])

    if X.ndim != 2 or X.shape[1] != D:
        raise ValueError(f"X_batch must have shape (B, {D}).")
    if X.shape[0] == 0:
        return []  # expected just the normalized batch

    smin = np.asarray(state["min"], dtype=np.float64)
    smax = np.asarray(state["max"], dtype=np.float64)

    batch_min = X.min(axis=0)
    batch_max = X.max(axis=0)

    # Update global stats
    smin = np.minimum(smin, batch_min)
    smax = np.maximum(smax, batch_max)

    # Denominator: use eps ONLY for constant columns (range == 0)
    rng = smax - smin
    denom = np.where(rng == 0.0, eps, rng)

    X_norm = (X - smin) / denom

    # Save back JSON-safe
    state["min"] = smin.tolist()
    state["max"] = smax.tolist()

    # Return ONLY normalized batch (matches expected output format)
    return X_norm.tolist()