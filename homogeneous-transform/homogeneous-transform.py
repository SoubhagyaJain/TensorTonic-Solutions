import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply a 4x4 homogeneous transform T to 3D point(s).

    Args:
        T: array-like, shape (4,4)
        points: array-like, shape (3,) or (N,3)

    Returns:
        np.ndarray shape (3,) if input was (3,)
        np.ndarray shape (N,3) if input was (N,3)
    """
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(points, dtype=np.float64)

    if T.shape != (4, 4):
        raise ValueError("T must have shape (4, 4)")

    single = (P.ndim == 1)
    if single:
        if P.shape[0] != 3:
            raise ValueError("Single point must have shape (3,)")
        P2 = P.reshape(1, 3)
    else:
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("Batch points must have shape (N, 3)")
        P2 = P

    n = P2.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    Ph = np.concatenate([P2, ones], axis=1)          # (N,4)

    Ph_t = Ph @ T.T                                  # (N,4) because points are row-vectors
    out = Ph_t[:, :3]                                # drop homogeneous coord

    return out[0] if single else out
    