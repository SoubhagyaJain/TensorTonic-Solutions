import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    Returns np.nan if either vector is a zero vector.
    """
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    
    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)
    
    if norm_v == 0 or norm_w == 0:
        return np.nan
    
    cos_theta = np.dot(v, w) / (norm_v * norm_w)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    return float(np.arccos(cos_theta))