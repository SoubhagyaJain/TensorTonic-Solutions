import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    Returns: (var, std)
    """
    x = np.asarray(x, dtype=float)
    var = float(np.var(x, ddof=1))   # sample variance with Bessel correction
    std = float(np.std(x, ddof=1))   # sample std with Bessel correction
    return var, std