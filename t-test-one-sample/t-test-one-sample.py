import numpy as np

def t_test_one_sample(x, mu0):
    x = np.asarray(x, dtype=float)

    n = x.size
    mean = np.mean(x)
    s = np.std(x, ddof=1)  # sample std with Bessel correction

    t_stat = (mean - mu0) / (s / np.sqrt(n))
    return float(t_stat)