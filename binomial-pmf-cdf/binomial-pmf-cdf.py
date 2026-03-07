import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    Returns: (pmf, cdf)
    """
    n = int(n)
    k = int(k)
    p = float(p)

    # Edge cases
    if p == 0.0:
        pmf = 1.0 if k == 0 else 0.0
        cdf = 1.0
        return pmf, cdf

    if p == 1.0:
        pmf = 1.0 if k == n else 0.0
        cdf = 0.0 if k < n else 1.0
        return pmf, cdf

    i = np.arange(k + 1)
    pmf_values = comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    pmf = float(pmf_values[-1])
    cdf = float(np.sum(pmf_values))

    return pmf, cdf