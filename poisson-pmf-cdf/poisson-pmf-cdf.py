import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    Returns: (pmf, cdf)
    """
    k = int(k)
    lam = float(lam)

    ks = np.arange(k + 1, dtype=float)

    if k == 0:
        log_fact = np.array([0.0])
    else:
        log_fact = np.concatenate((
            [0.0],
            np.cumsum(np.log(np.arange(1, k + 1, dtype=float)))
        ))

    log_pmfs = -lam + ks * np.log(lam) - log_fact
    pmfs = np.exp(log_pmfs)

    pmf = float(pmfs[-1])
    cdf = float(np.sum(pmfs))

    return pmf, cdf