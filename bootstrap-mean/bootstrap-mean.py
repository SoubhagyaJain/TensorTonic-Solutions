import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    x = np.asarray(x, dtype=float)
    n = x.size

    boot_means = np.empty(n_bootstrap, dtype=float)

    # Process in chunks to avoid huge memory use
    chunk_size = max(1, min(n_bootstrap, 1000000 // max(1, n)))

    for start in range(0, n_bootstrap, chunk_size):
        end = min(start + chunk_size, n_bootstrap)
        size = end - start

        if rng is not None:
            indices = rng.integers(0, n, size=(size, n))
        else:
            indices = np.random.randint(0, n, size=(size, n))

        boot_means[start:end] = x[indices].mean(axis=1)

    alpha = 1.0 - ci
    lower = float(np.quantile(boot_means, alpha / 2))
    upper = float(np.quantile(boot_means, 1 - alpha / 2))

    return boot_means, lower, upper