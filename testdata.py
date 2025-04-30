import numpy as np

def mvn_with_correlation(n, seed=0):
    rng = np.random.default_rng(seed=seed)
    return rng.multivariate_normal([0, 0, 0, 0], [[1, 0.5, -0.5, -0.5],[0.5, 1, -0.5, -0.5], [-0.5, -0.5, 1, 0.5], [-0.5, -0.5, 0.5, 1]], size=n)
