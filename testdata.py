import numpy as np
from optikon import full_propositionalization

def diblock_mvn_sample(n, k=2, rho_within=0.5, rho_between=-0.5, seed=0):
    """
    Generate multivariate normal samples with a 2-block correlation structure.

    The 2k-covariance matrix has:
    - `rho_within` correlation within each of the two size-k blocks,
    - `rho_between` correlation between variables from different blocks,
    - unit variance on the diagonal.

    Args:
        n (int): Number of samples to generate.
        k (int): Size of each positively correlated block. Total dimension is 2k.
        rho_within (float): Correlation coefficient within each block.
        rho_between (float): Correlation coefficient between blocks.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: An (n, 2k) array of samples from the specified multivariate normal distribution.

    Example:
        >>> x = mvn_with_correlation(5, k=2, seed=42)
        >>> x.shape
        (5, 4)
    """
    rng = np.random.default_rng(seed=seed)
    size = 2 * k

    cov = np.full((size, size), rho_between)

    block1 = slice(0, k)
    block2 = slice(k, 2 * k)
    cov[block1, block1] = rho_within
    cov[block2, block2] = rho_within
    np.fill_diagonal(cov, 1.0)

    return rng.multivariate_normal(np.zeros(size), cov, size=n)

class TestInput:

    def __init__(self, x, y, prop_fac, selectable_sups, opt_weighted_support=None, opt_weighted_support_set=None):
        self.x = x
        self.y = y
        self.prop_fac = prop_fac
        self.selectable_sups = selectable_sups
        self.opt_weighted_support = opt_weighted_support
        self.opt_weighted_support_set = opt_weighted_support_set

TINY_1 = TestInput(
    np.array([[1], [2], [3], [4]]),
    np.array([-1, 1, 1, -1]),
    full_propositionalization,
    [
        [],
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [1],
        [1, 2],
        [1, 2, 3],
        [2],
        [2, 3],
        [3]
    ],
    2,
    np.array([1, 2]))

_SMALL_1_x = np.array([
                [0.0, 1.0, 3.0],
                [1.0, 2.0, 2.0],
                [2.0, 3.0, 1.0],
                [3.0, 4.0, 0.0],
                [4.0, 5.0, 4.0],
            ])

SMALL_1 = TestInput(
    _SMALL_1_x,
    np.array([-1, 1, 1, -1, 1]),
    full_propositionalization,
    [
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3],
    [0, 1, 2],
    [0, 1],
    [0],
    [],
    [1, 2, 3, 4],
    [1, 2, 3],
    [1, 2],
    [1],
    [2, 3, 4],
    [2, 3],
    [2],
    [3, 4],
    [3],
    [4],
    [0, 1, 2, 4],
    [0, 1, 4],
    [0, 4],
    [1, 2, 4],
    [1, 4],
    [2, 4]
    ],
    3,
    np.array([1, 2, 4])
)
