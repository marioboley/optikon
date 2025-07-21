import numpy as np
from optikon import full_propositionalization

def mvn_with_correlation(n, seed=0):
    rng = np.random.default_rng(seed=seed)
    return rng.multivariate_normal([0, 0, 0, 0], [[1, 0.5, -0.5, -0.5],[0.5, 1, -0.5, -0.5], [-0.5, -0.5, 1, 0.5], [-0.5, -0.5, 0.5, 1]], size=n)

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
