import numpy as np
from optikon import compute_bounds

def test_compute_bounds_nonempty():
    x = np.array([[1., 2.], [0., -1.]])
    l, u = compute_bounds(x)
    assert np.allclose(l, [0., -1.])
    assert np.allclose(u, [1., 2.])

