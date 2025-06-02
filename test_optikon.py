import numpy as np
import numba as nb
from optikon import compute_bounds, make_maxheap_class, max_weighted_support, Propositionalization

def test_compute_bounds_nonempty():
    x = np.array([[1., 2.], [0., -1.]])
    l, u = compute_bounds(x)
    assert np.allclose(l, [0., -1.])
    assert np.allclose(u, [1., 2.])

def test_compute_bounds_empty():
    x = np.empty((0, 3))
    l, u = compute_bounds(x)
    assert l.shape == (3,)
    assert u.shape == (3,)
    assert np.all(np.isposinf(l))
    assert np.all(np.isneginf(u))

def test_int_string_maxheap():
    IntStringHeap = make_maxheap_class(nb.types.int64, nb.types.string)
    int_string_heap = IntStringHeap()
    assert not int_string_heap
    int_string_heap.push(1, 'one')
    assert int_string_heap
    int_string_heap.push(10, 'ten')
    int_string_heap.push(2, 'two')
    int_string_heap.push(-1, 'minus one')
    assert int_string_heap.pop() == (10, 'ten')
    assert int_string_heap.pop() == (2, 'two')
    int_string_heap.push(2, 'two')
    assert int_string_heap.pop() == (2, 'two')
    assert int_string_heap.pop() == (1, 'one')
    assert int_string_heap.pop() == (-1, 'minus one')
    assert not int_string_heap

def test_float_string_maxheap():
    FloatStringHeap = make_maxheap_class(nb.types.float64, nb.types.string)
    float_string_heap = FloatStringHeap()
    assert not float_string_heap
    float_string_heap.push(1, 'one')
    assert float_string_heap
    float_string_heap.push(10, 'ten')
    float_string_heap.push(2, 'two')
    float_string_heap.push(-1, 'minus one')
    assert float_string_heap.pop() == (10, 'ten')
    assert float_string_heap.pop() == (2, 'two')
    float_string_heap.push(2, 'two')
    assert float_string_heap.pop() == (2, 'two')
    assert float_string_heap.pop() == (1, 'one')
    assert float_string_heap.pop() == (-1, 'minus one')
    assert not float_string_heap

def test_propositionalisation_fancy_indexing():
    v = np.array([0, 1, 2, 3], dtype=np.int64)
    t = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    s = np.array([10, 11, 12, 13], dtype=np.int64)
    p = Propositionalization(v, t, s)

    idxs = np.array([1, 3], dtype=np.int64)
    p_sel = p[idxs]

    np.testing.assert_array_equal(p_sel.v, np.array([1, 3]))
    np.testing.assert_array_equal(p_sel.t, np.array([0.2, 0.4]))
    np.testing.assert_array_equal(p_sel.s, np.array([11, 13]))

def test_lex_treesearch():
    from testdata import SMALL_1
    key, val, created, candidate_edges = max_weighted_support(SMALL_1.x, SMALL_1.y, SMALL_1.prop)
    assert val == 3
