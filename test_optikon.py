import numpy as np
import numba as nb
import pytest
from optikon import sort_columns, compute_bounds, make_maxheap_class, max_weighted_support_bb, max_weighted_support_greedy, Propositionalization, full_propositionalization, equal_width_propositionalization
from testdata import SMALL_1, TINY_1

def test_sort_columns():
    x = np.array([[1.0, 4.0], [-1.0, 5.0], [0.0, 4.5]])
    x = sort_columns(x)
    assert np.array_equal(x, np.array([[-1.0, 4.0], [-0.0, 4.5], [1.0, 5.0]]))

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

def test_fullprop():
    x = np.array([[1.0, 4.0], [-1.0, 5.0], [0.0, 4.5]])
    prop = full_propositionalization(x)
    assert len(prop) == 8
    assert np.array_equal(prop.v, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
    assert np.array_equal(prop.s*prop.t, np.array([1.0, 0.0, -1.0, 0.0, 5.0, 4.5, 4.0, 4.5]))

def test_equal_width_prop():
    x = np.linspace(0, 12, 27).reshape(-1, 1)
    prop = equal_width_propositionalization(x)
    assert len(prop) == 4

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

@pytest.mark.parametrize('case', [TINY_1, SMALL_1])
def test_lex_treesearch(case):
    res, val, stats = max_weighted_support_bb(case.x, case.y, case.prop_fac(case.x))
    assert val == case.opt_weighted_support
    np.testing.assert_array_equal(res.support_all(case.x), case.opt_weighted_support_set)

@pytest.mark.parametrize('case', [TINY_1, SMALL_1])
def test_greedy_max_weighted_support(case):
    res, val, _ = max_weighted_support_greedy(case.x, case.y)
    assert val == case.opt_weighted_support
    np.testing.assert_array_equal(res.support_all(case.x), case.opt_weighted_support_set)

def test_str_methods():
    v = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    s = np.array([1, -1, -1, 1, -1], dtype=np.int64)
    t = np.array([-0.25, -0.5, 11.5, -2.115, 0.2], dtype=np.float64)
    prop = Propositionalization(v, t, s)

    # Test str_from_prop
    assert prop.str_from_prop(0, 2) == 'x1 >= -0.25'
    assert prop.str_from_prop(1, 2) == 'x1 <= 0.50'
    assert prop.str_from_prop(2, 1) == 'x2 <= -11.5'
    assert prop.str_from_prop(2, 2) == 'x2 <= -11.50'
    assert prop.str_from_prop(3, 2) == 'x3 >= -2.12'
    assert prop.str_from_prop(4, 2) == 'x3 <= -0.20'

    # Test as_str
    assert prop.as_str('[', ']', ', ', 2) == '[x1 >= -0.25, x1 <= 0.50, x2 <= -11.50, x3 >= -2.12, x3 <= -0.20]'

    # Test as_conj_str
    assert prop.as_conj_str(3) == 'x1 >= -0.250 & x1 <= 0.500 & x2 <= -11.500 & x3 >= -2.115 & x3 <= -0.200'

    # Test as_disj_str
    assert prop.as_disj_str(0) == 'x1 >= 0 | x1 <= 1 | x2 <= -12 | x3 >= -2 | x3 <= 0'

    # Test __str__
    assert str(prop) == '[x1 >= -0.250, x1 <= 0.500, x2 <= -11.500, x3 >= -2.115, x3 <= -0.200]'