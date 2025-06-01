import numpy as np
import numba as nb
from optikon import compute_bounds, make_maxheap_class

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