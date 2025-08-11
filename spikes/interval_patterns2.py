"""Prototypical implementation of interval pattern enumeration with fast canonicity check through
loop inversion, i.e., by pre-computing narrowing feasible restriction ranges by iterating over the prefix
variables instead of checking prefix preservation for each candidate.

This implementation uses a dummy objective function (number of non-trivial condition), which should lead
to complete enumeration, because a trivial bounding function is used.

Author: Mario Boley
Date  : 2025-08-10

Revision 1: 2025-08-11
using prefix preserving indices for simplification
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from optikon import Propositionalization
from testdata import diblock_mvn_sample, SMALL_1
import heapq

from numba import njit
from numba.experimental import jitclass
from numba.types import int64, float64
import numpy as np

@njit
def range_preserving_suffix_and_prefix(x):
    """
    Computes smallest suffix and prefix of x that preserve range (min and max 
    values are retained).

    Args:
        x (ndarray): non-empty array of shape (m,).

    Returns:
        Tuple[int, int]: (i, j) such that i is the largest index such that all occurrences
        of both x.min() and x.max() are still present in x[i:], and j is the the smallest
        index j such that all occurrences are still present in x[:j+1].

    Note:
        The function runs in O(m)

    Examples:
        >>> range_preserving_suffix_and_prefix(np.array([1., 2., 3., 4., 5.]))
        (0, 4)
        >>> range_preserving_suffix_and_prefix(np.array([2., 2., 2., 2., 2.]))
        (4, 0)
        >>> range_preserving_suffix_and_prefix(np.array([2., 1., 3., 4., 5.]))
        (1, 4)
        >>> range_preserving_suffix_and_prefix(np.array([4., 2., 2., 3., 4.]))
        (2, 1)
    """
    m = x.shape[0]
    z_min = x.min()
    z_max = x.max()
    count_z_min = (x == z_min).sum()
    count_z_max = (x == z_max).sum()

    z_min_remaining = count_z_min
    z_max_remaining = count_z_max
    i = 0
    while i < m and z_min_remaining > 0 and z_max_remaining > 0:
        if x[i] == z_min:
            z_min_remaining -= 1
        if x[i] == z_max:
            z_max_remaining -= 1
        i += 1

    z_min_remaining = count_z_min
    z_max_remaining = count_z_max
    j = m - 1
    while j >= 0 and z_min_remaining > 0 and z_max_remaining > 0:
        if x[j] == z_min:
            z_min_remaining -= 1
        if x[j] == z_max:
            z_max_remaining -= 1
        j -= 1

    return i - 1, j + 1

@njit
def prefix_preserving_threshold_bounds(x, orders):
    """
    Computes prefix preserving upper and lower bound for thresholding each variable of a dataset. 

    Specifically, for each variable k, this function computes:
      - the largest threshold l for restricting the dataset via x[:, k] >= l and
      - the smallest threshold u for restricting the dataset via x[:, k] <= u
    such that those restrictions do not reduce the value range of all variables j < k.

    Args:
        x (ndarray): A dataset of shape (m, d).
        orders (ndarray): An array of shape (m, d), where each column contains the 
            indices that would sort x[:, k] in ascending order.

    Returns:
        Tuple[ndarray, ndarray]: Two arrays of shape (d,), where the first contains 
        the maximal prefix-preserving lower-bound thresholds, and the second contains 
        the minimal prefix-preserving upper-bound thresholds.

    Notes:
        The function runs in time O(d^2 m)

    Examples:
        >>> import numpy as np
        >>> x = np.array([[0.1, 1.0, -0.5], 
        ...               [0.3, 2.0, -1.0],
        ...               [0.2, 0.5, 0.0]])
        >>> orders = np.argsort(x, axis=0)
        >>> l, u = prefix_preserving_threshold_bounds(x, orders)
        >>> np.round(l, 2)
        array([inf,  1., -1.])
        >>> np.round(u, 2)
        array([-inf,   2.,   0.])

        >>> x_empty = np.empty((0, 3))
        >>> orders_empty = np.empty((0, 3), dtype=np.int64)
        >>> l, u = prefix_preserving_threshold_bounds(x_empty, orders_empty)
        >>> np.all(np.isposinf(l)) and np.all(np.isneginf(u))
        True
    """
    _, d = x.shape
    max_pp_lb_thresholds = np.full(d, np.inf)
    min_pp_ub_thresholds = np.full(d, -np.inf)

    for k in range(d):
        for j in range(k):
            l, u = range_preserving_suffix_and_prefix(x[orders[:, k], j])
            if x[orders[l, k], k] < max_pp_lb_thresholds[k]:
                max_pp_lb_thresholds[k] = x[orders[l, k], k]
            if x[orders[u, k], k] > min_pp_ub_thresholds[k]:
                min_pp_ub_thresholds[k] = x[orders[u, k], k]

    return max_pp_lb_thresholds, min_pp_ub_thresholds

@njit
def prefix_preserving_index_bounds(x, orders):
    """
    Computes prefix preserving upper and lower bound for thresholding each variable of a dataset. 

    Specifically, for each variable k, this function computes:
      - the largest threshold l for restricting the dataset via x[:, k] >= l and
      - the smallest threshold u for restricting the dataset via x[:, k] <= u
    such that those restrictions do not reduce the value range of all variables j < k.

    Args:
        x (ndarray): A dataset of shape (m, d).
        orders (ndarray): An array of shape (m, d), where each column contains the 
            indices that would sort x[:, k] in ascending order.

    Returns:
        Tuple[ndarray, ndarray]: Two arrays of shape (d,), where the first contains 
        the maximal prefix-preserving lower-bound thresholds, and the second contains 
        the minimal prefix-preserving upper-bound thresholds.

    Notes:
        The function runs in time O(d^2 m)

    Examples:
        >>> import numpy as np
        >>> x = np.array([[0.1, 1.0, -0.5], 
        ...               [0.3, 2.0, -1.0],
        ...               [0.2, 0.5, 0.0]])
        >>> orders = np.argsort(x, axis=0)
        >>> l, u = prefix_preserving_index_bounds(x, orders)
        >>> np.round(l, 2)
        array([3, 1, 0])
        >>> np.round(u, 2)
        array([-1,  2,  2])
    """
    n, d = x.shape
    max_pp_lb_indices = np.full(d, n-1, dtype=np.int64)
    min_pp_ub_indices = np.full(d, 0, dtype=np.int64)

    for k in range(d):
        for j in range(k):
            l, u = range_preserving_suffix_and_prefix(x[orders[:, k], j])
            if l < max_pp_lb_indices[k]:
                max_pp_lb_indices[k] = l
            if u > min_pp_ub_indices[k]:
                min_pp_ub_indices[k] = u

    return max_pp_lb_indices, min_pp_ub_indices

@jitclass
class IntervalPatternSearchNode:
    l: float64[:]
    u: float64[:]
    support: int64[:]
    min_active_j: int64

    def __init__(self, l, u, support, min_active_j):
        self.l = l
        self.u = u
        self.support = support
        self.min_active_j = min_active_j

    def num_non_trivial_bounds(self):
        res = 0
        for j in range(len(self.l)):
            if self.l[j] > -np.inf:
                res += 1
            if self.u[j] < np.inf:
                res += 1
        return res

    def to_propositionalization(self):
        k = self.num_non_trivial_bounds()
        v = np.zeros(k, dtype=np.int64)
        t = np.zeros(k, dtype=np.float64)
        s = np.zeros(k, dtype=np.int64)

        r = 0        
        for j in range(len(self.l)):
            if self.l[j] > -np.inf:
                v[r] = j
                t[r] = self.l[j]
                s[r] = 1
                r += 1
            if self.u[j] < np.inf:
                v[r] = j
                t[r] = -self.u[j]
                s[r] = -1
                r += 1
        return Propositionalization(v, t, s)


# FastCanonicalTreeSearchNodeType = FastCanonicalTreeSearchNode


class FastCanonicalTreeSearch:

    def __init__(self, x):
        self.x = x
        self.n, self.d = x.shape
        self.sups_to_keys = {}

    def refinement(self, node):
        if len(node.support) == 0:
            print('warning: zero support dequeued', flush=True)

        x_sub = self.x[node.support]
        sub_orders = np.argsort(x_sub, axis=0)

        max_pp_lb, min_pp_ub = prefix_preserving_index_bounds(x_sub, sub_orders)

        res = []

        for j in range(node.min_active_j, self.d):

            # should we create view: vals = x_sub[sub_orders[:, j], j]
            # or would this be detremental for performance?

            col = x_sub[:, j]
            order = sub_orders[:, j]

            # if np.array_equal(node.l , np.array([1., -np.inf, -np.inf])) and np.array_equal(node.u, np.array([3., np.inf, np.inf])):
            #     print('refinement generation of x1 in [1, 3]:')
            #     print('col', j)
            #     print('max_pp_lb', max_pp_lb[j])
            #     print('min_pp_ub', min_pp_ub[j])

            if np.isneginf(node.l[j]):
                
                # create all canonical nodes from lower bounds
                for i in range(1, max_pp_lb[j]+1):
                    t = col[order[i]] 
                    if t > col[order[i-1]]:
                        _l = node.l.copy()
                        _l[j] = t
                        _sup = node.support[np.flatnonzero(col >= t)]
                        # probably cheaper but changes order: _sup = node.support[order[i:]]
                        res.append(IntervalPatternSearchNode(_l, node.u, _sup, j))


            # create all canonical nodes from upper bounds
            for i in range(len(node.support)-2, min_pp_ub[j]-1, -1):
                t = col[order[i]] 
                if t < col[order[i+1]]:
                    _u = node.u.copy()
                    _u[j] = t
                    _sup = node.support[np.flatnonzero(col <= t)]
                    # probably cheaper but changes order: _sup = node.support[order[:i+1]]
                    res.append(IntervalPatternSearchNode(node.l, _u, _sup, j+1))

        return res


    def make_root(self):
        n, d = self.x.shape
        l, u = np.full(d, -np.inf), np.full(d, np.inf)
        return IntervalPatternSearchNode(l, u, np.arange(n), 0)

    def run(self, max_depth=8):
        heap = []
        nodes = []
        freelist = []

        root = self.make_root()
        nodes.append(root)
        heapq.heappush(heap, (-2*len(root.l), 0))

        best_value = 0
        best_node = root
        created = 1
        # non_canonical = 0

        while heap:
            neg_bound, idx = heapq.heappop(heap)
            node = nodes[idx]
            freelist.append(idx)

            # print(node.l, node.u, node.support, node.min_active_j)
            
            if tuple(node.support) in self.sups_to_keys:
                print('sup', node.support, 'enumerated repeatedly')
                old_l, old_u = self.sups_to_keys[tuple(node.support)]
                print('first from', (old_l, old_u)) #, '(', IntervalPatternSearchNode(old_l, old_u, node.support).to_propositionalization().as_conj_str() ,')')
                print('then again from', (node.l, node.u)) #, '(', node.to_propositionalization().as_conj_str() ,')')
                # break
            self.sups_to_keys[tuple(node.support)]=(node.l, node.u)

            if -neg_bound < best_value:
                continue
            if node.num_non_trivial_bounds() >= max_depth:
                continue

            children = self.refinement(node)        

            # non_canonical += len(node.remaining) - len(children)
            created += len(children)

            for child in children:

                val = child.num_non_trivial_bounds()
                bnd = 2*len(child.l)
                
                if val > best_value:
                    best_value = val
                    best_node = child

                if len(freelist) > 0:
                    reuse_idx = freelist.pop()
                    nodes[reuse_idx] = child
                    heapq.heappush(heap, (-bnd, reuse_idx))
                else:
                    nodes.append(child)
                    heapq.heappush(heap, (-bnd, len(nodes) - 1))

        print("Best", best_node.l, best_node.u)
        print("Best value:", best_value)
        print("Total nodes created:", created)
        # print("None canonical edges:", non_canonical)
        return best_node.to_propositionalization(), best_value

# import sys
# with open('new_supports2.txt', 'w') as f:
#     sys.stdout = f
#     x = mvn_with_correlation(100, seed=0)
#     prop = full_propositionalization(x) # equal_width_propositionalization(x)
#     fast_search = FastCanonicalTreeSearch(x, prop)
#     fast_search.run(2)

if __name__=='__main__':
    import doctest
    doctest.testmod()

    x = diblock_mvn_sample(100, seed=0)
    fast_search = FastCanonicalTreeSearch(x)
    best, val = fast_search.run(3)
    print(best.as_conj_str())

    from testdata import SMALL_1
    x2 = SMALL_1.x
    fast_search2 = FastCanonicalTreeSearch(x2)
    best2, val2 = fast_search2.run()
    print(len(SMALL_1.selectable_sups))
    print(best2.as_conj_str())
    print(x2)
