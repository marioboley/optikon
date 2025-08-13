"""Prototypical implementation of interval pattern enumeration with fast canonicity check through
loop inversion, i.e., by pre-computing narrowing feasible restriction ranges by iterating over the prefix
variables instead of checking prefix preservation for each candidate.

This implementation uses a dummy objective function (number of non-trivial condition), which should lead
to complete enumeration, because a trivial bounding function is used.

Author: Mario Boley
Date  : 2025-08-10

Revision 1: 2025-08-11
using prefix preserving indices for simplification

Revision 2: 2025-08-11
use weighted support instead of dummy objective

Revision 3: 2025-08-12
numba compatible version
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from optikon import IntervalPatternSearchNode, IntervallPatternNodeHeap, make_interval_search_root, argsort_columns
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
        Tuple[int, int]: (i, j) such that i is the largest index such that at least on
        occurrences of each x.min() and x.max() are still present in x[i:], and j is
        the smallest index j such that at least one of those occurrences are
        still present in x[:j+1].

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
def prefix_preserving_index_bounds(x, orders, min_k=0):
    """
    Computes longest non-empty prefix-preserving index ranges for each variable in a dataset.

    Specifically, for each variable k >= min_k, this function computes:
      - the largest index l for restricting the dataset via x[:, k] >= x[orders[l, k]],
        or equivalently to x[orders[l:, k]], and
      - the smallest index u for restricting the dataset via x[:, k] <= x[orders[u, k]]
        or equivalently to x[orders[:u+1, k]]
    such that those restrictions do not reduce the value range of all variables j < k.

    Args:
        x (ndarray): A dataset of shape (m, d).
        orders (ndarray): An array of shape (m, d), where each column contains the 
            indices that would sort x[:, k] in ascending order.
        min_k (int): smallest index for which to compute value ranges

    Returns:
        Tuple[ndarray, ndarray]: Two arrays of shape (d,), where the first contains 
        the maximal prefix-preserving lower-bound indices, and the second contains 
        the minimal prefix-preserving upper-bound indices. Arrays are padded with 
        default values n-1 and 0 for max lower and min upper bounds indices, respectively,

    Notes:
        - The function runs in time O((d-min_k)^2 m) <= O(d^2 m)
        - Default values for k < min_k are n-1 and 0

    Examples:
        >>> import numpy as np
        >>> x = np.array([[0.1, 1.0, -0.5], 
        ...               [0.3, 2.0, -1.0],
        ...               [0.2, 0.5, 0.0]])
        >>> orders = np.argsort(x, axis=0)
        >>> l, u = prefix_preserving_index_bounds(x, orders)
        >>> np.round(l, 2)
        array([2, 1, 0])
        >>> np.round(u, 2)
        array([0, 2, 2])
    """
    n, d = x.shape
    max_pp_lb_indices = np.full(d, n-1, dtype=np.int64)
    min_pp_ub_indices = np.full(d, 0, dtype=np.int64)

    for k in range(min_k, d):
        for j in range(k):
            l, u = range_preserving_suffix_and_prefix(x[orders[:, k], j])
            if l < max_pp_lb_indices[k]:
                max_pp_lb_indices[k] = l
            if u > min_pp_ub_indices[k]:
                min_pp_ub_indices[k] = u

    return max_pp_lb_indices, min_pp_ub_indices


@njit
def max_weighted_support_fips(x, w, max_depth):
    n, d = x.shape
    heap = IntervallPatternNodeHeap()
    
    root = make_interval_search_root(x, w)
    root_bound = w[root.pos_support].sum()
    root_value = w.sum()
    heap.push(root_bound, root)

    best_value = root_value
    best_node = root
    created = 1

    while heap:
        bound, node = heap.pop()

        # print(node.l, node.u, node.support, node.min_active_j)

        if len(node.support) == 0:
            print('warning: zero support dequeued')
            continue

        if bound < best_value:
            continue
        if node.num_non_trivial_bounds() >= max_depth:
            continue

        x_sub = x[node.support]
        sub_orders = argsort_columns(x_sub) # np.argsort(x_sub, axis=0)
        x_sub_pos = x[node.pos_support]
        w_sub = w[node.support]
        w_sub_pos = w[node.pos_support]
        
        max_pp_lb, min_pp_ub = prefix_preserving_index_bounds(x_sub, sub_orders, node.min_active_j)

        for j in range(node.min_active_j, d):

            # should we create view: vals = x_sub[sub_orders[:, j], j]
            # or would this be detremental for performance?

            col_data = x_sub[:, j]
            order = sub_orders[:, j]
            pos_col_data = x_sub_pos[:, j]

            if np.isneginf(node.l[j]):
                
                # create all canonical nodes from lower bounds

                sum_w = w_sub.sum()
                sum_pos_w = w_sub_pos.sum()

                for i in range(1, max_pp_lb[j]+1):
                    t = col_data[order[i]]
                    w_rem = w_sub[order[i-1]]
                    sum_w -= w_rem
                    if w_rem > 0:
                        sum_pos_w -= w_rem

                    if t > col_data[order[i-1]]:
                        _l = node.l.copy()
                        _l[j] = t
                        _sup = node.support[np.flatnonzero(col_data >= t)]
                        _pos_sup = node.pos_support[np.flatnonzero(pos_col_data >= t)]
                        # probably cheaper but changes order: _sup = node.support[order[i:]]
                        
                        child = IntervalPatternSearchNode(_l, node.u, _sup, _pos_sup, j)
                        if sum_w > best_value:
                            best_value = sum_w
                            best_node = child
                        if sum_pos_w > best_value:
                            heap.push(sum_pos_w, child)
                        else:
                            break
                        
            # create all canonical nodes from upper bounds
            sum_w = w_sub.sum()
            sum_pos_w = w_sub_pos.sum()
            for i in range(len(node.support)-2, min_pp_ub[j]-1, -1):
                t = col_data[order[i]] 
                w_rem = w_sub[order[i+1]]
                sum_w -= w_rem
                if w_rem > 0:
                    sum_pos_w -= w_rem
                if t < col_data[order[i+1]]:
                    _u = node.u.copy()
                    _u[j] = t
                    _sup = node.support[np.flatnonzero(col_data <= t)]
                    _pos_sup = node.pos_support[np.flatnonzero(pos_col_data <= t)]
                    # probably cheaper but changes order: _sup = node.support[order[:i+1]]
                    child = IntervalPatternSearchNode(node.l, _u, _sup, _pos_sup, j+1)
                    if sum_w > best_value:
                        best_value = sum_w
                        best_node = child
                    if sum_pos_w > best_value:
                        heap.push(sum_pos_w, child)
                    else:
                        break

    print("Best", best_node.l, best_node.u)
    print("Best value:", best_value)
    print("Total nodes created:", created)
    # print("None canonical edges:", non_canonical)
    return best_node.to_propositionalization(), best_value

class FastIntervalPatternSearch:

    def __init__(self, x, w):
        self.x = x
        self.w = w
        self.n, self.d = x.shape
        self.heap = []
        self.nodes = []
        self.freelist = []
        self.created = 0

    def make_root(self):
        n, d = self.x.shape
        l, u = np.full(d, -np.inf), np.full(d, np.inf)
        return IntervalPatternSearchNode(l, u, np.arange(n), np.flatnonzero(self.w > 0), 0)

    def push(self, node, bnd):
        self.created += 1
        if len(self.freelist) > 0:
            reuse_idx = self.freelist.pop()
            self.nodes[reuse_idx] = node
            heapq.heappush(self.heap, (-bnd, reuse_idx))
        else:
            self.nodes.append(node)
            heapq.heappush(self.heap, (-bnd, len(self.nodes) - 1))

    def run(self, max_depth=8):
        # root = self.make_root()
        # self.nodes.append(root)
        # heapq.heappush(self.heap, (-2*len(root.l), 0))
        root = self.make_root()
        self.push(root, self.w[root.pos_support].sum())

        best_value = w.sum()
        best_node = root
        self.created = 1
        # non_canonical = 0

        while self.heap:
            neg_bound, idx = heapq.heappop(self.heap)
            node = self.nodes[idx]
            self.freelist.append(idx)

            # print(node.l, node.u, node.support, node.min_active_j)

            if len(node.support) == 0:
                print('warning: zero support dequeued', flush=True)
                continue

            if -neg_bound < best_value:
                continue
            if node.num_non_trivial_bounds() >= max_depth:
                continue

            x_sub = self.x[node.support]
            sub_orders = np.argsort(x_sub, axis=0)
            x_sub_pos = self.x[node.pos_support]
            w_sub = self.w[node.support]
            w_sub_pos = self.w[node.pos_support]
            
            max_pp_lb, min_pp_ub = prefix_preserving_index_bounds(x_sub, sub_orders, node.min_active_j)

            for j in range(node.min_active_j, self.d):

                # should we create view: vals = x_sub[sub_orders[:, j], j]
                # or would this be detremental for performance?

                col_data = x_sub[:, j]
                order = sub_orders[:, j]
                pos_col_data = x_sub_pos[:, j]

                if np.isneginf(node.l[j]):
                    
                    # create all canonical nodes from lower bounds

                    sum_w = w_sub.sum()
                    sum_pos_w = w_sub_pos.sum()

                    for i in range(1, max_pp_lb[j]+1):
                        t = col_data[order[i]]
                        w_rem = w_sub[order[i-1]]
                        sum_w -= w_rem
                        if w_rem > 0:
                            sum_pos_w -= w_rem

                        if t > col_data[order[i-1]]:
                            _l = node.l.copy()
                            _l[j] = t
                            _sup = node.support[np.flatnonzero(col_data >= t)]
                            _pos_sup = node.pos_support[np.flatnonzero(pos_col_data >= t)]
                            # probably cheaper but changes order: _sup = node.support[order[i:]]
                            
                            child = IntervalPatternSearchNode(_l, node.u, _sup, _pos_sup, j)
                            if sum_w > best_value:
                                best_value = sum_w
                                best_node = child
                            if sum_pos_w > best_value:
                                self.push(child, sum_pos_w)
                            else:
                                break
                            
                # create all canonical nodes from upper bounds
                sum_w = w_sub.sum()
                sum_pos_w = w_sub_pos.sum()
                for i in range(len(node.support)-2, min_pp_ub[j]-1, -1):
                    t = col_data[order[i]] 
                    w_rem = w_sub[order[i+1]]
                    sum_w -= w_rem
                    if w_rem > 0:
                        sum_pos_w -= w_rem
                    if t < col_data[order[i+1]]:
                        _u = node.u.copy()
                        _u[j] = t
                        _sup = node.support[np.flatnonzero(col_data <= t)]
                        _pos_sup = node.pos_support[np.flatnonzero(pos_col_data <= t)]
                        # probably cheaper but changes order: _sup = node.support[order[:i+1]]
                        child = IntervalPatternSearchNode(node.l, _u, _sup, _pos_sup, j+1)
                        if sum_w > best_value:
                            best_value = sum_w
                            best_node = child
                        if sum_pos_w > best_value:
                            self.push(child, sum_pos_w)
                        else:
                            break

        print("Best", best_node.l, best_node.u)
        print("Best value:", best_value)
        print("Total nodes created:", self.created)
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

    # n = 12
    n = 100
    w = np.random.default_rng(seed=0).normal(size=n)
    x = diblock_mvn_sample(n, seed=0)
    # x = np.round(x, 3)
    # w = np.round(w, 3)
    # fast_search = FastIntervalPatternSearch(x, w)
    # best, val = fast_search.run(2)

    best, val = max_weighted_support_fips(x, w, 2)
    print(best.as_conj_str())
    print(w[best.support_all(x)].sum())

    # from testdata import SMALL_1
    # x2 = SMALL_1.x
    # fast_search2 = FastIntervalPatternSearch(x2, SMALL_1.y)
    # best2, val2 = fast_search2.run()
    # print(SMALL_1.opt_weighted_support)
    # print(best2.as_conj_str())
    # print(x2)

    from optikon import max_weighted_support_bb, full_propositionalization
    props = full_propositionalization(x)
    best_control, val_control, stats = max_weighted_support_bb(x, w, props, 2)
    print(best_control.as_conj_str())
    print(val_control)
    print(stats['nodes_created'])
    # print(props.as_str())

    # print(x)
    # print(w)

    # x_orders = np.argsort(x, axis=0)
    # print(x_orders[:, 2])
    # print(x[x_orders[:, 2], 2])
    # print(x[x_orders[:, 2], 0], min(x[:, 0]), max(x[:, 0]))

