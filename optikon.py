"""Module optikon for finding optimal conjunction of propositions about numerical data.

(c) Mario Boley
"""

import numpy as np

from numba import njit
from numba.experimental import jitclass
from numba.types import int64, float64

### Utility ###
###############

@njit
def compute_bounds(x):
    """
    Compute per-variable bounds over a dataset.

    Args:
        x (ndarray): Data matrix of shape (n, d), where each row is a sample 
        and each column is a variable.

    Returns:
        Tuple[ndarray, ndarray]: A pair (l, u) of arrays, each of shape (d,), where 
        l[j] is the minimum and u[j] is the maximum of variable j over all n samples.

    Notes:
        If the input has zero rows (n == 0), the returned bounds are (inf, -inf) 
        for each variable. This convention ensures that all propositions are 
        treated as trivially satisfied on the empty domain.

    Examples:
        >>> x = np.array([[1.0, -1.0], [0.0, 0.0]])
        >>> compute_bounds(x)
        (array([ 0., -1.]), array([1., 0.]))

        >>> x = np.empty((0, 2))
        >>> compute_bounds(x)
        (array([inf, inf]), array([-inf, -inf]))
    """
    n, d = x.shape
    l = np.full(d, np.inf)
    u = np.full(d, -np.inf)
    for i in range(n):
        for j in range(d): # TODO: benchmark loop inversion with parallelisation
            if x[i, j] < l[j]:
                l[j] = x[i, j]
            if x[i, j] > u[j]:
                u[j] = x[i, j]
    return l, u

compute_bounds.compile("(float64[:, :],)")

##### Propositionaliation #####
###############################

@jitclass
class Propositionalization:
    """
    Represents a fixed propositionalization over a d-dimensional dataset.

    Each of the p propositions represents an inequality s*x_v >= t defined by:
      - a variable index v in {0, ..., d-1},
      - a float64 threshold t,
      - and a sign s in {-1, 1} indicating the direction of comparison.
    """

    v: int64[:]
    t: float64[:]
    s: int64[:]

    def __init__(self, v, t, s):
        self.v = v
        self.t = t
        self.s = s

    def support(self, p, x):
        return np.flatnonzero(self.s[p]*x[:,self.v[p]] >= self.t[p])
    
    def trivial(prop, l, u, subset):
        """
        Identify trivial (tautological) propositions over the given variable bounds.

        Args:
            l (ndarray): Lower bounds for each of the d variables (shape: [d]).
            u (ndarray): Upper bounds for each of the d variables (shape: [d]).
            subset (ndarray): Indices of the propositions to check (shape: [m], values in [0, p)).

        Returns:
            ndarray: Indices in `subset` of propositions that are tautological 
            (i.e., always satisfied given the bounds).

        Examples:
            >>> from opticon import Propositionalization
            >>> import numpy as np
            >>> prop = Propositionalization(np.array([0, 1]), np.array([0.5, -1.0]), np.array([1, -1]))
            >>> l = np.array([0.0, -2.0])
            >>> u = np.array([1.0, 0.0])
            >>> prop.tautologies(l, u, np.array([0, 1]))
            array([1])
        """
        v = prop.v[subset]
        t = prop.t[subset]
        s = prop.s[subset]

        res = np.zeros(len(subset), dtype=np.bool_)

        lower = s == 1
        upper = s == -1

        res[lower] = l[v[lower]] >= t[lower]
        res[upper] = -u[v[upper]] >= t[upper]

        return subset[res] #return np.flatnonzero(res)

    def nontrivial(prop, l, u, subset):
        """
        Identify propositions that are not tautological over the given variable bounds.

        Args:
            l (ndarray): Lower bounds for each of the d variables (shape: [d]).
            u (ndarray): Upper bounds for each of the d variables (shape: [d]).
            subset (ndarray): Indices of the propositions to check (shape: [m], values in {0, ..., p-1}).

        Returns:
            ndarray: Indices in `subset` of propositions that are not tautological
            (i.e., not always satisfied under the given bounds).

        Examples:
            >>> from opticon import Propositionalization
            >>> import numpy as np
            >>> prop = Propositionalization(np.array([0, 1]), np.array([0.5, -1.0]), np.array([1, -1]))
            >>> l = np.array([0.0, -2.0])
            >>> u = np.array([1.0, 0.0])
            >>> nontrivial(prop, l, u, np.array([0, 1]))
            array([0])
        """
        v = prop.v[subset]
        t = prop.t[subset]
        s = prop.s[subset]

        res = np.zeros(len(subset), dtype=np.bool_)

        lower = s == 1
        upper = s == -1

        res[lower] = l[v[lower]] < t[lower]
        res[upper] = -u[v[upper]] < t[upper]

        return subset[res]# np.flatnonzero(res) 
    
    def binarize(self, x):
        """
        Binarizes a dataset based on the propositionalisation.

        Args:
            x (ndarray): Data matrix of shape (n, d), where each row is a sample.

        Returns:
            ndarray: Binary matrix of shape (n, p), where entry (i, j) is 1 if 
            the j-th proposition is satisfied by the i-th sample, and 0 otherwise.
        """        
        return self.s*x[:, self.v] >= self.t
    
    def __len__(self):
        """
        Returns the number of propositions (p) in this propositionalization.

        Returns:
            int: Total number of propositions.

        Examples:
            >>> len(prop)
            2
        """
        return len(self.v)

# @njit
# def str_from_prop(prop, j):
#     return f'x{prop.v[j]+1} {'>=' if prop.s[j]==1 else '<='} {prop.s[j]*prop.t[j]:0.3f}'

    # @njit
    def str_from_prop(prop, j):
        """
        Numba-compatible string construction for proposition j with manual float formatting.
        Output format: "x{v+1} >= int.frac" or "x{v+1} <= int.frac"
        """
        v_idx = prop.v[j] + 1
        sign = '>=' if prop.s[j] == 1 else '<='
        value = prop.s[j] * prop.t[j]

        int_part = int(value)
        frac_part = int((abs(value) - abs(int_part)) * 1000 + 0.5)

        int_str = str(int_part)
        frac_str = str(frac_part).rjust(3, '0')

        return 'x' + str(v_idx) + ' ' + sign + ' ' + int_str + '.' + frac_str

    # @njit
    def str_from_conj(prop, q):
        result = ''
        for i in range(len(q)):
            if i > 0:
                result += ' & '
            result += prop.str_from_prop(q[i])
        return result

def full_propositionalization(x):
    """
    Constructs propositionalization with all non-trivial threshold propositions from x in
    lexicographic order by (v, t, s).
    """
    n, d = x.shape
    max_props = 2 * n * d
    v_out = np.empty(max_props, dtype=np.int64)
    t_out = np.empty(max_props, dtype=np.float64)
    s_out = np.empty(max_props, dtype=np.int64)
    count = 0

    for v in range(d):
        thresholds = np.unique(x[:, v])
        # lower bounds strictest to weakest, exluding trivial 
        for t in thresholds[n-1:0:-1]:  
            v_out[count] = v
            t_out[count] = t
            s_out[count] = 1
            count += 1
        # upper bounds: strictest to weeakest, excluding trivial
        for t in thresholds[:-1]:  
            v_out[count] = v
            t_out[count] = -t
            s_out[count] = -1
            count += 1

    return Propositionalization(v_out[:count], t_out[:count], s_out[:count])

def equal_frequency_propositionalization(x, k=None):
    n, d = x.shape
    k = k if k is not None else 2*np.ceil(n**(1/3)).astype(int)
    quantile_targets = np.linspace(0, 1, k + 1)[1:-1]

    quantiles = np.quantile(x, quantile_targets, axis=0)  # shape (n_splitpoints, n_cols)
    v = np.repeat(np.arange(d), quantiles.shape[0])
    t = quantiles.flatten()

    keep = np.empty_like(v, dtype=bool)
    keep[0] = True
    keep[1:] = (v[1:] != v[:-1]) | (t[1:] != t[:-1])
    v, t = v[keep], t[keep]

    s = np.repeat([1, -1], len(v))
    return Propositionalization(np.concatenate((v, v)), np.concatenate((t, -t)), s)

def equal_width_propositionalization(x):
    return equal_width_propositionalization_sorted(np.sort(x, axis=0))

@njit
def equal_width_propositionalization_sorted(x_sorted):
    n, d = x_sorted.shape

    max_possible = 2 * d * n
    v = np.empty(max_possible, dtype=np.int64)
    t = np.empty(max_possible, dtype=np.float64)
    s = np.empty(max_possible, dtype=np.int64)
    idx = 0

    for j in range(d):
        col_data = x_sorted[:, j]
        l_j = col_data[0]
        u_j = col_data[-1]

        if u_j == l_j:
            continue

        q25 = col_data[int(0.25 * (n-1))]
        q75 = col_data[int(0.75 * (n-1))]
        iqr = q75 - q25

        width = 2 * iqr / max(1, n**(1/3))
        if width == 0:
            continue

        n_bins = int(np.ceil((u_j - l_j) / width))
        if n_bins <= 1:
            continue

        edges = l_j + width * np.arange(1, n_bins)

        positions = np.searchsorted(col_data, edges, side='left')
        positions_ext = np.empty(len(positions) + 1, dtype=np.int64)
        positions_ext[0] = 0
        positions_ext[1:] = positions
        diffs = np.diff(positions_ext)
        nontrivial = diffs > 0

        for k in range(len(edges)):
            if nontrivial[k]:
                # upper bound first (s=1, decreasing thresholds)
                v[idx] = j
                t[idx] = edges[len(edges) - 1 - k]
                s[idx] = 1
                idx += 1

        for k in range(len(edges)):
            if nontrivial[k]:
                # lower bound second (s=-1, increasing thresholds)
                v[idx] = j
                t[idx] = -edges[k]
                s[idx] = -1
                idx += 1

    return Propositionalization(v[:idx], t[:idx], s[:idx])

equal_width_propositionalization_sorted.compile("(float64[:, :],)")


if __name__=='__main__':
    import doctest
    doctest.testmod()
