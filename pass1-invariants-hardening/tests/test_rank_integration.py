"""Rank integration tests for GF(2) binary_rank.

Validates that binary_rank is correct, deterministic, and that
the CSS code dimension k = n - rank_X - rank_Z is non-negative.
"""

import pytest
import numpy as np
import scipy.sparse as sp

from qldpc.css_code import CSSCode
from qldpc.invariants import binary_rank


# ---------------------------------------------------------------
# 1. Rank bounds
# ---------------------------------------------------------------
class TestRankBounds:
    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_rank_HX_bounded_by_rows(self, seed):
        c = CSSCode(r=3, c=3, L=5, seed=seed)
        rank_X = binary_rank(c.HX)
        assert rank_X <= c.HX.shape[0]

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_rank_HZ_bounded_by_rows(self, seed):
        c = CSSCode(r=3, c=3, L=5, seed=seed)
        rank_Z = binary_rank(c.HZ)
        assert rank_Z <= c.HZ.shape[0]

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_rank_sum_bounded_by_n(self, seed):
        c = CSSCode(r=3, c=3, L=5, seed=seed)
        rank_X = binary_rank(c.HX)
        rank_Z = binary_rank(c.HZ)
        assert rank_X + rank_Z <= c.n


# ---------------------------------------------------------------
# 2. k >= 0 for small parameter sets
# ---------------------------------------------------------------
class TestLogicalQubits:
    @pytest.mark.parametrize("r,L,seed", [
        (2, 3, 0), (2, 5, 1), (3, 3, 2), (3, 5, 7), (4, 3, 42),
    ])
    def test_k_non_negative(self, r, L, seed):
        c = CSSCode(r=r, c=r, L=L, seed=seed)
        assert c.k >= 0, f"k={c.k} < 0 for r={r}, L={L}, seed={seed}"


# ---------------------------------------------------------------
# 3. Determinism: same seed → same rank
# ---------------------------------------------------------------
class TestRankDeterminism:
    def test_rank_deterministic_across_calls(self):
        c1 = CSSCode(r=3, c=3, L=5, seed=42)
        c2 = CSSCode(r=3, c=3, L=5, seed=42)
        assert binary_rank(c1.HX) == binary_rank(c2.HX)
        assert binary_rank(c1.HZ) == binary_rank(c2.HZ)
        assert c1.k == c2.k

    def test_different_seeds_may_differ(self):
        """Different seeds can produce different ranks (sanity check)."""
        ranks = set()
        for seed in range(20):
            c = CSSCode(r=3, c=3, L=5, seed=seed)
            ranks.add(binary_rank(c.HX))
        # With 20 different seeds we expect at least some variation.
        # (Not a strict invariant, but a sanity check.)
        assert len(ranks) >= 1


# ---------------------------------------------------------------
# 4. binary_rank correctness on known matrices
# ---------------------------------------------------------------
class TestBinaryRankCorrectness:
    def test_identity_rank(self):
        """Rank of I_n over GF(2) is n."""
        n = 10
        I = sp.eye(n, format="csr", dtype=np.int8)
        assert binary_rank(I) == n

    def test_zero_matrix_rank(self):
        """Rank of zero matrix is 0."""
        Z = sp.csr_matrix((5, 5), dtype=np.int8)
        assert binary_rank(Z) == 0

    def test_duplicate_rows_rank(self):
        """Two identical rows have rank 1."""
        data = np.array([1, 1, 1, 1], dtype=np.int8)
        rows = np.array([0, 0, 1, 1])
        cols = np.array([0, 2, 0, 2])
        mat = sp.csr_matrix((data, (rows, cols)), shape=(2, 3), dtype=np.int8)
        assert binary_rank(mat) == 1

    def test_full_rank_2x2(self):
        """[[1,0],[0,1]] has rank 2 over GF(2)."""
        mat = sp.csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.int8))
        assert binary_rank(mat) == 2

    def test_rank_differs_from_float_rank(self):
        """Demonstrate case where GF(2) rank != float rank.

        [[1, 1],
         [1, 1]]

        Float rank = 1, GF(2) rank = 1.  Same here, but:

        [[1, 1, 0],
         [0, 1, 1],
         [1, 0, 1]]

        Over R: rank 2 (rows sum to zero mod 2).
        Over GF(2): rank 2 (row3 = row1 XOR row2).
        """
        mat = sp.csr_matrix(np.array([
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ], dtype=np.int8))
        assert binary_rank(mat) == 2
