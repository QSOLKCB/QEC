"""CSS code construction via shared-circulant lifted protograph.

Orthogonality guarantee
-----------------------
We build H_X and H_Z from *base matrices* B_X and B_Z that satisfy

    B_X  @  B_Z^T  ==  0   (mod 2)          … (★)

at the protograph level.  When every 1-entry is replaced by the
**same** circulant permutation matrix (from the shared lift table),
the lifted matrices inherit the orthogonality:

    H_X  @  H_Z^T  ==  0   (mod 2)

Proof sketch:  each (i,k) block of the product is
    Σ_j  P^{s_{i,j}} · (P^{s_{k,j}})^T
where the sum runs over columns j.  Because we use the **same**
shift s for both H_X and H_Z on column j, each term is
P^{s_{i,j} - s_{k,j}}, and the block-sum equals (B_X @ B_Z^T)_{i,k}
copies of identity-like circulant blocks.  If the base product
entry is 0 mod 2 the block vanishes.

Default ensemble
----------------
We use the *hypergraph-product* (HGP) / *bicycle* family:

    B_X = [ A | B ]      B_Z = [ B^T | A^T ]

where A and B are random binary matrices of matching dimension.

    B_X @ B_Z^T  =  A B^T + B A^T        (mod 2)

For any matrices over GF(2),  A B^T + B A^T  ==  0  (mod 2)
because  (AB^T)^T = BA^T  and  X + X == 0  in GF(2).

So orthogonality is **algebraically guaranteed** — no repair needed.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .lift import SharedLiftTable, lift_matrix
from .invariants import (
    ConstructionInvariantError,
    binary_rank,
    check_css_orthogonality,
    check_no_zero_rows_or_cols,
)


def _circulant_matrix(first_row: np.ndarray) -> np.ndarray:
    """Build a circulant matrix from its first row.

    Row i is ``np.roll(first_row, i)``.  Circulant matrices commute
    under multiplication, which is essential for the bicycle
    orthogonality guarantee.
    """
    r = len(first_row)
    return np.array([np.roll(first_row, i) for i in range(r)], dtype=np.int8)


class CSSCode:
    """Lifted CSS code from a bicycle protograph.

    The bicycle construction uses circulant seed matrices A and B.
    Because circulant matrices commute (AB = BA), the base-matrix
    orthogonality holds algebraically:

        B_X @ B_Z^T = AB + BA = 2AB ≡ 0  (mod 2)

    Parameters
    ----------
    r, c : int
        Dimensions of the seed matrices A (r×c) and B (r×c).
        Must satisfy r == c for the bicycle construction.
    L : int
        Circulant lift size.
    seed : int
        Deterministic RNG seed.
    """

    def __init__(self, r: int, c: int, L: int, seed: int) -> None:
        self.r = r
        self.c = c
        self.L = L
        self.seed = seed

        if r != c:
            raise ValueError(
                "This implementation requires r == c for the bicycle construction."
            )

        rng = np.random.RandomState(seed)

        # --- Step 1: sample circulant seed matrices A, B ---
        # Each circulant is defined by its first row.  We ensure both
        # rows are non-zero so that no block is all-zero.
        row_a = rng.randint(0, 2, size=r).astype(np.int8)
        if not row_a.any():
            row_a[rng.randint(0, r)] = 1
        row_b = rng.randint(0, 2, size=r).astype(np.int8)
        if not row_b.any():
            row_b[rng.randint(0, r)] = 1

        A = _circulant_matrix(row_a)
        B = _circulant_matrix(row_b)

        # --- Step 2: build protograph base matrices ---
        # B_X = [ A | B ]     (r, 2r)
        # B_Z = [ B^T | A^T ] (r, 2r)
        #
        # Orthogonality: B_X @ B_Z^T = AB + BA.
        # Since A, B are circulant → AB = BA → AB + BA = 2AB ≡ 0 (mod 2).
        self.base_X = np.hstack([A, B]).astype(np.int8)
        self.base_Z = np.hstack([B.T, A.T]).astype(np.int8)

        # Verify base orthogonality (algebraic guarantee).
        base_product = (self.base_X @ self.base_Z.T) % 2
        if np.any(base_product):
            raise ConstructionInvariantError(
                "Base-matrix orthogonality violated — this should be "
                "algebraically impossible for the bicycle construction."
            )

        # --- Step 3: shared lift table ---
        # The lift table covers all (i, j) indices that both base_X
        # and base_Z look up.  Both are (r, 2r).
        self.lift_table = SharedLiftTable(rows=r, cols=2 * r, L=L, seed=seed + 1)

        # --- Step 4: lift ---
        self.HX: sp.csr_matrix = lift_matrix(self.base_X, self.lift_table)
        self.HZ: sp.csr_matrix = lift_matrix(self.base_Z, self.lift_table)

        # --- Step 5: verify invariants ---
        check_css_orthogonality(self.HX, self.HZ)
        check_no_zero_rows_or_cols(self.HX, name="H_X")
        check_no_zero_rows_or_cols(self.HZ, name="H_Z")

    @property
    def n(self) -> int:
        """Number of physical qubits."""
        return self.HX.shape[1]

    @property
    def k(self) -> int:
        """Number of logical qubits via GF(2) rank."""
        rank_X = binary_rank(self.HX)
        rank_Z = binary_rank(self.HZ)
        return self.n - rank_X - rank_Z

    def __repr__(self) -> str:
        return (
            f"CSSCode(r={self.r}, c={self.c}, L={self.L}, seed={self.seed}, "
            f"n={self.n}, HX={self.HX.shape}, HZ={self.HZ.shape})"
        )
