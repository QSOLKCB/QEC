"""
Shared-circulant lifting for protograph-based QLDPC codes.

Each protograph edge (i, j) maps to exactly ONE circulant shift integer
s in [0, P-1].  The critical invariant:

    If the same protograph position (i, j) is used by both H_X and H_Z,
    they must reuse the SAME shift s via a shared LiftTable instance.

Shifts use an additive structure: s(i, j) = (r_i + c_j) mod P, where
r_i and c_j are per-row and per-column offsets sampled lazily from a
seeded RNG.  This ensures that shift differences between rows are
constant across columns, which is required for CSS orthogonality.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from typing import Dict, Tuple

from .field import GF2e


class LiftTable:
    """
    Shared circulant lifting table with structured additive shifts.

    Shifts are computed as ``s(i, j) = (r_i + c_j) mod P``, where
    ``r_i`` and ``c_j`` are per-row and per-column offsets sampled lazily
    from a seeded RNG.  This additive structure ensures that the shift
    difference between any two rows i, k is constant across all columns:

        s(i, j) - s(k, j) = r_i - r_k   (independent of j)

    which is required for CSS orthogonality to hold after binary
    expansion of multi-row protographs with overlapping column support.

    Both H_X and H_Z construction must use the SAME LiftTable instance
    so that shared protograph positions receive identical circulant shifts.

    Thread safety
    -------------
    LiftTable is NOT thread-safe.  The lazy sampling writes to internal
    dicts without locking.  H_X and H_Z must be assembled sequentially
    using the same instance.

    Determinism
    -----------
    Shift values are deterministic for a given seed AND traversal order.
    The first call to ``get_shift`` with a previously unseen row index i
    (or column index j) advances the internal RNG to sample that offset.
    Determinism is guaranteed when the same ``(i, j)`` pairs are queried
    in the same order across runs.

    Parameters
    ----------
    P : int
        Circulant size (permutation matrices are P x P).
    seed : int
        Deterministic random seed for offset sampling.
    """

    def __init__(self, P: int, seed: int):
        if P < 1:
            raise ValueError(f"Circulant size P must be >= 1, got {P}")
        self._P = P
        self._rng = np.random.default_rng(seed)
        self._row_offsets: Dict[int, int] = {}
        self._col_offsets: Dict[int, int] = {}

    @property
    def P(self) -> int:
        """Circulant size."""
        return self._P

    def get_shift(self, i: int, j: int) -> int:
        """
        Return the circulant shift for protograph position (i, j).

        Computed as ``(r_i + c_j) mod P``.  Row offset ``r_i`` and column
        offset ``c_j`` are sampled from the seeded RNG on first access
        and cached for subsequent calls.
        """
        if i not in self._row_offsets:
            self._row_offsets[i] = int(self._rng.integers(0, self._P))
        if j not in self._col_offsets:
            self._col_offsets[j] = int(self._rng.integers(0, self._P))
        return (self._row_offsets[i] + self._col_offsets[j]) % self._P

    def table_size(self) -> int:
        """Return the number of cached row + column offsets."""
        return len(self._row_offsets) + len(self._col_offsets)

    def __repr__(self) -> str:
        return (
            f"LiftTable(P={self._P}, "
            f"rows={len(self._row_offsets)}, "
            f"cols={len(self._col_offsets)})"
        )


def circulant_shift_matrix(P: int, s: int) -> sparse.csr_matrix:
    """
    Build the P x P binary circulant permutation matrix for shift s.

    Entry (row, col) is 1 iff col = (row - s) mod P,
    equivalently row = (col + s) mod P.

    Parameters
    ----------
    P : int
        Matrix dimension (circulant size).  Must be >= 1.
    s : int
        Shift amount.  Reduced mod P internally, so values outside
        [0, P) (including negative) are accepted.

    Returns
    -------
    scipy.sparse.csr_matrix
        P x P permutation matrix with dtype uint8.
    """
    if P < 1:
        raise ValueError(f"Circulant size P must be >= 1, got {P}")
    s = s % P
    rows = np.arange(P)
    cols = (rows - s) % P
    data = np.ones(P, dtype=np.uint8)
    return sparse.csr_matrix((data, (rows, cols)), shape=(P, P), dtype=np.uint8)


def kron_companion_circulant(
    field: GF2e, a: int, P: int, s: int,
) -> sparse.csr_matrix:
    """
    Compute C(a) ⊗ π_s as a sparse binary matrix of shape (eP, eP).

    C(a) is the e x e companion matrix for field element a, and π_s is
    the P x P circulant permutation for shift s.

    Parameters
    ----------
    field : GF2e
        Finite field instance.
    a : int
        Field element in [0, 2^e).
    P : int
        Circulant size.  Must be >= 1.
    s : int
        Circulant shift.

    Returns
    -------
    scipy.sparse.csr_matrix
        Binary matrix of shape (eP, eP) with dtype uint8.
    """
    if P < 1:
        raise ValueError(f"Circulant size P must be >= 1, got {P}")

    e = field.e
    eP = e * P

    if a == 0:
        return sparse.csr_matrix((eP, eP), dtype=np.uint8)

    comp = field.companion_matrix(a)  # e x e dense uint8
    circ = circulant_shift_matrix(P, s)  # P x P sparse uint8

    # Kronecker product: sparse inputs avoid dense eP x eP intermediates.
    result = sparse.kron(sparse.csr_matrix(comp), circ, format="csr")
    result = result.astype(np.uint8)
    return result


# Backward-compat aliases so qldpc/__init__.py lazy imports don't crash.
LiftingTable = LiftTable


def generate_lifting_table(*args, **kwargs):
    raise NotImplementedError(
        "generate_lifting_table() has been removed. "
        "Use LiftTable(P, seed) with lazy get_shift(i, j) instead."
    )
