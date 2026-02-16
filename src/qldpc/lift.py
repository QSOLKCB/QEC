"""
Shared-circulant lifting for protograph-based QLDPC codes.

Each protograph edge (i, j) maps to exactly ONE circulant shift integer
s in [0, P-1].  The critical invariant:

    If the same protograph position (i, j) is used by both H_X and H_Z,
    they must reuse the SAME shift s via a shared LiftTable instance.

Shifts are sampled lazily: the first call to ``get_shift(i, j)`` draws
from a seeded RNG and caches the result; subsequent calls return the
cached value.  Determinism is guaranteed by the seed.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from typing import Dict, Tuple

from .field import GF2e


class LiftTable:
    """
    Shared circulant lifting table with lazy per-edge shift sampling.

    Both H_X and H_Z construction must use the SAME LiftTable instance
    so that shared protograph positions receive identical circulant shifts.

    Parameters
    ----------
    P : int
        Circulant size (permutation matrices are P x P).
    seed : int
        Deterministic random seed for shift sampling.
    """

    def __init__(self, P: int, seed: int):
        if P < 1:
            raise ValueError(f"Circulant size P must be >= 1, got {P}")
        self._P = P
        self._rng = np.random.default_rng(seed)
        self._shifts: Dict[Tuple[int, int], int] = {}

    @property
    def P(self) -> int:
        """Circulant size."""
        return self._P

    def get_shift(self, i: int, j: int) -> int:
        """
        Return the circulant shift for protograph position (i, j).

        On first call for a given (i, j), samples uniformly from [0, P)
        using the seeded RNG and caches the result.  Subsequent calls
        return the cached value.
        """
        key = (i, j)
        if key not in self._shifts:
            self._shifts[key] = int(self._rng.integers(0, self._P))
        return self._shifts[key]

    def table_size(self) -> int:
        """Return the number of (i, j) positions with cached shifts."""
        return len(self._shifts)

    def __repr__(self) -> str:
        return f"LiftTable(P={self._P}, cached={self.table_size()})"


def circulant_shift_matrix(P: int, s: int) -> sparse.csr_matrix:
    """
    Build the P x P binary circulant permutation matrix for shift s.

    Entry (row, col) is 1 iff col = (row - s) mod P,
    equivalently row = (col + s) mod P.

    Parameters
    ----------
    P : int
        Matrix dimension (circulant size).
    s : int
        Shift amount in [0, P).

    Returns
    -------
    scipy.sparse.csr_matrix
        P x P permutation matrix with dtype uint8.
    """
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
        Circulant size.
    s : int
        Circulant shift.

    Returns
    -------
    scipy.sparse.csr_matrix
        Binary matrix of shape (eP, eP) with dtype uint8.
    """
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
