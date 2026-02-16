"""
Shared-circulant lifting for protograph-based QLDPC codes.

Each protograph edge (i, j) maps to exactly ONE circulant shift integer
s in [0, P-1].  The critical invariant:

    If the same protograph position (i, j) is used by both H_X and H_Z,
    they must reuse the SAME shift s via a shared LiftTable instance.

Shifts use an additive structure: s(i, j) = (r_i + c_j) mod P, where
r_i and c_j are per-row and per-column offsets derived from a pure
deterministic hash of (seed, kind, index).  This ensures that shift
differences between rows are constant across columns, which is required
for CSS orthogonality.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np
from scipy import sparse
from typing import Dict

from .field import GF2e


def _offset(seed: int, kind: int, index: int, P: int) -> int:
    """
    Pure deterministic function: (seed, kind, index, P) -> int in [0, P).

    Uses SHA-256 of the packed (seed, kind, index) triple.  The result
    depends only on the inputs, not on call order or mutable state.
    ``kind`` distinguishes rows (0) from columns (1).
    """
    digest = hashlib.sha256(
        struct.pack(">qqq", seed, kind, index)
    ).digest()
    # Read first 8 bytes as unsigned 64-bit big-endian integer.
    value = struct.unpack(">Q", digest[:8])[0]
    return value % P


_KIND_ROW = 0
_KIND_COL = 1


class LiftTable:
    """
    Shared circulant lifting table with structured additive shifts.

    Shifts are computed as ``s(i, j) = (r_i + c_j) mod P``, where
    ``r_i`` and ``c_j`` are per-row and per-column offsets derived from
    a pure deterministic hash of ``(seed, kind, index)``.  This additive
    structure ensures that the shift difference between any two rows
    i, k is constant across all columns:

        s(i, j) - s(k, j) = r_i - r_k   (independent of j)

    which is required for CSS orthogonality to hold after binary
    expansion of multi-row protographs with overlapping column support.

    Both H_X and H_Z construction must use the SAME LiftTable instance
    so that shared protograph positions receive identical circulant shifts.

    Determinism
    -----------
    Offsets are computed by a pure function of ``(seed, index, P)`` with
    no mutable RNG state.  Shifts are:

    - Deterministic across traversal order (query order does not matter)
    - Reproducible across runs and processes
    - Thread-safe for concurrent reads

    Results are cached for speed, but caching does not affect output
    values.

    Parameters
    ----------
    P : int
        Circulant size (permutation matrices are P x P).
    seed : int
        Deterministic seed used in the hash function.
    """

    def __init__(self, P: int, seed: int):
        if P < 1:
            raise ValueError(f"Circulant size P must be >= 1, got {P}")
        self._P = P
        self._seed = seed
        self._row_cache: Dict[int, int] = {}
        self._col_cache: Dict[int, int] = {}

    @property
    def P(self) -> int:
        """Circulant size."""
        return self._P

    def get_shift(self, i: int, j: int) -> int:
        """
        Return the circulant shift for protograph position (i, j).

        Computed as ``(r_i + c_j) mod P`` where ``r_i`` and ``c_j`` are
        pure deterministic functions of ``(seed, index, P)``.  Cached
        for speed; caching does not affect output values.
        """
        r = self._row_cache.get(i)
        if r is None:
            r = _offset(self._seed, _KIND_ROW, i, self._P)
            self._row_cache[i] = r
        c = self._col_cache.get(j)
        if c is None:
            c = _offset(self._seed, _KIND_COL, j, self._P)
            self._col_cache[j] = c
        return (r + c) % self._P

    def table_size(self) -> int:
        """Return the number of cached row + column offsets."""
        return len(self._row_cache) + len(self._col_cache)

    def __repr__(self) -> str:
        return (
            f"LiftTable(P={self._P}, "
            f"rows={len(self._row_cache)}, "
            f"cols={len(self._col_cache)})"
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
