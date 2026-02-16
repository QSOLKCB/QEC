"""Deterministic shared-circulant lifting with additive shifts.

Shift structure
---------------
The shift for base-matrix entry (i, j) is computed as

    s(i, j) = (row_shift[i] + col_shift[j]) mod L

where ``row_shift`` and ``col_shift`` are deterministic arrays derived
from the seed.  This **additive** structure is the key to preserving
CSS orthogonality through lifting.

Why additive shifts preserve orthogonality
------------------------------------------
The (i, k) block of  H_X @ H_Z^T  is

    Σ_j  base_X[i,j] · base_Z[k,j] · P^{s(i,j) − s(k,j)}

With additive shifts, s(i,j) − s(k,j) = row_shift[i] − row_shift[k],
which is **constant over j**.  Factor it out:

    P^{row_shift[i] − row_shift[k]}  ·  Σ_j  base_X[i,j] · base_Z[k,j] · I

The scalar sum is exactly (base_X @ base_Z^T)[i,k].  For the bicycle
construction this is 0 mod 2, so every block vanishes.

``lift_matrix`` expands a small *base matrix* (over {0, 1}) into the
binary *lifted matrix* by replacing every 1-entry with a cyclic
permutation matrix of size L and every 0-entry with the L×L zero
block.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class SharedLiftTable:
    """Deterministic, immutable additive-shift lift mapping.

    Shifts are computed as  s(i, j) = (row_shift[i] + col_shift[j]) mod L
    where ``row_shift`` and ``col_shift`` are drawn once from a seeded RNG.

    This additive structure guarantees that  s(i,j) − s(k,j)  is
    independent of j, which is necessary and sufficient for lifted CSS
    orthogonality to follow from base-matrix orthogonality.

    Parameters
    ----------
    rows, cols : int
        Dimensions of the protograph (base matrix).
    L : int
        Circulant block size (lift size).
    seed : int
        RNG seed — same seed ⇒ identical table.
    """

    def __init__(self, rows: int, cols: int, L: int, seed: int) -> None:
        self.rows = rows
        self.cols = cols
        self.L = L
        self.seed = seed

        rng = np.random.RandomState(seed)  # deterministic
        self._row_shift: np.ndarray = rng.randint(0, L, size=rows)
        self._col_shift: np.ndarray = rng.randint(0, L, size=cols)

    def __getitem__(self, key: tuple[int, int]) -> int:
        i, j = key
        return int((self._row_shift[i] + self._col_shift[j]) % self.L)

    def __repr__(self) -> str:
        return f"SharedLiftTable(rows={self.rows}, cols={self.cols}, L={self.L}, seed={self.seed})"


def _circulant_permutation(shift: int, L: int) -> sp.csr_matrix:
    """Return the L×L circulant permutation matrix with given shift.

    Entry (r, (r + shift) % L) = 1 for each row r.
    """
    row_idx = np.arange(L)
    col_idx = (row_idx + shift) % L
    data = np.ones(L, dtype=np.int8)
    return sp.csr_matrix((data, (row_idx, col_idx)), shape=(L, L), dtype=np.int8)


def lift_matrix(
    base: np.ndarray,
    lift_table: SharedLiftTable,
) -> sp.csr_matrix:
    """Lift a binary base matrix into a sparse lifted matrix.

    Each 1-entry at position (i, j) in the base matrix is replaced by
    the L×L circulant permutation  P^{s(i,j)}  where the shift is
    obtained from ``lift_table[(i, j)]``.  Each 0-entry becomes the
    L×L zero block.

    Parameters
    ----------
    base : ndarray of shape (r, c)
        Binary base / protograph matrix with entries in {0, 1}.
    lift_table : SharedLiftTable
        The shared additive-shift lift mapping.

    Returns
    -------
    scipy.sparse.csr_matrix
        Binary lifted matrix of shape (r*L, c*L).
    """
    r, c = base.shape
    L = lift_table.L
    blocks: list[list[sp.csr_matrix]] = []
    for i in range(r):
        row_blocks: list[sp.csr_matrix] = []
        for j in range(c):
            if base[i, j]:
                shift = lift_table[(i, j)]
                row_blocks.append(_circulant_permutation(shift, L))
            else:
                row_blocks.append(sp.csr_matrix((L, L), dtype=np.int8))
        blocks.append(row_blocks)
    return sp.bmat(blocks, format="csr").astype(np.int8)
