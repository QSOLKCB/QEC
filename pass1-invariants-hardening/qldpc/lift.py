"""Deterministic shared-circulant lifting.

A *SharedLiftTable* is an immutable mapping

    edge_lifts[(i, j)]  →  circulant_shift   (int in [0, L))

built **once** from a seeded RNG.  Both H_X and H_Z must reuse the
same table — no independent sampling allowed.  This is what
guarantees CSS orthogonality by construction.

``lift_matrix`` expands a small *base matrix* (over {0, 1}) into the
binary *lifted matrix* by replacing every 1-entry with a cyclic
permutation matrix of size L and every 0-entry with the L×L zero
block.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from typing import Dict, Tuple


class SharedLiftTable:
    """Deterministic, immutable edge-lift mapping.

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
        self._table: Dict[Tuple[int, int], int] = {}
        for i in range(rows):
            for j in range(cols):
                self._table[(i, j)] = int(rng.randint(0, L))

    def __getitem__(self, key: Tuple[int, int]) -> int:
        return self._table[key]

    def items(self):
        return self._table.items()

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
    row_offset: int = 0,
    col_offset: int = 0,
) -> sp.csr_matrix:
    """Lift a binary base matrix into a sparse lifted matrix.

    Parameters
    ----------
    base : ndarray of shape (r, c)
        Binary base / protograph matrix with entries in {0, 1}.
    lift_table : SharedLiftTable
        The shared edge-lift mapping.
    row_offset, col_offset : int
        Offsets into the lift table indices.  Useful when H_X and H_Z
        use different rows of the same protograph but share the lift
        table.

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
                shift = lift_table[(i + row_offset, j + col_offset)]
                row_blocks.append(_circulant_permutation(shift, L))
            else:
                row_blocks.append(sp.csr_matrix((L, L), dtype=np.int8))
        blocks.append(row_blocks)
    return sp.bmat(blocks, format="csr").astype(np.int8)
