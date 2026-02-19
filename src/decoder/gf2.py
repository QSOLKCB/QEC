"""
Dense GF(2) linear algebra utilities.

Provides row-echelon form and rank computation over GF(2)
using standard Gaussian elimination with XOR row operations
on numpy uint8 arrays.  No scipy dependency.
"""

from __future__ import annotations

import numpy as np


def gf2_row_echelon(M, n_pivot_cols=None):
    """Row-reduce a binary matrix over GF(2).

    Args:
        M: Binary matrix (m x n), values in {0, 1}.
        n_pivot_cols: Only search for pivots in the first *n_pivot_cols*
            columns.  Row operations still apply to the full row width.
            Defaults to ``M.shape[1]`` (all columns eligible).

    Returns:
        (R, pivot_cols):
            R — row-echelon form, shape (m, n), dtype uint8.
            pivot_cols — list of pivot column indices (length = GF(2) rank).
    """
    R = (np.asarray(M, dtype=np.uint8) % 2).copy()
    m, n = R.shape
    if n_pivot_cols is None:
        n_pivot_cols = n

    pivot_cols: list[int] = []
    pivot_row = 0

    for col in range(n_pivot_cols):
        # Find first nonzero entry in this column at or below pivot_row.
        found = -1
        for row in range(pivot_row, m):
            if R[row, col] == 1:
                found = row
                break
        if found == -1:
            continue

        # Swap rows.
        if found != pivot_row:
            R[[pivot_row, found]] = R[[found, pivot_row]]

        # Eliminate below (full row width, including augmented columns).
        for row in range(pivot_row + 1, m):
            if R[row, col] == 1:
                R[row] ^= R[pivot_row]

        pivot_cols.append(col)
        pivot_row += 1

    return R, pivot_cols


def binary_rank_dense(M):
    """Compute the GF(2) rank of a dense binary matrix.

    Args:
        M: Binary matrix (m x n), values in {0, 1}.

    Returns:
        Integer rank over GF(2).
    """
    _, pivot_cols = gf2_row_echelon(M)
    return len(pivot_cols)
