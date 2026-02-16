"""Structural invariant checks for QLDPC CSS codes.

Every check either passes silently or raises ``ConstructionInvariantError``.
No repair / retry / patching is ever attempted.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class ConstructionInvariantError(Exception):
    """Raised when a construction invariant is violated."""


def check_css_orthogonality(
    HX: sp.csr_matrix,
    HZ: sp.csr_matrix,
) -> None:
    """Assert (H_X @ H_Z^T) mod 2 == 0.

    Raises ``ConstructionInvariantError`` if the CSS orthogonality
    condition is violated.  Fully sparse — no dense conversion.
    """
    product = (HX @ HZ.T).tocsr()
    product.data %= 2
    product.eliminate_zeros()
    if product.nnz != 0:
        raise ConstructionInvariantError(
            f"CSS orthogonality violated: H_X @ H_Z^T has "
            f"{product.nnz} non-zero entries mod 2."
        )


def binary_rank(mat: sp.spmatrix) -> int:
    """Compute rank of a binary matrix over GF(2).

    Uses sparse Gaussian elimination with set-based row operations.
    No float arithmetic.  No dense conversion.
    """
    csr = mat.tocsr()
    m = csr.shape[0]

    # Represent each row as a frozenset of column indices where entry = 1.
    # Reduce entries mod 2 first (in case of duplicates).
    rows: list[set[int]] = []
    for i in range(m):
        start, end = csr.indptr[i], csr.indptr[i + 1]
        cols = set()
        for idx in range(start, end):
            c = int(csr.indices[idx])
            v = int(csr.data[idx]) % 2
            if v:
                cols.add(c)
        if cols:
            rows.append(cols)

    # Gaussian elimination over GF(2).
    pivot_row: dict[int, set[int]] = {}  # pivot_col -> row
    rank = 0

    for row in rows:
        current = set(row)
        while current:
            min_col = min(current)
            if min_col in pivot_row:
                # XOR with existing pivot row (symmetric_difference = XOR)
                current = current.symmetric_difference(pivot_row[min_col])
            else:
                pivot_row[min_col] = current
                rank += 1
                break

    return rank


def check_no_zero_rows_or_cols(
    mat: sp.csr_matrix,
    name: str = "matrix",
) -> None:
    """Assert that no row or column is entirely zero.

    Raises ``ConstructionInvariantError`` on failure.
    """
    # Row check (CSR row sums are cheap).
    row_sums = np.asarray(np.abs(mat).sum(axis=1)).ravel()
    zero_rows = np.where(row_sums == 0)[0]
    if zero_rows.size > 0:
        raise ConstructionInvariantError(
            f"{name} has {zero_rows.size} all-zero row(s): {zero_rows.tolist()}"
        )

    col_sums = np.asarray(np.abs(mat).sum(axis=0)).ravel()
    zero_cols = np.where(col_sums == 0)[0]
    if zero_cols.size > 0:
        raise ConstructionInvariantError(
            f"{name} has {zero_cols.size} all-zero column(s): {zero_cols.tolist()}"
        )


def check_column_weight(
    mat: sp.csr_matrix,
    expected_weight: int,
    name: str = "matrix",
) -> None:
    """Assert every column has exactly ``expected_weight`` non-zero entries.

    Raises ``ConstructionInvariantError`` on failure.
    """
    col_weights = np.asarray(np.abs(mat).sum(axis=0)).ravel()
    bad = np.where(col_weights != expected_weight)[0]
    if bad.size > 0:
        examples = bad[:5].tolist()
        weights = col_weights[bad[:5]].tolist()
        raise ConstructionInvariantError(
            f"{name}: expected column weight {expected_weight} but "
            f"columns {examples} have weights {weights}"
        )
