"""
Structural invariants for QLDPC CSS codes.

All invariant checks are assertions that must hold by construction.
If any check fails, it indicates a bug in the construction logic,
not a recoverable runtime condition.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


class ConstructionInvariantError(Exception):
    """Raised when a code construction violates a structural invariant."""


def verify_css_orthogonality_sparse(
    H_X: sparse.spmatrix, H_Z: sparse.spmatrix
) -> bool:
    """
    Check H_X @ H_Z^T = 0 (mod 2) using sparse matrices.

    Returns True if CSS orthogonality holds.
    """
    product = H_X.astype(np.int32) @ H_Z.astype(np.int32).T
    # Convert to dense for mod check (product is typically small:
    # m_X x m_Z where m is number of check rows)
    if sparse.issparse(product):
        product = product.toarray()
    return bool(np.all(product % 2 == 0))


def verify_css_orthogonality_dense(
    H_X: np.ndarray, H_Z: np.ndarray
) -> bool:
    """
    Check H_X @ H_Z^T = 0 (mod 2) using dense matrices.

    Returns True if CSS orthogonality holds.
    """
    product = (H_X.astype(np.int32) @ H_Z.astype(np.int32).T) % 2
    return bool(np.all(product == 0))


def verify_column_weight(
    H: sparse.spmatrix, target_weight: int
) -> bool:
    """
    Verify that every column of H has weight exactly *target_weight*.

    Parameters
    ----------
    H : sparse matrix
        Binary parity check matrix.
    target_weight : int
        Expected column weight.

    Returns
    -------
    bool
        True if all columns have the target weight.
    """
    if sparse.issparse(H):
        col_weights = np.array(H.astype(bool).astype(int).sum(axis=0)).ravel()
    else:
        col_weights = np.sum(H.astype(bool).astype(int), axis=0)
    return bool(np.all(col_weights == target_weight))
