"""
CSS code construction from a protograph pair over GF(2^e).

Construction pipeline:
    1. Accept base matrices H_X_base, H_Z_base over GF(2^e)
    2. Instantiate a shared LiftTable(P, seed)
    3. For each nonzero entry a at position (i, j):
       - Obtain shared circulant shift s = lift_table.get_shift(i, j)
       - Build block = C(a) ⊗ π_s   (eP × eP sparse binary)
    4. Assemble sparse H_X and H_Z in CSR format
    5. Verify CSS orthogonality: (H_X @ H_Z^T) % 2 == 0
       If violated, raise ConstructionInvariantError (no repair).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .field import GF2e
from .invariants import ConstructionInvariantError
from .lift import LiftTable, kron_companion_circulant


class ProtographPair:
    """
    Pair of base parity-check matrices over GF(2^e).

    Parameters
    ----------
    field : GF2e
        Finite field for entry validation.
    H_X_base : array_like
        2D integer array with entries in [0, field.q).
    H_Z_base : array_like
        2D integer array with entries in [0, field.q).
        Must have the same number of columns as H_X_base.
    """

    def __init__(
        self,
        field: GF2e,
        H_X_base: np.ndarray,
        H_Z_base: np.ndarray,
    ):
        H_X_base = np.asarray(H_X_base, dtype=np.int32)
        H_Z_base = np.asarray(H_Z_base, dtype=np.int32)

        if H_X_base.ndim != 2 or H_Z_base.ndim != 2:
            raise ValueError("Base matrices must be 2D")

        if H_X_base.shape[1] != H_Z_base.shape[1]:
            raise ValueError(
                f"Column count mismatch: H_X_base has {H_X_base.shape[1]}, "
                f"H_Z_base has {H_Z_base.shape[1]}"
            )

        if np.any((H_X_base < 0) | (H_X_base >= field.q)):
            raise ValueError(
                f"H_X_base entries must be in [0, {field.q})"
            )
        if np.any((H_Z_base < 0) | (H_Z_base >= field.q)):
            raise ValueError(
                f"H_Z_base entries must be in [0, {field.q})"
            )

        self.field = field
        self.H_X_base = H_X_base
        self.H_Z_base = H_Z_base


class CSSCode:
    """
    CSS quantum LDPC code from a protograph pair over GF(2^e).

    Construction is fully deterministic given the same parameters and seed.
    CSS orthogonality (H_X @ H_Z^T = 0 mod 2) is verified after assembly;
    failure raises ConstructionInvariantError with no repair attempt.

    Attributes
    ----------
    H_X : sparse.csr_matrix
        X-type binary parity-check matrix.
    H_Z : sparse.csr_matrix
        Z-type binary parity-check matrix.
    lift_table : LiftTable
        The shared circulant lifting table used for both H_X and H_Z.
    """

    def __init__(
        self,
        field: GF2e,
        proto: ProtographPair,
        P: int,
        seed: int,
    ):
        lift_table = LiftTable(P, seed)

        # Assemble binary matrices — H_X first so its get_shift calls
        # populate the LiftTable before H_Z reuses the same entries.
        H_X = _assemble_binary(field, proto.H_X_base, P, lift_table)
        H_Z = _assemble_binary(field, proto.H_Z_base, P, lift_table)

        # CSS orthogonality check: (H_X @ H_Z^T) % 2 must be all-zero.
        product = H_X.astype(np.int32) @ H_Z.astype(np.int32).T
        if sparse.issparse(product):
            product = product.copy()
            product.data = product.data % 2
            product.eliminate_zeros()
            orthogonal = product.nnz == 0
        else:
            orthogonal = bool(np.all(product % 2 == 0))

        if not orthogonal:
            raise ConstructionInvariantError(
                "H_X @ H_Z^T != 0 mod 2 — CSS orthogonality violated. "
                "This indicates the base matrices or lifting are incompatible."
            )

        self.H_X = H_X
        self.H_Z = H_Z
        self.lift_table = lift_table


def _assemble_binary(
    field: GF2e,
    base: np.ndarray,
    P: int,
    lift_table: LiftTable,
) -> sparse.csr_matrix:
    """
    Expand a base matrix over GF(2^e) into a binary sparse matrix.

    Each nonzero entry a at position (i, j) becomes the eP × eP block
    C(a) ⊗ π_s placed at block position (i, j), where s is obtained
    from the shared lift_table.

    Parameters
    ----------
    field : GF2e
        Finite field.
    base : np.ndarray
        m × n matrix of field elements (integers).
    P : int
        Circulant size.
    lift_table : LiftTable
        Shared lifting table.

    Returns
    -------
    sparse.csr_matrix
        Binary matrix of shape (m * e * P, n * e * P).
    """
    m, n = base.shape
    e = field.e
    eP = e * P

    block_rows = []
    for i in range(m):
        blocks = []
        for j in range(n):
            a = int(base[i, j])
            if a == 0:
                blocks.append(sparse.csr_matrix((eP, eP), dtype=np.uint8))
            else:
                s = lift_table.get_shift(i, j)
                blocks.append(kron_companion_circulant(field, a, P, s))
        block_rows.append(sparse.hstack(blocks, format="csr"))

    return sparse.vstack(block_rows, format="csr").astype(np.uint8)


# Backward-compat stubs so qldpc/__init__.py lazy imports don't crash.
PREDEFINED_CODES = {}


def create_code(*args, **kwargs):
    raise NotImplementedError(
        "create_code() has been removed. "
        "Use CSSCode(field, proto, P, seed) directly."
    )
