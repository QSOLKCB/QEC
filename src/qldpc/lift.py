"""
Shared-circulant lifting for protograph-based QLDPC codes.

The lifting table maps each protograph position (i, j) to a circulant
shift value.  This table is shared between H_X and H_Z construction,
which is the key structural requirement for CSS orthogonality.

For orthogonality to hold after lifting, all positions in the same
protograph column j must share the same circulant shift.  This ensures:

    (C(a) kron pi_j) @ (C(b) kron pi_j)^T = C(a) C(b)^T kron I_P

so GF(2^e)-level orthogonality carries through to the binary expansion.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from typing import Dict, Tuple
from dataclasses import dataclass, field as dc_field

from .field import GF2e
from .protograph import ProtographPair


@dataclass
class LiftingTable:
    """
    Shared circulant lifting table.

    Maps protograph positions (i, j) to circulant shift values.
    Both H_X and H_Z must use the SAME table instance --
    generating independent shifts for each would break CSS orthogonality.

    Attributes
    ----------
    shifts : dict
        Mapping (i, j) -> circulant shift in [0, P).
    P : int
        Circulant size (permutation matrices are P x P).
    """

    shifts: Dict[Tuple[int, int], int]
    P: int

    def get_shift(self, i: int, j: int) -> int:
        """Look up the circulant shift for protograph position (i, j)."""
        return self.shifts[(i, j)]

    def has_position(self, i: int, j: int) -> bool:
        """Check if position (i, j) has a registered shift."""
        return (i, j) in self.shifts


def generate_lifting_table(
    proto: ProtographPair,
    P: int,
    seed: int,
) -> LiftingTable:
    """
    Generate a shared circulant lifting table.

    One shift per column j, shared across all rows and between H_X / H_Z.
    The table is keyed by (i, j) for each nonzero protograph position,
    but all entries in the same column j map to the same shift value.

    Parameters
    ----------
    proto : ProtographPair
        The protograph pair (B_X, B_Z).
    P : int
        Circulant permutation size.
    seed : int
        Deterministic random seed.

    Returns
    -------
    LiftingTable
        Shared lifting table for both H_X and H_Z construction.
    """
    rng = np.random.default_rng(seed)
    J, L = proto.J, proto.L

    # Generate one shift per column (deterministic, ordered by j).
    col_shifts = [int(rng.integers(0, P)) for _ in range(L)]

    # Populate the (i, j) -> shift mapping.
    # All nonzero positions in the same column share the same shift.
    shifts: Dict[Tuple[int, int], int] = {}
    for j in range(L):
        for i in range(J):
            if proto.B_X[i, j] != 0 or proto.B_Z[i, j] != 0:
                shifts[(i, j)] = col_shifts[j]

    return LiftingTable(shifts=shifts, P=P)


def _build_circulant_permutation(shift: int, P: int) -> sparse.csr_matrix:
    """
    Build a P x P circulant permutation matrix with given shift.

    Entry (row, col) is 1 iff row = (col + shift) mod P.
    """
    rows = np.arange(P)
    cols = (rows - shift) % P
    data = np.ones(P, dtype=np.uint8)
    return sparse.csr_matrix((data, (rows, cols)), shape=(P, P), dtype=np.uint8)


def lift_to_binary(
    proto: ProtographPair,
    table: LiftingTable,
    gf: GF2e,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Lift a protograph pair to binary parity check matrices H_X, H_Z.

    Construction:
        For each nonzero protograph entry (i, j):
            block(i, j) = C(B[i, j]) kron pi(i, j)
        where C is the companion matrix and pi is the shared circulant.

    The SAME lifting table is used for both H_X and H_Z, ensuring
    CSS orthogonality is preserved from the GF(2^e) level.

    Parameters
    ----------
    proto : ProtographPair
        Orthogonal protograph pair.
    table : LiftingTable
        Shared circulant lifting table.
    gf : GF2e
        Finite field for companion matrices.

    Returns
    -------
    (H_X, H_Z) : tuple of sparse.csr_matrix
        Binary parity check matrices in CSR format.
    """
    J, L = proto.J, proto.L
    e = gf.e
    P = table.P
    block = e * P

    # Pre-compute circulant permutations (cached per column).
    circulants: Dict[int, sparse.csr_matrix] = {}
    for (i, j), shift in table.shifts.items():
        if shift not in circulants:
            circulants[shift] = _build_circulant_permutation(shift, P)

    # Build H_X and H_Z in COO format for efficient construction.
    hx_rows, hx_cols, hx_data = [], [], []
    hz_rows, hz_cols, hz_data = [], [], []

    for i in range(J):
        for j in range(L):
            r0 = i * block
            c0 = j * block

            if not table.has_position(i, j):
                continue

            shift = table.get_shift(i, j)
            perm = circulants[shift]

            if proto.B_X[i, j] != 0:
                comp = gf.companion_matrix(int(proto.B_X[i, j]))
                block_mat = sparse.kron(
                    sparse.csr_matrix(comp), perm, format="coo"
                )
                hx_rows.extend(block_mat.row + r0)
                hx_cols.extend(block_mat.col + c0)
                hx_data.extend(block_mat.data)

            if proto.B_Z[i, j] != 0:
                comp = gf.companion_matrix(int(proto.B_Z[i, j]))
                block_mat = sparse.kron(
                    sparse.csr_matrix(comp), perm, format="coo"
                )
                hz_rows.extend(block_mat.row + r0)
                hz_cols.extend(block_mat.col + c0)
                hz_data.extend(block_mat.data)

    total_rows = J * block
    total_cols = L * block

    H_X = sparse.coo_matrix(
        (hx_data, (hx_rows, hx_cols)),
        shape=(total_rows, total_cols),
        dtype=np.uint8,
    ).tocsr()

    H_Z = sparse.coo_matrix(
        (hz_data, (hz_rows, hz_cols)),
        shape=(total_rows, total_cols),
        dtype=np.uint8,
    ).tocsr()

    return H_X, H_Z
