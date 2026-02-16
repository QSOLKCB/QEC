"""
CSS quantum LDPC code from protograph pair over GF(2^e).

Construction pipeline:
    1. Build orthogonal protograph pair (B_X, B_Z) over GF(2^e)
    2. Generate shared circulant lifting table
    3. Lift to binary parity check matrices H_X, H_Z (sparse CSR)
    4. Verify structural invariants (orthogonality, column weight)

Parameters:
    n  = e * P * L     physical qubits
    k  = n - rank(H_X) - rank(H_Z)   logical qubits
    R ~= 1 - 2J/L     code rate
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from typing import Dict, Optional, Tuple

from .field import GF2e
from .protograph import ProtographPair, build_protograph_pair
from .lift import LiftingTable, generate_lifting_table, lift_to_binary
from .invariants import (
    ConstructionInvariantError,
    verify_css_orthogonality_sparse,
    verify_column_weight,
)


class CSSCode:
    """
    CSS quantum LDPC code from a protograph pair over GF(2^e).

    All construction is deterministic given the same parameters + seed.
    CSS orthogonality (H_X @ H_Z^T = 0 mod 2) is verified after
    construction and raises ConstructionInvariantError on failure.

    Attributes
    ----------
    H_X : sparse.csr_matrix
        X-type parity check matrix (detects Z errors).
    H_Z : sparse.csr_matrix
        Z-type parity check matrix (detects X errors).
    n : int
        Number of physical qubits.
    k : int
        Number of logical qubits.
    rate : float
        Code rate k/n.
    protograph : ProtographPair
        The underlying protograph pair.
    lifting_table : LiftingTable
        The shared circulant lifting table.
    """

    def __init__(
        self,
        protograph: ProtographPair,
        lifting_size: int = 32,
        seed: int = 42,
        target_col_weight: Optional[int] = None,
    ):
        self.protograph = protograph
        self.P = lifting_size
        self.field = protograph.field
        self.e = protograph.field.e
        self.seed = seed

        # Generate shared lifting table (deterministic from seed).
        self.lifting_table = generate_lifting_table(
            protograph, lifting_size, seed
        )

        # Lift to binary parity check matrices (sparse CSR).
        self.H_X, self.H_Z = lift_to_binary(
            protograph, self.lifting_table, self.field
        )

        # --- Structural invariant: CSS orthogonality ---
        if not verify_css_orthogonality_sparse(self.H_X, self.H_Z):
            raise ConstructionInvariantError(
                "H_X @ H_Z^T != 0 mod 2 -- CSS orthogonality violated. "
                "This is a construction bug, not a recoverable condition."
            )

        # --- Column weight constraint (if applicable) ---
        if target_col_weight is not None:
            if not verify_column_weight(self.H_X, target_col_weight):
                raise ConstructionInvariantError(
                    f"H_X column weight != {target_col_weight}. "
                    f"Column weight constraint violated."
                )
            if not verify_column_weight(self.H_Z, target_col_weight):
                raise ConstructionInvariantError(
                    f"H_Z column weight != {target_col_weight}. "
                    f"Column weight constraint violated."
                )

        self.n = self.H_X.shape[1]
        self.m_X = self.H_X.shape[0]
        self.m_Z = self.H_Z.shape[0]

        # Rank computation for moderate sizes.
        if self.n <= 20000:
            H_X_dense = self.H_X.toarray().astype(np.float64)
            H_Z_dense = self.H_Z.toarray().astype(np.float64)
            rx = np.linalg.matrix_rank(H_X_dense)
            rz = np.linalg.matrix_rank(H_Z_dense)
            self.k = max(0, self.n - rx - rz)
        else:
            self.k = max(0, self.n - self.m_X - self.m_Z)

        self.rate = self.k / self.n if self.n > 0 else 0.0

    def syndrome_X(self, z_error: np.ndarray) -> np.ndarray:
        """X-type syndrome from a Z-error pattern (length-n binary vector)."""
        return np.asarray(
            self.H_X.astype(np.int32) @ z_error.astype(np.int32)
        ).ravel() % 2

    def syndrome_Z(self, x_error: np.ndarray) -> np.ndarray:
        """Z-type syndrome from an X-error pattern (length-n binary vector)."""
        return np.asarray(
            self.H_Z.astype(np.int32) @ x_error.astype(np.int32)
        ).ravel() % 2

    def __repr__(self) -> str:
        return (
            f"CSSCode(n={self.n}, k={self.k}, "
            f"rate={self.rate:.4f}, "
            f"J={self.protograph.J}, L={self.protograph.L}, "
            f"P={self.P}, e={self.e})"
        )


# ── Pre-defined code configurations ──────────────────────────────────

PREDEFINED_CODES: Dict[str, Dict] = {
    "rate_0.50": {
        "J": 1,
        "L": 4,
        "field_degree": 3,
        "description": (
            "Rate-1/2 protograph QLDPC code (J=1, L=4, GF(2^3)). "
            "Komoto-Kasai 2025, Table I."
        ),
    },
    "rate_0.60": {
        "J": 2,
        "L": 10,
        "field_degree": 3,
        "description": (
            "Rate-3/5 protograph QLDPC code (J=2, L=10, GF(2^3)). "
            "Komoto-Kasai 2025, Table I."
        ),
    },
    "rate_0.75": {
        "J": 1,
        "L": 8,
        "field_degree": 3,
        "description": (
            "Rate-3/4 protograph QLDPC code (J=1, L=8, GF(2^3)). "
            "Komoto-Kasai 2025, Table I."
        ),
    },
}


def create_code(
    name: str = "rate_0.50",
    lifting_size: int = 32,
    seed: int = 42,
    target_col_weight: Optional[int] = None,
) -> CSSCode:
    """
    Instantiate a pre-defined quantum LDPC code.

    Parameters
    ----------
    name : str
        One of 'rate_0.50', 'rate_0.60', 'rate_0.75'.
    lifting_size : int
        Circulant permutation size P.
    seed : int
        Construction seed (deterministic).
    target_col_weight : int or None
        If set, verify binary column weight constraint.

    Returns
    -------
    CSSCode
        Fully constructed CSS code with verified invariants.
    """
    if name not in PREDEFINED_CODES:
        raise ValueError(
            f"Unknown code '{name}'. Choose from {sorted(PREDEFINED_CODES)}"
        )
    cfg = PREDEFINED_CODES[name]
    gf = GF2e(cfg["field_degree"])
    proto = build_protograph_pair(
        J=cfg["J"], L=cfg["L"], gf=gf, seed=seed
    )
    return CSSCode(
        proto,
        lifting_size=lifting_size,
        seed=seed,
        target_col_weight=target_col_weight,
    )
