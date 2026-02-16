"""
Protograph pair construction for CSS QLDPC codes.

Builds orthogonal protograph matrix pairs (B_X, B_Z) over GF(2^e)
satisfying:
    sum_j B_X[i1, j] * B_Z[i2, j] = 0  in GF(2^e)
for all row pairs (i1, i2).

Orthogonality is enforced by construction (self-orthogonal: B_X = B_Z),
not by iterative repair.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .field import GF2e
from .invariants import ConstructionInvariantError


@dataclass
class ProtographPair:
    """
    Orthogonal protograph matrix pair (B_X, B_Z) over GF(2^e).

    Attributes
    ----------
    J : int
        Number of check rows per stabiliser type.
    L : int
        Number of variable columns.
    B_X : np.ndarray
        J x L matrix with entries in {0, ..., q-1}.
    B_Z : np.ndarray
        J x L matrix with entries in {0, ..., q-1}.
    field : GF2e
        The finite field.
    """

    J: int
    L: int
    B_X: np.ndarray
    B_Z: np.ndarray
    field: GF2e

    @property
    def rate(self) -> float:
        """Code rate R = 1 - 2J/L."""
        return 1.0 - 2.0 * self.J / self.L


def verify_orthogonality_gf(
    B_X: np.ndarray, B_Z: np.ndarray, gf: GF2e
) -> bool:
    """Check B_X . B_Z^T = 0 over GF(2^e)."""
    J_X = B_X.shape[0]
    J_Z = B_Z.shape[0]
    for i1 in range(J_X):
        for i2 in range(J_Z):
            acc = 0
            for k in range(B_X.shape[1]):
                acc = gf.add(acc, gf.mul(int(B_X[i1, k]), int(B_Z[i2, k])))
            if acc != 0:
                return False
    return True


def build_protograph_pair(
    J: int,
    L: int,
    gf: GF2e,
    seed: int = 42,
    unit_entries: bool = False,
) -> ProtographPair:
    """
    Build a self-orthogonal protograph pair over GF(2^e).

    Orthogonality is enforced **by construction**: B_X = B_Z = B where
    B @ B^T = 0 over GF(2^e).

    Parameters
    ----------
    J : int
        Number of check rows per stabiliser type.
    L : int
        Number of variable columns (must satisfy L >= 2*J).
    gf : GF2e
        Finite field.
    seed : int
        Deterministic random seed.
    unit_entries : bool
        If True, all nonzero entries are set to 1 (identity element).
        This guarantees binary column weight equals protograph column weight.

    Construction strategies:
    - J = 1: all-ones row.  Self-orthogonal because L * 1 = 0 in char 2
      (when L is even).
    - J >= 2: paired-column blocks with identical entries in each pair,
      exploiting char-2 cancellation: a + a = 0.
    """
    rng = np.random.default_rng(seed)
    q = gf.q

    B = np.zeros((J, L), dtype=np.int32)

    if J == 1:
        # All-ones row.  sum = L * 1 = 0 in char 2 when L is even.
        B[0, :] = 1
        # Verify self-orthogonality (L must be even for char-2 field)
        acc = 0
        for j in range(L):
            acc = gf.add(acc, gf.mul(1, 1))
        if acc != 0:
            # L is odd: adjust last entry so the sum cancels.
            target = gf.add(acc, gf.mul(1, 1))  # remove old contribution
            found = False
            for v in range(1, q):
                if gf.mul(v, v) == target:
                    B[0, L - 1] = v
                    found = True
                    break
            if not found:
                B[0, L - 1] = 0
    else:
        # Multi-row self-orthogonal construction.
        #
        # Fill B in 2-column blocks.  For paired columns (j1, j2),
        # assign identical values: B[r, j1] = B[r, j2].
        #
        # Contribution to B @ B^T for any row pair (r1, r2):
        #   B[r1,j1]*B[r2,j1] + B[r1,j2]*B[r2,j2]
        #   = a*c + a*c  (in char 2)
        #   = 0
        #
        # This guarantees B @ B^T = 0 globally without pairwise patching.
        n_pairs = L // 2
        row_idx = 0
        for p_idx in range(n_pairs):
            j1 = 2 * p_idx
            j2 = 2 * p_idx + 1
            r1 = row_idx % J
            r2 = (row_idx + 1) % J
            row_idx += 2

            if unit_entries:
                a, c = 1, 1
            else:
                a = int(rng.integers(1, q))
                c = int(rng.integers(1, q))

            B[r1, j1] = a
            B[r1, j2] = a
            B[r2, j1] = c
            B[r2, j2] = c

        # Odd leftover column: all-zero (contributes nothing to B @ B^T)

    B_X = B.copy()
    B_Z = B.copy()

    if not verify_orthogonality_gf(B_X, B_Z, gf):
        raise ConstructionInvariantError(
            f"Protograph pair (J={J}, L={L}) failed GF(2^{gf.e}) "
            f"orthogonality check. This is a bug in the construction."
        )

    return ProtographPair(J=J, L=L, B_X=B_X, B_Z=B_Z, field=gf)
