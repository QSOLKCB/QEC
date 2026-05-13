"""Deterministic CSS construction from protograph pairs.

This module intentionally provides a minimal, invariant-first surface used by
repository tests and local deterministic construction flows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .field import GF2e
from .invariants import ConstructionInvariantError, verify_css_orthogonality_sparse
from .lift import LiftTable, kron_companion_circulant


@dataclass(frozen=True)
class ProtographPair:
    """GF(2^e) protograph pair used for CSS lifting."""

    gf: GF2e
    H_X_base: np.ndarray
    H_Z_base: np.ndarray

    def __post_init__(self) -> None:
        hx = np.asarray(self.H_X_base, dtype=np.int32)
        hz = np.asarray(self.H_Z_base, dtype=np.int32)
        if hx.ndim != 2 or hz.ndim != 2:
            raise ValueError("H_X_base and H_Z_base must be 2D arrays")
        if hx.shape[1] != hz.shape[1]:
            raise ValueError("H_X_base and H_Z_base must have same column count")
        object.__setattr__(self, "H_X_base", hx)
        object.__setattr__(self, "H_Z_base", hz)


class CSSCode:
    """Deterministic CSS code lifted from a protograph pair."""

    def __init__(self, gf: GF2e, proto: ProtographPair, P: int, seed: int = 0):
        if P < 1:
            raise ValueError("P must be >= 1")
        if proto.gf.e != gf.e or proto.gf.q != gf.q or proto.gf.poly != gf.poly:
            raise ValueError("proto.gf must match gf")

        self.gf = gf
        self.proto = proto
        self.P = int(P)
        self.seed = int(seed)
        self.lift_table = LiftTable(self.P, self.seed)

        self.H_X = self._build_binary_parity(proto.H_X_base)
        self.H_Z = self._build_binary_parity(proto.H_Z_base)

        if not self._verify_gf_orthogonality(proto.H_X_base, proto.H_Z_base):
            raise ConstructionInvariantError("GF-level protograph orthogonality failed")
        if not verify_css_orthogonality_sparse(self.H_X, self.H_Z):
            raise ConstructionInvariantError("Binary CSS orthogonality failed")

    def _verify_gf_orthogonality(self, hx: np.ndarray, hz: np.ndarray) -> bool:
        for i in range(hx.shape[0]):
            for j in range(hz.shape[0]):
                acc = 0
                for k in range(hx.shape[1]):
                    acc = self.gf.add(acc, self.gf.mul(int(hx[i, k]), int(hz[j, k])))
                if acc != 0:
                    return False
        return True

    def _build_binary_parity(self, base: np.ndarray) -> sparse.csr_matrix:
        m, n = base.shape
        e = self.gf.e
        eP = e * self.P
        out = sparse.lil_matrix((m * eP, n * eP), dtype=np.uint8)

        for i in range(m):
            row0 = i * eP
            row1 = row0 + eP
            for j in range(n):
                a = int(base[i, j])
                if a == 0:
                    continue
                col0 = j * eP
                col1 = col0 + eP
                s = self.lift_table.get_shift(i, j)
                out[row0:row1, col0:col1] = kron_companion_circulant(self.gf, a, self.P, s)

        return out.tocsr()


PREDEFINED_CODES: dict[str, tuple[int, int, int]] = {}


def create_code(*args, **kwargs) -> CSSCode:
    """Compatibility factory returning CSSCode."""
    return CSSCode(*args, **kwargs)
