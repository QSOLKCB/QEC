"""Deterministic CSS construction from protograph pairs.

This module intentionally provides a minimal, invariant-first surface used by
repository tests and local deterministic construction flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

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
        # Validate GF(2^e) element range: entries must be in [0, q)
        q = self.gf.q
        if np.any(hx < 0) or np.any(hx >= q):
            raise ValueError(
                f"H_X_base entries must be in range [0, {q}), "
                f"got values in [{hx.min()}, {hx.max()}]"
            )
        if np.any(hz < 0) or np.any(hz >= q):
            raise ValueError(
                f"H_Z_base entries must be in range [0, {q}), "
                f"got values in [{hz.min()}, {hz.max()}]"
            )
        object.__setattr__(self, "H_X_base", hx)
        object.__setattr__(self, "H_Z_base", hz)


def _extract_proto_attrs(proto) -> tuple:
    """
    Extract (field, H_X_base, H_Z_base) from a protograph pair object.

    Supports both:
    - ProtographPair from this module (gf, H_X_base, H_Z_base)
    - ProtographPair from protograph.py (field, B_X, B_Z)
    """
    # Try the canonical protograph.py interface first (field, B_X, B_Z)
    if hasattr(proto, "field") and hasattr(proto, "B_X") and hasattr(proto, "B_Z"):
        return proto.field, proto.B_X, proto.B_Z
    # Fall back to this module's interface (gf, H_X_base, H_Z_base)
    if hasattr(proto, "gf") and hasattr(proto, "H_X_base") and hasattr(proto, "H_Z_base"):
        return proto.gf, proto.H_X_base, proto.H_Z_base
    raise TypeError(
        "proto must have either (field, B_X, B_Z) or (gf, H_X_base, H_Z_base) attributes"
    )


class CSSCode:
    """Deterministic CSS code lifted from a protograph pair."""

    def __init__(self, gf: GF2e, proto, P: int, seed: int = 0):
        if P < 1:
            raise ValueError("P must be >= 1")

        # Extract attributes from either protograph pair type
        proto_field, H_X_base, H_Z_base = _extract_proto_attrs(proto)

        if proto_field.e != gf.e or proto_field.q != gf.q or proto_field.poly != gf.poly:
            raise ValueError("proto field must match gf")

        # Validate GF(2^e) element range
        q = gf.q
        hx = np.asarray(H_X_base, dtype=np.int32)
        hz = np.asarray(H_Z_base, dtype=np.int32)
        if np.any(hx < 0) or np.any(hx >= q):
            raise ValueError(
                f"H_X_base entries must be in range [0, {q}), "
                f"got values in [{hx.min()}, {hx.max()}]"
            )
        if np.any(hz < 0) or np.any(hz >= q):
            raise ValueError(
                f"H_Z_base entries must be in range [0, {q}), "
                f"got values in [{hz.min()}, {hz.max()}]"
            )

        self.gf = gf
        self.proto = proto
        self.P = int(P)
        self.seed = int(seed)
        self.lift_table = LiftTable(self.P, self.seed)

        self.H_X = self._build_binary_parity(hx)
        self.H_Z = self._build_binary_parity(hz)

        if not self._verify_gf_orthogonality(hx, hz):
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
