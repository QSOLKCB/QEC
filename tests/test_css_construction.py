"""
Tests for CSS code construction from protograph pairs.

Run with:
    PYTHONPATH=src python -m pytest tests/test_css_construction.py -v
"""

import numpy as np
import pytest
from scipy import sparse

from qldpc.field import GF2e
from qldpc.lift import LiftTable
from qldpc.css_code import CSSCode, ProtographPair
from qldpc.invariants import ConstructionInvariantError


def _gf4_single_row_pair():
    """
    1×2 GF(4)-orthogonal pair.

    H_X = [[1, 2]], H_Z = [[2, 1]].
    Check: 1·2 ⊕ 2·1 = 2 ⊕ 2 = 0 over GF(4). ✓
    """
    gf = GF2e(e=2)
    H_X_base = np.array([[1, 2]], dtype=np.int32)
    H_Z_base = np.array([[2, 1]], dtype=np.int32)
    return gf, ProtographPair(gf, H_X_base, H_Z_base)


def _gf4_multi_row_disjoint_pair():
    """
    2×4 multi-row GF(4)-orthogonal pair with disjoint row support.

    B_X = B_Z = [[1, 1, 0, 0],
                  [0, 0, 2, 2]]

    Self-orthogonality:
      row0·row0 = 1·1 + 1·1 + 0 + 0 = 0 ✓
      row0·row1 = 0 + 0 + 0 + 0 = 0 ✓  (disjoint support)
      row1·row1 = 0 + 0 + 2·2 + 2·2 = 3 + 3 = 0 ✓
    """
    gf = GF2e(e=2)
    B = np.array([[1, 1, 0, 0],
                   [0, 0, 2, 2]], dtype=np.int32)
    return gf, ProtographPair(gf, B, B.copy())


def _gf4_multi_row_overlapping_pair():
    """
    2×4 multi-row GF(4)-orthogonal pair with OVERLAPPING row support.

    B_X = B_Z = [[1, 1, 2, 2],
                  [2, 2, 3, 3]]

    Self-orthogonality:
      row0·row0 = 1+1+2·2+2·2 = 0+3+3 = 0 ✓
      row0·row1 = 1·2+1·2+2·3+2·3 = 2+2+1+1 = 0 ✓  (overlapping!)
      row1·row1 = 2·2+2·2+3·3+3·3 = 3+3+2+2 = 0 ✓

    This exercises the additive shift structure s(i,j) = (r_i + c_j) % P
    which keeps row-shift differences constant across columns, enabling
    orthogonality even with overlapping row support.
    """
    gf = GF2e(e=2)
    B = np.array([[1, 1, 2, 2],
                   [2, 2, 3, 3]], dtype=np.int32)
    return gf, ProtographPair(gf, B, B.copy())


# ── Test 1: Orthogonality ────────────────────────────────────────────


def test_css_orthogonality_small():
    """
    Construct CSS codes from both a single-row and a multi-row protograph;
    verify binary orthogonality holds for each.
    """
    for label, builder, P in [
        ("1-row", _gf4_single_row_pair, 5),
        ("2-row-disjoint", _gf4_multi_row_disjoint_pair, 7),
        ("2-row-overlapping", _gf4_multi_row_overlapping_pair, 7),
    ]:
        gf, proto = builder()
        e = gf.e
        n_cols = proto.H_X_base.shape[1]
        m_x = proto.H_X_base.shape[0]
        m_z = proto.H_Z_base.shape[0]

        # Must not raise.
        code = CSSCode(gf, proto, P, seed=42)

        # Shape: (m * e * P) × (n * e * P)
        assert code.H_X.shape == (m_x * e * P, n_cols * e * P), f"{label}"
        assert code.H_Z.shape == (m_z * e * P, n_cols * e * P), f"{label}"

        # Explicit independent orthogonality verification.
        product = code.H_X.astype(np.int32) @ code.H_Z.astype(np.int32).T
        if sparse.issparse(product):
            product = product.toarray()
        assert np.all(product % 2 == 0), (
            f"{label}: H_X @ H_Z^T != 0 mod 2"
        )


# ── Test 2: Deterministic construction ───────────────────────────────


def test_deterministic_construction():
    """Same parameters and seed produce identical H_X and H_Z."""
    gf, proto = _gf4_multi_row_overlapping_pair()
    P = 7
    seed = 123

    code1 = CSSCode(gf, proto, P, seed)
    code2 = CSSCode(gf, proto, P, seed)

    # Exact sparse equality via dense comparison (small matrices).
    assert np.array_equal(code1.H_X.toarray(), code2.H_X.toarray())
    assert np.array_equal(code1.H_Z.toarray(), code2.H_Z.toarray())


# ── Test 3: Shared LiftTable ────────────────────────────────────────


def test_shared_lifttable_instance():
    """CSSCode stores one LiftTable; both H_X and H_Z share it."""
    gf, proto = _gf4_multi_row_overlapping_pair()
    code = CSSCode(gf, proto, P=5, seed=0)

    assert isinstance(code.lift_table, LiftTable)

    # Shifts for positions used by both base matrices are cached once.
    s1 = code.lift_table.get_shift(0, 0)
    s2 = code.lift_table.get_shift(0, 0)
    assert s1 == s2

    s3 = code.lift_table.get_shift(0, 1)
    s4 = code.lift_table.get_shift(0, 1)
    assert s3 == s4


# ── Test 4: Invalid orthogonality raises ─────────────────────────────


def test_invalid_orthogonality_raises():
    """Base matrices that violate GF-level orthogonality must raise."""
    gf = GF2e(e=2)

    # H_X = [[1, 1]], H_Z = [[1, 0]].
    # GF check: 1·1 + 1·0 = 1 ≠ 0.  Binary orthogonality also fails.
    H_X_base = np.array([[1, 1]], dtype=np.int32)
    H_Z_base = np.array([[1, 0]], dtype=np.int32)
    proto = ProtographPair(gf, H_X_base, H_Z_base)

    with pytest.raises(ConstructionInvariantError):
        CSSCode(gf, proto, P=5, seed=42)
