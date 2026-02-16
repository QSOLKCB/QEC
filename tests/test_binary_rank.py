"""
Tests for binary_rank over GF(2) and CSS rank / dimension validation.

Run with:
    PYTHONPATH=src python -m pytest tests/test_binary_rank.py -v
"""

import numpy as np
import pytest
from scipy import sparse

from qldpc.invariants import binary_rank
from qldpc.field import GF2e
from qldpc.css_code import CSSCode, ProtographPair


# ── Test 1: Simple rank cases ─────────────────────────────────────────


def test_binary_rank_simple_cases():
    """Identity, zero, and a known small matrix."""
    # 4×4 identity → rank 4
    I4 = sparse.eye(4, format="csr", dtype=np.uint8)
    assert binary_rank(I4) == 4

    # 3×5 zero matrix → rank 0
    Z = sparse.csr_matrix((3, 5), dtype=np.uint8)
    assert binary_rank(Z) == 0

    # Known 3×3 matrix over GF(2):
    #   [[1, 1, 0],
    #    [0, 1, 1],
    #    [1, 0, 1]]
    # Row 2 = Row 0 XOR Row 1 → rank 2
    A = sparse.csr_matrix(
        np.array([[1, 1, 0],
                  [0, 1, 1],
                  [1, 0, 1]], dtype=np.uint8)
    )
    assert binary_rank(A) == 2

    # COO input should also work
    A_coo = A.tocoo()
    assert binary_rank(A_coo) == 2


# ── Test 2: CSS rank and dimension for a small P ──────────────────────


def test_css_rank_and_dimension_small_P():
    """
    For a small valid CSS construction, verify k = n - rank(H_X) - rank(H_Z) >= 1.

    Uses the 2×4 overlapping GF(4) protograph with P=7.
    B_X = B_Z = [[1, 1, 2, 2],
                  [2, 2, 3, 3]]
    """
    gf = GF2e(e=2)
    B = np.array([[1, 1, 2, 2],
                  [2, 2, 3, 3]], dtype=np.int32)
    proto = ProtographPair(gf, B, B.copy())
    code = CSSCode(gf, proto, P=7, seed=42)

    n = code.H_X.shape[1]
    rx = binary_rank(code.H_X)
    rz = binary_rank(code.H_Z)
    k = n - rx - rz

    assert rx > 0, "H_X should have nonzero rank"
    assert rz > 0, "H_Z should have nonzero rank"
    assert k >= 1, (
        f"Expected k >= 1 but got k={k} "
        f"(n={n}, rank_X={rx}, rank_Z={rz})"
    )


# ── Test 3: Invalid / over-constrained edge case (k == 0) ────────────


def test_invalid_rank_edge_case():
    """
    Construct a trivial over-constrained case where k = n - rank(H_X) - rank(H_Z) == 0.

    Take H_X = H_Z = I_n.  Then:
        rank(H_X) = rank(H_Z) = n  →  k = n - n - n = -n,
    but since rows of H_X and H_Z span the full space the code encodes
    zero logical qubits.  We clamp k = max(0, ...) and verify k == 0.

    Note: This is NOT a valid CSS code (orthogonality fails for I @ I^T = I ≠ 0),
    so we test binary_rank directly rather than via CSSCode.
    """
    n = 6
    I_n = sparse.eye(n, format="csr", dtype=np.uint8)

    rx = binary_rank(I_n)
    rz = binary_rank(I_n)
    k = max(0, n - rx - rz)

    assert rx == n
    assert rz == n
    assert k == 0, f"Expected k == 0 for over-constrained case, got k={k}"
