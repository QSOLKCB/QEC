"""
Tests for the shared-circulant lifting module.

Run with:
    PYTHONPATH=src python -m pytest tests/test_lift.py -v
"""

import numpy as np
import pytest
from scipy import sparse

from qldpc.field import GF2e
from qldpc.lift import LiftTable, circulant_shift_matrix, kron_companion_circulant


# ── LiftTable tests ──────────────────────────────────────────────────


def test_shift_range_and_determinism():
    """Shifts are in [0, P) and reproducible across fresh instances."""
    P = 17
    seed = 123
    pairs = [(0, 0), (0, 1), (1, 0), (2, 3), (5, 7), (10, 15)]

    # First pass: collect shifts.
    t1 = LiftTable(P, seed)
    shifts_1 = [t1.get_shift(i, j) for i, j in pairs]

    # Range check.
    for s in shifts_1:
        assert 0 <= s < P, f"shift {s} out of range [0, {P})"

    # Second pass with a fresh instance (same seed, same request order).
    t2 = LiftTable(P, seed)
    shifts_2 = [t2.get_shift(i, j) for i, j in pairs]

    assert shifts_1 == shifts_2, "Shifts not deterministic across instances"

    # Repeat on the original instance — cached values must match.
    shifts_1_again = [t1.get_shift(i, j) for i, j in pairs]
    assert shifts_1 == shifts_1_again, "Cached shifts changed on re-query"


def test_shared_shift_reuse_same_edge():
    """Calling get_shift(i, j) twice on the same instance returns the same value."""
    t = LiftTable(P=31, seed=42)
    for i, j in [(0, 0), (3, 7), (99, 100)]:
        first = t.get_shift(i, j)
        second = t.get_shift(i, j)
        assert first == second, f"Shift changed for ({i},{j}): {first} vs {second}"


def test_shared_shift_reuse_across_consumers():
    """H_X and H_Z consumers sharing a LiftTable get identical shifts."""
    t = LiftTable(P=23, seed=999)

    # Positions that would appear in both B_X and B_Z of a protograph.
    shared_positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 3)]

    # Simulate H_X construction requesting shifts.
    hx_shifts = {pos: t.get_shift(*pos) for pos in shared_positions}

    # Simulate H_Z construction requesting the same positions.
    hz_shifts = {pos: t.get_shift(*pos) for pos in shared_positions}

    assert hx_shifts == hz_shifts, (
        "H_X and H_Z got different shifts for shared positions"
    )


def test_table_size():
    """table_size() tracks the number of cached row + column offsets."""
    t = LiftTable(P=7, seed=0)
    assert t.table_size() == 0

    t.get_shift(0, 0)  # row 0 + col 0 → 2 offsets
    assert t.table_size() == 2

    t.get_shift(0, 0)  # repeat — no new offsets
    assert t.table_size() == 2

    t.get_shift(1, 2)  # row 1 + col 2 → 2 new offsets
    assert t.table_size() == 4

    t.get_shift(1, 0)  # row 1 cached, col 0 cached → no new
    assert t.table_size() == 4

    t.get_shift(3, 4)  # row 3 + col 4 → 2 new offsets
    assert t.table_size() == 6


def test_additive_shift_structure():
    """Shift differences between rows are constant across columns."""
    t = LiftTable(P=13, seed=77)

    # Query a grid of (i, j) pairs.
    rows = [0, 1, 2]
    cols = [0, 1, 2, 3]
    shifts = {(i, j): t.get_shift(i, j) for i in rows for j in cols}

    # For any two rows i, k: s(i,j) - s(k,j) must be the same for all j.
    for i in rows:
        for k in rows:
            diffs = [(shifts[(i, j)] - shifts[(k, j)]) % t.P for j in cols]
            assert len(set(diffs)) == 1, (
                f"Row diff ({i},{k}) not constant: {diffs}"
            )


def test_invalid_lifttable_P_raises_value_error():
    """LiftTable constructor validates P and raises ValueError for invalid P."""
    with pytest.raises(ValueError):
        LiftTable(0, seed=42)
    with pytest.raises(ValueError):
        LiftTable(-1, seed=42)


# ── circulant_shift_matrix tests ─────────────────────────────────────


def test_circulant_matrix_is_permutation():
    """Every circulant matrix is a valid permutation matrix."""
    P = 11
    for s in [0, 1, 5, P - 1]:
        mat = circulant_shift_matrix(P, s)
        dense = mat.toarray().astype(int)

        row_sums = dense.sum(axis=1)
        col_sums = dense.sum(axis=0)

        assert np.all(row_sums == 1), f"s={s}: row sums {row_sums}"
        assert np.all(col_sums == 1), f"s={s}: col sums {col_sums}"


def test_circulant_identity_at_zero():
    """Shift 0 produces the identity matrix."""
    P = 7
    mat = circulant_shift_matrix(P, 0)
    assert np.array_equal(mat.toarray(), np.eye(P, dtype=np.uint8))


def test_circulant_is_sparse():
    """Output is a sparse CSR matrix."""
    mat = circulant_shift_matrix(13, 3)
    assert sparse.issparse(mat)
    assert isinstance(mat, sparse.csr_matrix)


def test_circulant_shift_modulo_behavior():
    """Shifts outside [0, P) are reduced mod P, including negatives."""
    P = 7
    for s_raw in [P, 2 * P + 1, -1, -P - 2]:
        mat = circulant_shift_matrix(P, s_raw)
        expected = circulant_shift_matrix(P, s_raw % P)
        assert np.array_equal(mat.toarray(), expected.toarray()), (
            f"s={s_raw}: modulo reduction mismatch"
        )


def test_circulant_invalid_P_raises():
    """circulant_shift_matrix validates P >= 1."""
    with pytest.raises(ValueError):
        circulant_shift_matrix(0, 0)
    with pytest.raises(ValueError):
        circulant_shift_matrix(-1, 0)


# ── kron_companion_circulant tests ───────────────────────────────────


def test_kron_companion_circulant_matches_companion_kron_pi_for_nontrivial_a():
    """For nontrivial a, kron_companion_circulant equals C(a) ⊗ π_s."""
    gf = GF2e(e=2)
    P = 5
    for a in [2, 3]:
        for s in [0, 2, P - 1]:
            M = kron_companion_circulant(gf, a, P, s).toarray()
            C = sparse.csr_matrix(gf.companion_matrix(a))
            Pi = circulant_shift_matrix(P, s)
            K = sparse.kron(C, Pi).toarray()
            assert np.array_equal(M, K), (
                f"a={a}, s={s}: kron does not match C(a) ⊗ π_s"
            )


def test_kron_companion_circulant_shapes_and_identity_cases():
    """Check shapes and special-case outputs for a=0, a=1."""
    gf = GF2e(e=3)
    e = gf.e
    P = 7
    eP = e * P

    # a=0: all-zero matrix.
    M0 = kron_companion_circulant(gf, 0, P, 3)
    assert M0.shape == (eP, eP)
    assert M0.nnz == 0

    # a=1, s=0: identity I_{eP}.
    M1 = kron_companion_circulant(gf, 1, P, 0)
    assert M1.shape == (eP, eP)
    assert np.array_equal(M1.toarray(), np.eye(eP, dtype=np.uint8))

    # a=1, general s: equals I_e ⊗ π_s.
    for s in [1, 3, P - 1]:
        M = kron_companion_circulant(gf, 1, P, s)
        I_e = sparse.eye(e, dtype=np.uint8, format="csr")
        pi_s = circulant_shift_matrix(P, s)
        expected = sparse.kron(I_e, pi_s, format="csr").astype(np.uint8)
        assert np.array_equal(M.toarray(), expected.toarray()), (
            f"a=1, s={s}: kron does not match I_e ⊗ π_s"
        )


def test_kron_companion_circulant_output_is_sparse():
    """Output is a sparse CSR matrix with correct dtype."""
    gf = GF2e(e=3)
    M = kron_companion_circulant(gf, 3, 11, 5)
    assert sparse.issparse(M)
    assert isinstance(M, sparse.csr_matrix)
    assert M.dtype == np.uint8


def test_kron_companion_circulant_invalid_P_raises():
    """kron_companion_circulant validates P >= 1."""
    gf = GF2e(e=2)
    with pytest.raises(ValueError):
        kron_companion_circulant(gf, 1, 0, 0)
    with pytest.raises(ValueError):
        kron_companion_circulant(gf, 1, -1, 0)
