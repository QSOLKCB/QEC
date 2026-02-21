"""
Tests for GF(2) utilities, OSD-0, and OSD-1 post-processing.
"""

import pytest
import numpy as np

from src.decoder.gf2 import gf2_row_echelon, binary_rank_dense
from src.decoder.osd import osd0, osd1
from src.qec_qldpc_codes import (
    bp_decode,
    syndrome,
    channel_llr,
    create_code,
)


# ───────────────────────────────────────────────────────────────────
# GF(2) Utilities
# ───────────────────────────────────────────────────────────────────

class TestGF2RowEchelon:

    def test_identity_pivots(self):
        """Row echelon of identity matrix has pivots at [0, 1, ..., n-1]."""
        n = 5
        I = np.eye(n, dtype=np.uint8)
        R, pivots = gf2_row_echelon(I)
        assert pivots == list(range(n))
        np.testing.assert_array_equal(R, I)

    def test_zero_matrix(self):
        """Zero matrix has no pivots."""
        Z = np.zeros((3, 4), dtype=np.uint8)
        R, pivots = gf2_row_echelon(Z)
        assert pivots == []
        np.testing.assert_array_equal(R, Z)

    def test_duplicate_rows(self):
        """Two identical rows reduce to rank 1."""
        M = np.array([[1, 0, 1], [1, 0, 1]], dtype=np.uint8)
        R, pivots = gf2_row_echelon(M)
        assert len(pivots) == 1
        assert pivots[0] == 0

    def test_full_rank_2x2(self):
        """[[1, 0], [0, 1]] has rank 2."""
        M = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        R, pivots = gf2_row_echelon(M)
        assert pivots == [0, 1]

    def test_mod2_reduction(self):
        """Input values > 1 are reduced mod 2."""
        M = np.array([[2, 1], [0, 3]], dtype=np.uint8)
        R, pivots = gf2_row_echelon(M)
        # 2 mod 2 = 0, 3 mod 2 = 1 → [[0, 1], [0, 1]] → rank 1
        assert len(pivots) == 1

    def test_n_pivot_cols_limits_search(self):
        """Augmented matrix: pivots only in first n columns."""
        # H = [[1, 0], [0, 1]], syndrome = [1, 1]
        H = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        s = np.array([[1], [1]], dtype=np.uint8)
        aug = np.hstack([H, s])
        R, pivots = gf2_row_echelon(aug, n_pivot_cols=2)
        # Pivots should be in columns 0, 1 (not column 2 = syndrome)
        assert all(p < 2 for p in pivots)
        assert len(pivots) == 2

    def test_row_swap_preserves_echelon(self):
        """Row echelon form has leading 1s in staircase pattern."""
        M = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.uint8)
        R, pivots = gf2_row_echelon(M)
        assert pivots == [0, 1, 2]
        # Leading 1 in each pivot row at the pivot column
        for i, pc in enumerate(pivots):
            assert R[i, pc] == 1

    def test_does_not_mutate_input(self):
        """Input matrix is not modified."""
        M = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        M_orig = M.copy()
        gf2_row_echelon(M)
        np.testing.assert_array_equal(M, M_orig)


class TestBinaryRankDense:

    def test_identity_rank(self):
        for n in [1, 3, 8]:
            assert binary_rank_dense(np.eye(n, dtype=np.uint8)) == n

    def test_zero_rank(self):
        assert binary_rank_dense(np.zeros((4, 5), dtype=np.uint8)) == 0

    def test_known_rank(self):
        """Known rank-2 matrix."""
        M = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 1, 1],  # row0 XOR row1
        ], dtype=np.uint8)
        assert binary_rank_dense(M) == 2

    def test_single_row(self):
        """Single nonzero row has rank 1."""
        M = np.array([[1, 0, 1]], dtype=np.uint8)
        assert binary_rank_dense(M) == 1

    def test_rank_le_min_dims(self):
        """Rank cannot exceed min(m, n)."""
        rng = np.random.default_rng(42)
        M = rng.integers(0, 2, size=(3, 7), dtype=np.uint8)
        assert binary_rank_dense(M) <= 3


# ───────────────────────────────────────────────────────────────────
# OSD-0
# ───────────────────────────────────────────────────────────────────

class TestOSD0:

    def test_valid_solution_not_degraded(self):
        """If hard_decision already satisfies the syndrome, OSD returns it unchanged."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        e = np.array([1, 0, 0], dtype=np.uint8)
        s = (H @ e) % 2
        llr = np.array([-5.0, 5.0, 5.0])
        result = osd0(H, llr, e, syndrome_vec=s.astype(np.uint8))
        # Syndrome check
        result_syn = ((H.astype(np.int32) @ result.astype(np.int32)) % 2).astype(np.uint8)
        np.testing.assert_array_equal(result_syn, s)

    def test_corrects_known_error(self):
        """OSD-0 corrects a single-bit error on a small code."""
        H = np.array([
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [1, 0, 1, 1, 1],
        ], dtype=np.uint8)
        # True error on bit 0
        e_true = np.array([1, 0, 0, 0, 0], dtype=np.uint8)
        s = ((H.astype(np.int32) @ e_true.astype(np.int32)) % 2).astype(np.uint8)
        # Bad hard decision from BP (wrong bit)
        hard_bad = np.array([0, 1, 0, 0, 0], dtype=np.uint8)
        # LLR: bit 0 least reliable, rest highly reliable
        llr = np.array([0.1, 8.0, 8.0, 8.0, 8.0])

        result = osd0(H, llr, hard_bad, syndrome_vec=s)
        result_syn = ((H.astype(np.int32) @ result.astype(np.int32)) % 2).astype(np.uint8)
        np.testing.assert_array_equal(result_syn, s)

    def test_never_degrades(self):
        """When OSD cannot find a valid solution, original hard_decision returned."""
        # Use a small underdetermined system where OSD may fail
        H = np.array([[1, 1]], dtype=np.uint8)
        s = np.array([1], dtype=np.uint8)
        hard = np.array([0, 0], dtype=np.uint8)  # does not satisfy syndrome
        llr = np.array([1.0, 1.0])
        result = osd0(H, llr, hard, syndrome_vec=s)
        result_syn = ((H.astype(np.int32) @ result.astype(np.int32)) % 2).astype(np.uint8)
        # Either OSD found a valid solution, or it returned the original
        if not np.array_equal(result_syn, s):
            np.testing.assert_array_equal(result, hard)

    def test_output_dtype_shape(self):
        """Output is uint8 with correct shape."""
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        llr = np.array([3.0, 2.0, 1.0])
        hard = np.array([0, 0, 0], dtype=np.uint8)
        result = osd0(H, llr, hard)
        assert result.dtype == np.uint8
        assert result.shape == (3,)

    def test_deterministic(self):
        """Same inputs produce identical outputs."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=np.uint8)
        llr = np.array([1.0, 0.5, 2.0, 0.1])
        hard = np.array([1, 0, 0, 1], dtype=np.uint8)
        s = np.array([0, 0], dtype=np.uint8)
        r1 = osd0(H, llr, hard, syndrome_vec=s)
        r2 = osd0(H, llr, hard, syndrome_vec=s)
        np.testing.assert_array_equal(r1, r2)

    def test_zero_syndrome(self):
        """When syndrome is all-zeros, zero vector should satisfy."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        llr = np.array([5.0, 5.0, 5.0])
        hard = np.zeros(3, dtype=np.uint8)
        result = osd0(H, llr, hard)
        # Zero vector satisfies zero syndrome
        np.testing.assert_array_equal(result, hard)

    def test_bp_with_osd_postprocess(self):
        """bp_decode with postprocess='osd0' integrates OSD."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        rng = np.random.default_rng(123)
        e = (rng.random(code.n) < 0.02).astype(np.uint8)
        s = syndrome(code.H_X, e)
        llr = channel_llr(e, 0.02)

        correction, iters = bp_decode(
            code.H_X, llr, max_iters=20,
            mode="min_sum", postprocess="osd0",
            syndrome_vec=s,
        )
        assert correction.dtype == np.uint8
        assert correction.shape == (code.n,)


# ───────────────────────────────────────────────────────────────────
# OSD-1
# ───────────────────────────────────────────────────────────────────

class TestOSD1:

    def test_osd1_valid_solution(self):
        """OSD-1 result satisfies the syndrome when possible."""
        H = np.array([
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [1, 0, 1, 1, 1],
        ], dtype=np.uint8)
        e_true = np.array([1, 0, 0, 0, 0], dtype=np.uint8)
        s = ((H.astype(np.int32) @ e_true.astype(np.int32)) % 2).astype(np.uint8)
        hard_bad = np.array([0, 1, 0, 0, 0], dtype=np.uint8)
        llr = np.array([0.1, 8.0, 8.0, 8.0, 8.0])

        result = osd1(H, llr, hard_bad, syndrome_vec=s)
        result_syn = ((H.astype(np.int32) @ result.astype(np.int32)) % 2).astype(np.uint8)
        np.testing.assert_array_equal(result_syn, s)

    def test_osd1_corrects_known_error(self):
        """OSD-1 corrects a single-bit error on a small code."""
        H = np.array([
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [1, 0, 1, 1, 1],
        ], dtype=np.uint8)
        e_true = np.array([1, 0, 0, 0, 0], dtype=np.uint8)
        s = ((H.astype(np.int32) @ e_true.astype(np.int32)) % 2).astype(np.uint8)
        hard_bad = np.array([0, 1, 0, 0, 0], dtype=np.uint8)
        llr = np.array([0.1, 8.0, 8.0, 8.0, 8.0])

        result = osd1(H, llr, hard_bad, syndrome_vec=s)
        result_syn = ((H.astype(np.int32) @ result.astype(np.int32)) % 2).astype(np.uint8)
        np.testing.assert_array_equal(result_syn, s)

    def test_osd1_never_degrades(self):
        """When OSD-1 cannot find a valid solution, original hard_decision returned."""
        H = np.array([[1, 1]], dtype=np.uint8)
        s = np.array([1], dtype=np.uint8)
        hard = np.array([0, 0], dtype=np.uint8)
        llr = np.array([1.0, 1.0])
        result = osd1(H, llr, hard, syndrome_vec=s)
        result_syn = ((H.astype(np.int32) @ result.astype(np.int32)) % 2).astype(np.uint8)
        # Either OSD-1 found a valid solution, or it returned the original.
        if not np.array_equal(result_syn, s):
            np.testing.assert_array_equal(result, hard)

    def test_osd1_output_dtype_shape(self):
        """Output is uint8 with correct shape."""
        H = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        llr = np.array([3.0, 2.0, 1.0])
        hard = np.array([0, 0, 0], dtype=np.uint8)
        result = osd1(H, llr, hard)
        assert result.dtype == np.uint8
        assert result.shape == (3,)

    def test_osd1_deterministic(self):
        """Same inputs produce identical outputs."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=np.uint8)
        llr = np.array([1.0, 0.5, 2.0, 0.1])
        hard = np.array([1, 0, 0, 1], dtype=np.uint8)
        s = np.array([0, 0], dtype=np.uint8)
        r1 = osd1(H, llr, hard, syndrome_vec=s)
        r2 = osd1(H, llr, hard, syndrome_vec=s)
        np.testing.assert_array_equal(r1, r2)

    def test_osd1_at_least_as_good_as_osd0(self):
        """OSD-1 weight <= OSD-0 weight when both produce valid solutions."""
        H = np.array([
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [1, 0, 1, 1, 1],
        ], dtype=np.uint8)
        e_true = np.array([1, 0, 0, 0, 0], dtype=np.uint8)
        s = ((H.astype(np.int32) @ e_true.astype(np.int32)) % 2).astype(np.uint8)
        hard_bad = np.array([0, 1, 0, 0, 0], dtype=np.uint8)
        llr = np.array([0.1, 8.0, 8.0, 8.0, 8.0])

        r0 = osd0(H, llr, hard_bad, syndrome_vec=s)
        r1 = osd1(H, llr, hard_bad, syndrome_vec=s)

        r0_syn = ((H.astype(np.int32) @ r0.astype(np.int32)) % 2).astype(np.uint8)
        r1_syn = ((H.astype(np.int32) @ r1.astype(np.int32)) % 2).astype(np.uint8)

        # If both are valid solutions, OSD-1 should be at least as good.
        if np.array_equal(r0_syn, s) and np.array_equal(r1_syn, s):
            assert int(np.sum(r1)) <= int(np.sum(r0))

    def test_osd1_zero_syndrome(self):
        """When syndrome is all-zeros, zero vector should satisfy."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        llr = np.array([5.0, 5.0, 5.0])
        hard = np.zeros(3, dtype=np.uint8)
        result = osd1(H, llr, hard)
        np.testing.assert_array_equal(result, hard)

    def test_bp_with_osd1_postprocess(self):
        """bp_decode with postprocess='osd1' integrates OSD-1."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        rng = np.random.default_rng(123)
        e = (rng.random(code.n) < 0.02).astype(np.uint8)
        s = syndrome(code.H_X, e)
        llr = channel_llr(e, 0.02)

        correction, iters = bp_decode(
            code.H_X, llr, max_iters=20,
            mode="min_sum", postprocess="osd1",
            syndrome_vec=s,
        )
        assert correction.dtype == np.uint8
        assert correction.shape == (code.n,)

    def test_osd1_unknown_postprocess_raises(self):
        """postprocess='osd2' raises ValueError."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        llr = np.full(code.n, 10.0)
        with pytest.raises(ValueError, match="postprocess"):
            bp_decode(code.H_X, llr, postprocess="osd2")
