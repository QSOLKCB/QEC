"""
Tests for the deterministic decimation module.
"""

import pytest
import numpy as np

from decoder.decimation import decimate, decimation_round, LLR_CLAMP_FACTOR
from qec_qldpc_codes import (
    bp_decode,
    syndrome,
    channel_llr,
    create_code,
)


# ───────────────────────────────────────────────────────────────────
# decimate()
# ───────────────────────────────────────────────────────────────────

class TestDecimate:

    def test_threshold_commits_high_confidence(self):
        """Variables with |belief| >= threshold are committed."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        beliefs = np.array([5.0, -3.0, 0.5])
        hard = np.array([0, 1, 0], dtype=np.uint8)
        threshold = 2.0

        new_hard, committed, forced = decimate(H, beliefs, threshold, hard)
        # |5.0| >= 2 → committed, belief >= 0 → 0
        assert committed[0] is np.True_
        assert new_hard[0] == 0
        # |-3.0| >= 2 → committed, belief < 0 → 1
        assert committed[1] is np.True_
        assert new_hard[1] == 1
        # |0.5| < 2 → NOT committed
        assert committed[2] is np.False_

    def test_tie_breaking_by_index(self):
        """Equal |belief| at threshold: committed in ascending index order."""
        H = np.eye(3, dtype=np.uint8)
        beliefs = np.array([2.0, -2.0, 2.0])
        hard = np.zeros(3, dtype=np.uint8)
        threshold = 2.0

        new_hard, committed, forced = decimate(H, beliefs, threshold, hard)
        # All three have |belief| == threshold → all committed
        assert np.all(committed)
        assert new_hard[0] == 0  # belief >= 0
        assert new_hard[1] == 1  # belief < 0
        assert new_hard[2] == 0  # belief >= 0

    def test_no_commit_below_threshold(self):
        """Variables with |belief| < threshold are NOT committed."""
        H = np.eye(3, dtype=np.uint8)
        beliefs = np.array([0.1, -0.2, 0.3])
        hard = np.zeros(3, dtype=np.uint8)
        threshold = 1.0

        new_hard, committed, forced = decimate(H, beliefs, threshold, hard)
        assert not np.any(committed)

    def test_already_committed_preserved(self):
        """Pre-committed variables remain committed."""
        H = np.eye(3, dtype=np.uint8)
        beliefs = np.array([0.1, -5.0, 0.3])
        hard = np.array([1, 0, 0], dtype=np.uint8)
        committed = np.array([True, False, False])
        threshold = 1.0

        new_hard, new_committed, forced = decimate(
            H, beliefs, threshold, hard, committed=committed,
        )
        assert new_committed[0]  # was already committed
        assert new_committed[1]  # |-5.0| >= 1.0
        assert not new_committed[2]  # |0.3| < 1.0

    def test_deterministic(self):
        """Same inputs produce identical outputs."""
        H = np.array([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=np.uint8)
        beliefs = np.array([3.0, -1.0, 2.5, -0.5])
        hard = np.zeros(4, dtype=np.uint8)
        threshold = 2.0

        r1 = decimate(H, beliefs, threshold, hard)
        r2 = decimate(H, beliefs, threshold, hard)
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])
        np.testing.assert_array_equal(r1[2], r2[2])

    def test_output_dtypes(self):
        """new_hard is uint8, new_committed is bool."""
        H = np.eye(2, dtype=np.uint8)
        beliefs = np.array([5.0, -5.0])
        hard = np.zeros(2, dtype=np.uint8)

        new_hard, committed, forced = decimate(H, beliefs, 1.0, hard)
        assert new_hard.dtype == np.uint8
        assert committed.dtype == bool
        assert forced.dtype == int

    def test_forced_values_shape_no_peel(self):
        """Without peeling, forced_values has shape (0, 2)."""
        H = np.eye(2, dtype=np.uint8)
        beliefs = np.array([5.0, -5.0])
        hard = np.zeros(2, dtype=np.uint8)

        _, _, forced = decimate(H, beliefs, 1.0, hard, peel=False)
        assert forced.shape == (0, 2)


# ───────────────────────────────────────────────────────────────────
# Peeling
# ───────────────────────────────────────────────────────────────────

class TestPeeling:

    def test_degree1_peeling(self):
        """Degree-1 checks propagate forced values."""
        # H = [[1, 1, 0], [0, 0, 1]]
        # After committing variable 0, check 0 has degree 1 (only var 1 left).
        H = np.array([[1, 1, 0], [0, 0, 1]], dtype=np.uint8)
        beliefs = np.array([5.0, 0.1, 0.1])  # Only var 0 above threshold
        hard = np.array([0, 0, 0], dtype=np.uint8)
        s = np.array([1, 0], dtype=np.uint8)  # syndrome check 0 = 1
        threshold = 1.0

        new_hard, committed, forced = decimate(
            H, beliefs, threshold, hard, peel=True, syndrome_vec=s,
        )
        # Var 0 committed to 0 (belief >= 0).
        assert committed[0]
        assert new_hard[0] == 0
        # Check 0: H[0,:] @ x = s[0] → x[0] + x[1] = 1 → x[1] = 1
        assert committed[1]
        assert new_hard[1] == 1
        # forced should contain (1, 1)
        assert forced.shape[0] >= 1
        assert any(forced[i, 0] == 1 and forced[i, 1] == 1
                    for i in range(forced.shape[0]))

    def test_no_peeling_when_disabled(self):
        """peel=False skips peeling even when degree-1 checks exist."""
        H = np.array([[1, 1, 0], [0, 0, 1]], dtype=np.uint8)
        beliefs = np.array([5.0, 0.1, 0.1])
        hard = np.zeros(3, dtype=np.uint8)
        s = np.array([1, 0], dtype=np.uint8)
        threshold = 1.0

        new_hard, committed, forced = decimate(
            H, beliefs, threshold, hard, peel=False, syndrome_vec=s,
        )
        assert committed[0]      # above threshold
        assert not committed[1]   # not peeled
        assert forced.shape == (0, 2)

    def test_peeling_order_deterministic(self):
        """Peeling processes checks in ascending index order."""
        H = np.array([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], dtype=np.uint8)
        beliefs = np.array([5.0, 0.1, 5.0, 0.1])
        hard = np.zeros(4, dtype=np.uint8)
        s = np.array([0, 0], dtype=np.uint8)
        threshold = 1.0

        r1 = decimate(H, beliefs, threshold, hard, peel=True, syndrome_vec=s)
        r2 = decimate(H, beliefs, threshold, hard, peel=True, syndrome_vec=s)
        np.testing.assert_array_equal(r1[0], r2[0])
        np.testing.assert_array_equal(r1[1], r2[1])
        np.testing.assert_array_equal(r1[2], r2[2])


# ───────────────────────────────────────────────────────────────────
# decimation_round()
# ───────────────────────────────────────────────────────────────────

class TestDecimationRound:

    def test_basic_round(self):
        """decimation_round runs BP -> commit -> BP."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        rng = np.random.default_rng(99)
        e = (rng.random(code.n) < 0.02).astype(np.uint8)
        s = syndrome(code.H_X, e)
        llr = channel_llr(e, 0.02)

        correction, total_iters, rounds = decimation_round(
            code.H_X, llr, threshold=3.0,
            bp_kwargs={"max_iters": 20, "mode": "min_sum"},
            max_rounds=3, syndrome_vec=s,
        )
        assert correction.dtype == np.uint8
        assert correction.shape == (code.n,)
        assert total_iters >= 1
        assert 1 <= rounds <= 3

    def test_max_rounds_respected(self):
        """Does not exceed max_rounds."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        llr = np.full(code.n, 0.5)  # uncertain → many rounds
        s = np.zeros(code.m_X, dtype=np.uint8)

        _, _, rounds = decimation_round(
            code.H_X, llr, threshold=10.0,
            bp_kwargs={"max_iters": 5, "mode": "min_sum"},
            max_rounds=2, syndrome_vec=s,
        )
        assert rounds <= 2

    def test_deterministic(self):
        """Same inputs produce identical outputs."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        rng = np.random.default_rng(77)
        e = (rng.random(code.n) < 0.02).astype(np.uint8)
        s = syndrome(code.H_X, e)
        llr = channel_llr(e, 0.02)

        kw = {"max_iters": 15, "mode": "min_sum"}
        r1 = decimation_round(code.H_X, llr, 3.0, kw, max_rounds=3, syndrome_vec=s)
        r2 = decimation_round(code.H_X, llr, 3.0, kw, max_rounds=3, syndrome_vec=s)
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]
        assert r1[2] == r2[2]

    def test_with_different_bp_modes(self):
        """Works with sum_product mode."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        rng = np.random.default_rng(55)
        e = (rng.random(code.n) < 0.01).astype(np.uint8)
        s = syndrome(code.H_X, e)
        llr = channel_llr(e, 0.01)

        correction, _, _ = decimation_round(
            code.H_X, llr, threshold=5.0,
            bp_kwargs={"max_iters": 20, "mode": "sum_product"},
            max_rounds=2, syndrome_vec=s,
        )
        assert correction.dtype == np.uint8

    def test_with_llr_history(self):
        """decimation_round handles bp_decode returning 3-tuple (llr_history > 0)."""
        code = create_code('rate_0.50', lifting_size=8, seed=42)
        rng = np.random.default_rng(88)
        e = (rng.random(code.n) < 0.02).astype(np.uint8)
        s = syndrome(code.H_X, e)
        llr = channel_llr(e, 0.02)

        correction, total_iters, rounds = decimation_round(
            code.H_X, llr, threshold=3.0,
            bp_kwargs={"max_iters": 20, "mode": "min_sum", "llr_history": 5},
            max_rounds=3, syndrome_vec=s,
        )
        assert correction.dtype == np.uint8
        assert correction.shape == (code.n,)
        assert total_iters >= 1
        assert 1 <= rounds <= 3

    def test_clamp_factor_constant(self):
        """LLR_CLAMP_FACTOR is a positive number."""
        assert LLR_CLAMP_FACTOR > 0
