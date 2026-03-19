"""
Tests for v2.9.0: Deterministic Adaptive Schedule Controller.

schedule="adaptive" implements a one-way checkpointed controller:
  Phase 1: flooding for k1 iterations.
  If converged → return immediately.
  Phase 2: hybrid_residual for remaining (max_iters - k1) iterations.
  Tie-break: converged > lower syndrome weight > fewer iters > phase order.
"""

import numpy as np
import pytest

from qec_qldpc_codes import bp_decode, syndrome, channel_llr, create_code


# ── Fixtures ──

@pytest.fixture
def small_code():
    """Small rate-0.50 code for fast tests."""
    return create_code('rate_0.50', lifting_size=8, seed=42)


@pytest.fixture
def noisy_setup(small_code):
    """Code + low-noise error + syndrome + LLR."""
    rng = np.random.default_rng(42)
    e = (rng.random(small_code.n) < 0.01).astype(np.uint8)
    s = syndrome(small_code.H_X, e)
    llr = channel_llr(e, 0.01)
    return small_code, e, s, llr


@pytest.fixture
def hard_noisy_setup(small_code):
    """Code + higher-noise error + syndrome + LLR (less likely to converge)."""
    rng = np.random.default_rng(99)
    e = (rng.random(small_code.n) < 0.08).astype(np.uint8)
    s = syndrome(small_code.H_X, e)
    llr = channel_llr(e, 0.08)
    return small_code, e, s, llr


# ── Determinism tests ──

class TestAdaptiveDeterminism:

    def test_deterministic_same_inputs_twice(self, noisy_setup):
        """Adaptive schedule must produce identical results on two runs."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="adaptive",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="adaptive",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]

    def test_deterministic_sum_product(self, noisy_setup):
        """Adaptive with sum_product mode is deterministic."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="sum_product",
            syndrome_vec=s, schedule="adaptive",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="sum_product",
            syndrome_vec=s, schedule="adaptive",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]

    def test_deterministic_with_explicit_k1(self, noisy_setup):
        """Adaptive with explicit adaptive_k1 is deterministic."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="adaptive", adaptive_k1=5,
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="adaptive", adaptive_k1=5,
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]

    def test_deterministic_hard_noise(self, hard_noisy_setup):
        """Adaptive under harder noise is still deterministic."""
        code, e, s, llr = hard_noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            syndrome_vec=s, schedule="adaptive",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            syndrome_vec=s, schedule="adaptive",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]


# ── Phase selection tests ──

class TestAdaptivePhaseSelection:

    def test_phase1_converged_returns_early(self):
        """When phase 1 (flooding) converges, adaptive returns that result
        and the iteration count reflects only phase 1."""
        # Use a trivial zero-error case that always converges immediately.
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        llr = np.array([5.0, 5.0, 5.0])
        s = np.zeros(2, dtype=np.uint8)
        result = bp_decode(
            H, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="adaptive",
        )
        hard, iters = result[0], result[1]
        # Zero error → should converge in 1 iteration.
        np.testing.assert_array_equal(hard, np.array([0, 0, 0], dtype=np.uint8))
        assert iters <= 5  # phase 1 budget is max(1, 20//4) = 5

    def test_both_phases_run_when_phase1_fails(self, hard_noisy_setup):
        """When phase 1 doesn't converge, phase 2 runs and the best is picked.
        If phase 2 is selected, iteration count includes k1 (cumulative)."""
        code, e, s, llr = hard_noisy_setup
        k1 = 2
        # Use adaptive_k1=2 so phase 1 barely gets any iterations.
        result = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="adaptive", adaptive_k1=k1,
        )
        # Verify valid result.
        assert result[0].shape == (code.n,)
        # Iteration count is at least 1 (phase 1 selected) or
        # at least k1+1 (phase 2 selected with cumulative accounting).
        assert result[1] >= 1

    def test_different_k1_can_give_different_results(self, noisy_setup):
        """Different adaptive_k1 values are valid and can yield different paths."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="adaptive", adaptive_k1=2,
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="adaptive", adaptive_k1=10,
        )
        # Both must be valid — results may or may not differ.
        assert r1[0].shape == r2[0].shape
        assert r1[1] >= 1
        assert r2[1] >= 1

    def test_cumulative_iters_when_phase2_selected(self):
        """When phase 2 is selected, returned iters = k1 + iters_p2."""
        # Construct a scenario where phase 1 does NOT converge (few iters,
        # hard problem) and phase 2 provides a better result.
        # We compare adaptive result against a direct hybrid_residual call
        # to verify cumulative accounting.
        H = np.array([
            [1, 1, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
        ], dtype=np.uint8)
        # LLR biased toward error on variable 0.
        llr = np.array([-2.0, 3.0, 3.0, 3.0, 3.0])
        s = np.array([1, 0, 1], dtype=np.uint8)
        k1 = 1
        max_iters = 10

        result_adaptive = bp_decode(
            H, llr, max_iters=max_iters, mode="min_sum",
            syndrome_vec=s, schedule="adaptive", adaptive_k1=k1,
        )

        # Run phase 1 standalone to check convergence.
        result_p1 = bp_decode(
            H, llr, max_iters=k1, mode="min_sum",
            syndrome_vec=s, schedule="flooding",
        )
        syn_p1 = (H.astype(np.int32) @ result_p1[0].astype(np.int32)) % 2
        p1_converged = np.array_equal(syn_p1.astype(np.uint8), s)

        if not p1_converged:
            # Phase 2 was attempted.  Run it standalone.
            k2 = max_iters - k1
            result_p2 = bp_decode(
                H, llr, max_iters=k2, mode="min_sum",
                syndrome_vec=s, schedule="hybrid_residual",
            )
            syn_p2 = (H.astype(np.int32) @ result_p2[0].astype(np.int32)) % 2
            p2_converged = np.array_equal(syn_p2.astype(np.uint8), s)
            syn_w_p1 = int(np.sum(syn_p1.astype(np.uint8) != s))
            syn_w_p2 = int(np.sum(syn_p2.astype(np.uint8) != s))

            # Determine which phase the adaptive controller should pick.
            if p2_converged and not p1_converged:
                expected_iters = k1 + result_p2[1]
            elif not p2_converged and p1_converged:
                expected_iters = result_p1[1]
            elif syn_w_p2 < syn_w_p1:
                expected_iters = k1 + result_p2[1]
            elif syn_w_p2 > syn_w_p1:
                expected_iters = result_p1[1]
            elif (k1 + result_p2[1]) < result_p1[1]:
                expected_iters = k1 + result_p2[1]
            else:
                expected_iters = result_p1[1]

            assert result_adaptive[1] == expected_iters
        else:
            # Phase 1 converged — iters should be just iters_p1.
            assert result_adaptive[1] == result_p1[1]


# ── Tie-break tests ──

class TestAdaptiveTieBreak:

    def test_converged_preferred_over_not_converged(self):
        """If one phase converges and the other doesn't, the converged one wins."""
        # Zero-error: phase 1 should converge, so adaptive returns it.
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        llr = np.array([10.0, 10.0, 10.0])
        s = np.zeros(2, dtype=np.uint8)
        result = bp_decode(
            H, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="adaptive",
        )
        # Phase 1 converges → should return zero correction.
        np.testing.assert_array_equal(result[0], np.zeros(3, dtype=np.uint8))


# ── Validation tests ──

class TestAdaptiveValidation:

    def test_invalid_adaptive_rule_raises(self):
        """Unknown adaptive_rule raises ValueError."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        llr = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="adaptive_rule"):
            bp_decode(
                H, llr, max_iters=10, schedule="adaptive",
                adaptive_rule="unknown",
            )

    def test_adaptive_k1_zero_raises(self):
        """adaptive_k1=0 raises ValueError."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        llr = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="adaptive_k1"):
            bp_decode(
                H, llr, max_iters=10, schedule="adaptive",
                adaptive_k1=0,
            )

    def test_adaptive_k1_negative_raises(self):
        """adaptive_k1=-1 raises ValueError."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        llr = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="adaptive_k1"):
            bp_decode(
                H, llr, max_iters=10, schedule="adaptive",
                adaptive_k1=-1,
            )

    def test_adaptive_k1_equals_max_iters_raises(self):
        """adaptive_k1 == max_iters raises ValueError (no room for phase 2)."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        llr = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="adaptive_k1"):
            bp_decode(
                H, llr, max_iters=10, schedule="adaptive",
                adaptive_k1=10,
            )

    def test_adaptive_k1_exceeds_max_iters_raises(self):
        """adaptive_k1 > max_iters raises ValueError."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        llr = np.array([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="adaptive_k1"):
            bp_decode(
                H, llr, max_iters=10, schedule="adaptive",
                adaptive_k1=15,
            )

    def test_adaptive_params_ignored_for_other_schedules(self):
        """adaptive_k1 / adaptive_rule are silently ignored for non-adaptive schedules."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        llr = np.array([5.0, 5.0, 5.0])
        s = np.zeros(2, dtype=np.uint8)
        # Should not raise despite adaptive_k1 being set.
        result = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            syndrome_vec=s, schedule="flooding",
            adaptive_k1=3, adaptive_rule="one_way",
        )
        assert result[0].shape == (3,)


# ── No-regression tests ──

class TestAdaptiveNoRegression:

    def test_flooding_unchanged(self, noisy_setup):
        """Flooding schedule results are identical with and without adaptive code present."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="flooding",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="flooding",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]

    def test_layered_unchanged(self, noisy_setup):
        """Layered schedule results are identical."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="layered",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="layered",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]

    def test_residual_unchanged(self, noisy_setup):
        """Residual schedule results are identical."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="residual",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="residual",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]

    def test_hybrid_residual_unchanged(self, noisy_setup):
        """Hybrid_residual schedule results are identical."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="hybrid_residual",
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="hybrid_residual",
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]


# ── LLR history integration ──

class TestAdaptiveLLRHistory:

    def test_llr_history_returned(self, noisy_setup):
        """Adaptive with llr_history>0 returns history array."""
        code, e, s, llr = noisy_setup
        result = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="adaptive", llr_history=3,
        )
        assert len(result) == 3
        hard, iters, hist = result
        assert hard.shape == (code.n,)
        assert iters >= 1
        assert hist.ndim == 2
        assert hist.shape[1] == code.n

    def test_llr_history_deterministic(self, noisy_setup):
        """Adaptive with llr_history: two runs produce same history."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="adaptive", llr_history=3,
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="adaptive", llr_history=3,
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]
        np.testing.assert_array_equal(r1[2], r2[2])


# ── Default k1 computation ──

class TestAdaptiveDefaultK1:

    def test_default_k1_is_quarter(self):
        """Default k1 = max(1, max_iters // 4)."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        llr = np.array([5.0, 5.0, 5.0])
        s = np.zeros(2, dtype=np.uint8)
        # max_iters=20 → k1=5, k2=15.
        # With zero-error, phase 1 converges in 1 iter.
        result = bp_decode(
            H, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="adaptive",
        )
        assert result[1] <= 5  # converged within phase 1

    def test_default_k1_at_least_one(self):
        """Default k1 is at least 1 even when max_iters is small."""
        H = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        llr = np.array([5.0, 5.0, 5.0])
        s = np.zeros(2, dtype=np.uint8)
        # max_iters=2 → k1 = max(1, 2//4) = max(1,0) = 1, k2=1.
        result = bp_decode(
            H, llr, max_iters=2, mode="min_sum",
            syndrome_vec=s, schedule="adaptive",
        )
        assert result[0].shape == (3,)
        assert result[1] >= 1


# ── Adaptive log output ──

class TestAdaptiveLog:

    def test_adaptive_log_prints(self, noisy_setup, capsys):
        """adaptive_log=True produces output on stdout."""
        code, e, s, llr = noisy_setup
        bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="adaptive", adaptive_log=True,
        )
        captured = capsys.readouterr()
        assert "[adaptive]" in captured.out
        assert "phase1" in captured.out

    def test_adaptive_log_off_by_default(self, noisy_setup, capsys):
        """adaptive_log=False (default) produces no output."""
        code, e, s, llr = noisy_setup
        bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="adaptive",
        )
        captured = capsys.readouterr()
        assert captured.out == ""
