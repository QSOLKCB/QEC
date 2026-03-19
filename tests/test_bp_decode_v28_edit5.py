"""
Tests for v2.8.0 Edit 5: state-aware residual weighting.
"""

import numpy as np
import pytest

from qec_qldpc_codes import bp_decode, syndrome, channel_llr, create_code


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


def _make_state_arrays(m, n_states=3):
    """Build valid state-aware arrays for m checks with n_states states."""
    state_labels = np.arange(m, dtype=np.int64) % n_states
    phi = np.linspace(0.0, np.pi / 4, n_states)
    s_amp = np.linspace(0.5, 1.5, n_states)
    return phi, s_amp, state_labels


class TestStateAwareDeterminism:

    def test_residual_deterministic(self, noisy_setup):
        """state_aware_residual with residual schedule: two runs identical."""
        code, e, s, llr = noisy_setup
        m = code.H_X.shape[0]
        phi, s_amp, labels = _make_state_arrays(m)
        kwargs = dict(
            state_aware_residual=True,
            phi_by_state=phi, s_by_state=s_amp,
            state_label_by_check=labels,
        )
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="residual", **kwargs,
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="residual", **kwargs,
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]

    def test_hybrid_deterministic(self, noisy_setup):
        """state_aware_residual with hybrid_residual: two runs identical."""
        code, e, s, llr = noisy_setup
        m = code.H_X.shape[0]
        phi, s_amp, labels = _make_state_arrays(m)
        kwargs = dict(
            state_aware_residual=True,
            phi_by_state=phi, s_by_state=s_amp,
            state_label_by_check=labels,
        )
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="hybrid_residual", **kwargs,
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="hybrid_residual", **kwargs,
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]


class TestDisabledMatchesBaseline:

    def test_disabled_residual(self, noisy_setup):
        """state_aware_residual=False must be identical to baseline."""
        code, e, s, llr = noisy_setup
        r_base = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="residual",
        )
        r_off = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="residual",
            state_aware_residual=False,
        )
        np.testing.assert_array_equal(r_base[0], r_off[0])
        assert r_base[1] == r_off[1]

    def test_disabled_hybrid(self, noisy_setup):
        """state_aware_residual=False with hybrid_residual matches baseline."""
        code, e, s, llr = noisy_setup
        r_base = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="hybrid_residual",
        )
        r_off = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="hybrid_residual",
            state_aware_residual=False,
        )
        np.testing.assert_array_equal(r_base[0], r_off[0])
        assert r_base[1] == r_off[1]

    def test_disabled_layered(self, noisy_setup):
        """state_aware_residual=False with layered matches baseline."""
        code, e, s, llr = noisy_setup
        r_base = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="layered",
        )
        r_off = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="layered",
            state_aware_residual=False,
        )
        np.testing.assert_array_equal(r_base[0], r_off[0])
        assert r_base[1] == r_off[1]


class TestHybridResidualCompat:

    def test_hybrid_with_threshold(self, noisy_setup):
        """state_aware_residual + hybrid_residual + threshold: no crash, deterministic."""
        code, e, s, llr = noisy_setup
        m = code.H_X.shape[0]
        phi, s_amp, labels = _make_state_arrays(m)
        kwargs = dict(
            state_aware_residual=True,
            phi_by_state=phi, s_by_state=s_amp,
            state_label_by_check=labels,
            hybrid_residual_threshold=0.1,
        )
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="hybrid_residual", **kwargs,
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="hybrid_residual", **kwargs,
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]


class TestEnsembleCompat:

    def test_ensemble_with_state_aware(self, noisy_setup):
        """state_aware_residual + ensemble_k>1: no crash, deterministic."""
        code, e, s, llr = noisy_setup
        m = code.H_X.shape[0]
        phi, s_amp, labels = _make_state_arrays(m)
        kwargs = dict(
            state_aware_residual=True,
            phi_by_state=phi, s_by_state=s_amp,
            state_label_by_check=labels,
            ensemble_k=2,
        )
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="residual", **kwargs,
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="min_sum",
            syndrome_vec=s, schedule="residual", **kwargs,
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]


class TestValidation:

    def test_missing_phi(self, noisy_setup):
        """Missing phi_by_state raises ValueError."""
        code, e, s, llr = noisy_setup
        m = code.H_X.shape[0]
        _, s_amp, labels = _make_state_arrays(m)
        with pytest.raises(ValueError, match="phi_by_state"):
            bp_decode(
                code.H_X, llr, max_iters=5, mode="min_sum",
                syndrome_vec=s, state_aware_residual=True,
                s_by_state=s_amp, state_label_by_check=labels,
            )

    def test_wrong_label_length(self, noisy_setup):
        """state_label_by_check with wrong length raises ValueError."""
        code, e, s, llr = noisy_setup
        m = code.H_X.shape[0]
        phi, s_amp, _ = _make_state_arrays(m)
        bad_labels = np.zeros(m + 5, dtype=np.int64)
        with pytest.raises(ValueError, match="must equal m="):
            bp_decode(
                code.H_X, llr, max_iters=5, mode="min_sum",
                syndrome_vec=s, state_aware_residual=True,
                phi_by_state=phi, s_by_state=s_amp,
                state_label_by_check=bad_labels,
            )

    def test_label_out_of_range(self, noisy_setup):
        """state_label_by_check with out-of-range index raises ValueError."""
        code, e, s, llr = noisy_setup
        m = code.H_X.shape[0]
        phi, s_amp, _ = _make_state_arrays(m, n_states=3)
        bad_labels = np.full(m, 10, dtype=np.int64)  # max=10, but only 3 states
        with pytest.raises(ValueError, match="exceeds"):
            bp_decode(
                code.H_X, llr, max_iters=5, mode="min_sum",
                syndrome_vec=s, state_aware_residual=True,
                phi_by_state=phi, s_by_state=s_amp,
                state_label_by_check=bad_labels,
            )

    def test_negative_label(self, noisy_setup):
        """Negative label in state_label_by_check raises ValueError."""
        code, e, s, llr = noisy_setup
        m = code.H_X.shape[0]
        phi, s_amp, _ = _make_state_arrays(m, n_states=3)
        bad_labels = np.zeros(m, dtype=np.int64)
        bad_labels[0] = -1
        with pytest.raises(ValueError, match="must be >= 0"):
            bp_decode(
                code.H_X, llr, max_iters=5, mode="min_sum",
                syndrome_vec=s, state_aware_residual=True,
                phi_by_state=phi, s_by_state=s_amp,
                state_label_by_check=bad_labels,
            )
