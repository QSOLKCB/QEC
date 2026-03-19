"""
Tests for v2.8.0 Edit 4: deterministic ensemble decoding.
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


class TestEnsembleK1Identical:

    def test_ensemble_k1_matches_baseline(self, noisy_setup):
        """ensemble_k=1 must produce identical output to default (no ensemble)."""
        code, e, s, llr = noisy_setup
        r_base = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            syndrome_vec=s, schedule="layered",
        )
        r_ens1 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            syndrome_vec=s, schedule="layered",
            ensemble_k=1,
        )
        np.testing.assert_array_equal(r_base[0], r_ens1[0])
        assert r_base[1] == r_ens1[1]

    def test_ensemble_k1_flooding(self, noisy_setup):
        """ensemble_k=1 with flooding must match baseline."""
        code, e, s, llr = noisy_setup
        r_base = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            syndrome_vec=s, schedule="flooding",
        )
        r_ens1 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            syndrome_vec=s, schedule="flooding",
            ensemble_k=1,
        )
        np.testing.assert_array_equal(r_base[0], r_ens1[0])
        assert r_base[1] == r_ens1[1]


class TestEnsembleDeterminism:

    def test_ensemble_k2_deterministic(self, noisy_setup):
        """ensemble_k=2: two calls must be identical."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            syndrome_vec=s, schedule="layered",
            ensemble_k=2,
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=30, mode="min_sum",
            syndrome_vec=s, schedule="layered",
            ensemble_k=2,
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]

    def test_ensemble_k3_deterministic(self, noisy_setup):
        """ensemble_k=3: deterministic across calls."""
        code, e, s, llr = noisy_setup
        r1 = bp_decode(
            code.H_X, llr, max_iters=20, mode="norm_min_sum",
            syndrome_vec=s, schedule="residual",
            ensemble_k=3,
        )
        r2 = bp_decode(
            code.H_X, llr, max_iters=20, mode="norm_min_sum",
            syndrome_vec=s, schedule="residual",
            ensemble_k=3,
        )
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]


class TestEnsembleBaselinePreserved:

    def test_converged_baseline_returned(self, noisy_setup):
        """If member 0 converges, ensemble should not degrade result."""
        code, e, s, llr = noisy_setup
        # First verify baseline converges.
        r_base = bp_decode(
            code.H_X, llr, max_iters=50, mode="min_sum",
            syndrome_vec=s, schedule="layered",
        )
        syn_base = (
            (code.H_X.astype(np.int32) @ r_base[0].astype(np.int32)) % 2
        ).astype(np.uint8)
        base_converged = np.array_equal(syn_base, s)

        r_ens = bp_decode(
            code.H_X, llr, max_iters=50, mode="min_sum",
            syndrome_vec=s, schedule="layered",
            ensemble_k=4,
        )
        syn_ens = (
            (code.H_X.astype(np.int32) @ r_ens[0].astype(np.int32)) % 2
        ).astype(np.uint8)
        ens_converged = np.array_equal(syn_ens, s)

        # If baseline converged, ensemble must also converge.
        if base_converged:
            assert ens_converged, "Ensemble degraded a converged baseline"


class TestEnsembleReturnFormat:

    def test_return_tuple_no_history(self, noisy_setup):
        """ensemble_k>1 without llr_history returns (hard, iters)."""
        code, e, s, llr = noisy_setup
        result = bp_decode(
            code.H_X, llr, max_iters=10, mode="min_sum",
            syndrome_vec=s, ensemble_k=2,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].dtype == np.uint8
        assert isinstance(result[1], (int, np.integer))

    def test_return_tuple_with_history(self, noisy_setup):
        """ensemble_k>1 with llr_history returns (hard, iters, history)."""
        code, e, s, llr = noisy_setup
        result = bp_decode(
            code.H_X, llr, max_iters=10, mode="min_sum",
            syndrome_vec=s, ensemble_k=2, llr_history=3,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert result[0].dtype == np.uint8
        assert isinstance(result[1], (int, np.integer))
        assert isinstance(result[2], np.ndarray)
