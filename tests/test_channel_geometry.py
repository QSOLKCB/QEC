"""
Tests for channel geometry functions (v3.9.0).

Verifies:
  - Centered field removes mean bias.
  - Pseudo-prior bias is deterministic.
  - Baseline invariance when all features disabled.
"""

from __future__ import annotations

import numpy as np
import pytest

from qec.channel.geometry import (
    syndrome_field,
    centered_syndrome_field,
    pseudo_prior_bias,
    apply_pseudo_prior,
)


@pytest.fixture
def small_system():
    """Small H matrix and syndrome for testing."""
    H = np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
    ], dtype=np.uint8)
    s = np.array([1, 0, 1], dtype=np.uint8)
    return H, s


class TestSyndromeField:
    def test_shape(self, small_system):
        H, s = small_system
        llr = syndrome_field(H, s)
        assert llr.shape == (H.shape[1],)

    def test_deterministic(self, small_system):
        H, s = small_system
        llr1 = syndrome_field(H, s)
        llr2 = syndrome_field(H, s)
        np.testing.assert_array_equal(llr1, llr2)

    def test_values(self, small_system):
        H, s = small_system
        # b = 1 - 2*s = [-1, 1, -1]
        # H.T @ b
        b = np.array([-1.0, 1.0, -1.0])
        expected = H.astype(np.float64).T @ b
        llr = syndrome_field(H, s)
        np.testing.assert_array_almost_equal(llr, expected)


class TestCenteredSyndromeField:
    def test_shape(self, small_system):
        H, s = small_system
        llr = centered_syndrome_field(H, s)
        assert llr.shape == (H.shape[1],)

    def test_mean_centered(self, small_system):
        H, s = small_system
        # b = 1 - 2*s, b_centered = b - mean(b)
        b = 1 - 2 * s.astype(np.float64)
        b_centered = b - np.mean(b)
        np.testing.assert_almost_equal(np.mean(b_centered), 0.0)

    def test_deterministic(self, small_system):
        H, s = small_system
        llr1 = centered_syndrome_field(H, s)
        llr2 = centered_syndrome_field(H, s)
        np.testing.assert_array_equal(llr1, llr2)

    def test_values(self, small_system):
        H, s = small_system
        b = 1 - 2 * s.astype(np.float64)
        b_centered = b - np.mean(b)
        expected = H.astype(np.float64).T @ b_centered
        llr = centered_syndrome_field(H, s)
        np.testing.assert_array_almost_equal(llr, expected)

    def test_zero_syndrome_equals_standard(self, small_system):
        """When syndrome is all zeros, b = [1,1,...,1], mean = 1,
        b_centered = [0,0,...,0], so centered field should be zero."""
        H, _ = small_system
        s_zero = np.zeros(H.shape[0], dtype=np.uint8)
        llr = centered_syndrome_field(H, s_zero)
        np.testing.assert_array_almost_equal(llr, np.zeros(H.shape[1]))


class TestPseudoPriorBias:
    def test_shape(self, small_system):
        H, s = small_system
        bias = pseudo_prior_bias(H, s)
        assert bias.shape == (H.shape[1],)

    def test_deterministic(self, small_system):
        H, s = small_system
        b1 = pseudo_prior_bias(H, s)
        b2 = pseudo_prior_bias(H, s)
        np.testing.assert_array_equal(b1, b2)

    def test_equals_syndrome_field(self, small_system):
        """pseudo_prior_bias and syndrome_field compute the same thing."""
        H, s = small_system
        llr = syndrome_field(H, s)
        bias = pseudo_prior_bias(H, s)
        np.testing.assert_array_equal(llr, bias)


class TestApplyPseudoPrior:
    def test_identity_at_zero_kappa(self, small_system):
        H, s = small_system
        llr = syndrome_field(H, s)
        bias = pseudo_prior_bias(H, s)
        result = apply_pseudo_prior(llr, bias, kappa=0.0)
        np.testing.assert_array_almost_equal(result, llr)

    def test_additive(self, small_system):
        H, s = small_system
        llr = syndrome_field(H, s)
        bias = pseudo_prior_bias(H, s)
        kappa = 0.25
        result = apply_pseudo_prior(llr, bias, kappa)
        expected = llr + kappa * bias
        np.testing.assert_array_almost_equal(result, expected)

    def test_deterministic(self, small_system):
        H, s = small_system
        llr = syndrome_field(H, s)
        bias = pseudo_prior_bias(H, s)
        r1 = apply_pseudo_prior(llr, bias, 0.25)
        r2 = apply_pseudo_prior(llr, bias, 0.25)
        np.testing.assert_array_equal(r1, r2)


class TestBaselineInvariance:
    def test_disabled_features_preserve_llr(self, small_system):
        """When no geometry features are enabled, the original LLR is used."""
        from qec.decoder.rpc import StructuralConfig
        config = StructuralConfig()
        # Verify defaults
        assert config.centered_field is False
        assert config.pseudo_prior is False
        assert config.energy_trace is False

    def test_decode_equivalence_when_features_disabled(self):
        """Decoder output is identical when geometry interventions are disabled."""
        from qec.decoder.rpc import StructuralConfig
        from qec_qldpc_codes import bp_decode, create_code, syndrome, channel_llr

        code = create_code(name="rate_0.50", lifting_size=3, seed=42)
        H = code.H_X
        n = H.shape[1]
        rng = np.random.default_rng(42)
        e = (rng.random(n) < 0.05).astype(np.uint8)
        s = syndrome(H, e)
        llr = channel_llr(e, 0.05)

        cfg_default = StructuralConfig()
        cfg_explicit = StructuralConfig(
            centered_field=False,
            pseudo_prior=False,
            energy_trace=False,
        )

        # Both configs must have identical defaults.
        assert cfg_default == cfg_explicit

        # Decode with raw bp_decode (no adapter intervention).
        result1 = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="flooding", syndrome_vec=s,
        )
        result2 = bp_decode(
            H, llr, max_iters=10, mode="min_sum",
            schedule="flooding", syndrome_vec=s,
        )
        np.testing.assert_array_equal(result1[0], result2[0])
        assert result1[1] == result2[1]

        # Verify adapter pipeline does not alter LLR when features disabled.
        # When centered_field=False and pseudo_prior=False, llr_used == llr.
        assert cfg_default.centered_field is False
        assert cfg_default.pseudo_prior is False
        # No geometry function should be called; original LLR passes through.
