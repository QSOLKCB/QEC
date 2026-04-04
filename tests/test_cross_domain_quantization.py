"""Tests for cross-domain quantization framework — v136.10.0.

Covers:
1.  uniform quantizer correctness
2.  sample-rate downsampling behavior
3.  24-bit vs 8-bit level count
4.  bitcrusher-like degradation (4-bit)
5.  INT8 / INT4 weight quantization
6.  risk band thresholds
7.  phase-space bin correctness
8.  100-run determinism
9.  export equality
10. ledger hash stability
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from qec.analysis.cross_domain_quantization import (
    CROSS_DOMAIN_QUANTIZATION_VERSION,
    RISK_BANDS,
    QuantizationDecision,
    QuantizationLedger,
    bit_depth_quantize,
    build_ledger,
    export_quantization_bundle,
    phase_space_quantize,
    risk_band_quantize,
    sample_rate_quantize,
    uniform_quantize,
    weight_quantize,
)


# -----------------------------------------------------------------------
# 1. Uniform quantizer correctness
# -----------------------------------------------------------------------

class TestUniformQuantize:
    """Core primitive Q(x) = Δ · floor(x/Δ + 1/2)."""

    def test_exact_lattice_points(self):
        """Values already on the lattice must be unchanged."""
        delta = 0.5
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        q = uniform_quantize(x, delta)
        np.testing.assert_array_equal(q, x)

    def test_midpoint_rounding(self):
        """Midpoints round toward positive infinity (floor(x+0.5))."""
        delta = 1.0
        x = np.array([0.5, 1.5, -0.5, -1.5])
        q = uniform_quantize(x, delta)
        expected = np.array([1.0, 2.0, 0.0, -1.0])
        np.testing.assert_array_equal(q, expected)

    def test_small_delta(self):
        """Fine quantization should approximate input closely."""
        delta = 0.01
        x = np.array([0.123456, -0.987654])
        q = uniform_quantize(x, delta)
        assert np.max(np.abs(q - x)) <= delta

    def test_negative_delta_raises(self):
        with pytest.raises(ValueError, match="positive"):
            uniform_quantize(np.array([1.0]), -0.1)

    def test_zero_delta_raises(self):
        with pytest.raises(ValueError, match="positive"):
            uniform_quantize(np.array([1.0]), 0.0)


# -----------------------------------------------------------------------
# 2. Sample-rate downsampling behavior
# -----------------------------------------------------------------------

class TestSampleRateQuantize:

    def test_downsample_halves_length(self):
        """96kHz -> 48kHz should approximately halve sample count."""
        rng = np.random.RandomState(42)
        signal = rng.randn(9600)
        resampled, decision = sample_rate_quantize(
            signal, sample_rate_hz=48000, original_rate_hz=96000,
        )
        assert len(resampled) == 4800
        assert decision.domain == "audio_sample_rate"

    def test_upsample_identity(self):
        """Resampling at the same rate returns same length."""
        signal = np.linspace(-1, 1, 1000)
        resampled, _ = sample_rate_quantize(
            signal, sample_rate_hz=96000, original_rate_hz=96000,
        )
        assert len(resampled) == len(signal)

    def test_8khz_telephone(self):
        """96kHz -> 8kHz should drastically reduce samples."""
        signal = np.ones(9600)
        resampled, decision = sample_rate_quantize(
            signal, sample_rate_hz=8000, original_rate_hz=96000,
        )
        assert len(resampled) == 800

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError):
            sample_rate_quantize(np.ones(100), sample_rate_hz=0)


# -----------------------------------------------------------------------
# 3. 24-bit vs 8-bit level count
# -----------------------------------------------------------------------

class TestBitDepthLevels:

    def test_24bit_levels(self):
        """24-bit quantization -> 2^24 = 16,777,216 levels."""
        samples = np.linspace(-1, 1, 100)
        _, _, decision = bit_depth_quantize(samples, bit_depth=24)
        assert decision.output_levels == 2 ** 24

    def test_8bit_levels(self):
        """8-bit quantization -> 256 levels."""
        samples = np.linspace(-1, 1, 100)
        _, _, decision = bit_depth_quantize(samples, bit_depth=8)
        assert decision.output_levels == 256

    def test_16bit_levels(self):
        """16-bit quantization -> 65536 levels."""
        samples = np.linspace(-1, 1, 100)
        _, _, decision = bit_depth_quantize(samples, bit_depth=16)
        assert decision.output_levels == 65536

    def test_higher_depth_lower_noise(self):
        """24-bit should have less noise than 8-bit."""
        rng = np.random.RandomState(99)
        samples = rng.uniform(-1, 1, 500)
        _, noise_8, _ = bit_depth_quantize(samples, bit_depth=8)
        _, noise_24, _ = bit_depth_quantize(samples, bit_depth=24)
        assert noise_24 < noise_8


# -----------------------------------------------------------------------
# 4. Bitcrusher-like degradation (4-bit)
# -----------------------------------------------------------------------

class TestBitcrusher:

    def test_4bit_coarse(self):
        """4-bit quantization -> only 16 levels, high noise."""
        rng = np.random.RandomState(7)
        samples = rng.uniform(-1, 1, 1000)
        quantized, noise, decision = bit_depth_quantize(samples, bit_depth=4)
        assert decision.output_levels == 16
        # 4-bit should produce visible noise
        assert noise > 1e-4
        # Quantized values should be on the 16-level grid
        delta = 2.0 / 16
        residuals = np.abs(np.round(quantized / delta) * delta - quantized)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-12)

    def test_1bit_extreme(self):
        """1-bit quantization -> 2 levels."""
        samples = np.array([-0.9, -0.1, 0.1, 0.9])
        quantized, _, decision = bit_depth_quantize(samples, bit_depth=1)
        assert decision.output_levels == 2

    def test_invalid_depth_raises(self):
        with pytest.raises(ValueError):
            bit_depth_quantize(np.ones(10), bit_depth=0)


# -----------------------------------------------------------------------
# 5. INT8 / INT4 weight quantization
# -----------------------------------------------------------------------

class TestWeightQuantize:

    def test_int8_256_levels(self):
        """INT8 quantization -> 256 levels."""
        rng = np.random.RandomState(11)
        weights = rng.randn(100)
        quantized, max_err, mse, decision = weight_quantize(weights, bits=8)
        assert decision.output_levels == 256
        assert max_err >= 0.0
        assert mse >= 0.0
        assert decision.domain == "ai_weight"

    def test_int4_16_levels(self):
        """INT4 quantization -> 16 levels."""
        rng = np.random.RandomState(22)
        weights = rng.randn(100)
        quantized, max_err, mse, decision = weight_quantize(weights, bits=4)
        assert decision.output_levels == 16
        # INT4 has larger error than INT8
        _, _, mse8, _ = weight_quantize(weights, bits=8)
        assert mse > mse8

    def test_fp16_simulation(self):
        """16-bit quantization simulates FP16-like bucket."""
        rng = np.random.RandomState(33)
        weights = rng.randn(200) * 5.0
        quantized, _, mse, decision = weight_quantize(weights, bits=16)
        assert decision.output_levels == 65536
        assert mse < 0.01  # very fine grid

    def test_zero_weights(self):
        """All-zero weights should quantize to zero with no error."""
        weights = np.zeros(50)
        quantized, max_err, mse, _ = weight_quantize(weights, bits=8)
        np.testing.assert_array_equal(quantized, 0.0)
        assert max_err == 0.0
        assert mse == 0.0

    def test_invalid_bits_raises(self):
        with pytest.raises(ValueError):
            weight_quantize(np.ones(10), bits=0)


# -----------------------------------------------------------------------
# 6. Risk band thresholds
# -----------------------------------------------------------------------

class TestRiskBandQuantize:

    def test_low(self):
        band, _ = risk_band_quantize(0.0)
        assert band == "LOW"

    def test_watch(self):
        band, _ = risk_band_quantize(0.2)
        assert band == "WATCH"

    def test_warning(self):
        band, _ = risk_band_quantize(0.4)
        assert band == "WARNING"

    def test_critical(self):
        band, _ = risk_band_quantize(0.6)
        assert band == "CRITICAL"

    def test_collapse_imminent(self):
        band, _ = risk_band_quantize(0.8)
        assert band == "COLLAPSE_IMMINENT"

    def test_boundary_just_below(self):
        band, _ = risk_band_quantize(0.199)
        assert band == "LOW"

    def test_max_score(self):
        band, _ = risk_band_quantize(1.0)
        assert band == "COLLAPSE_IMMINENT"

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            risk_band_quantize(-0.1)
        with pytest.raises(ValueError):
            risk_band_quantize(1.01)

    def test_decision_has_5_levels(self):
        _, decision = risk_band_quantize(0.5)
        assert decision.output_levels == 5
        assert decision.domain == "control_risk_band"


# -----------------------------------------------------------------------
# 7. Phase-space bin correctness
# -----------------------------------------------------------------------

class TestPhaseSpaceQuantize:

    def test_origin(self):
        qQ, pQ, (iq, ip), _ = phase_space_quantize(0.0, 0.0, 1.0)
        assert qQ == 0.0
        assert pQ == 0.0
        assert iq == 0
        assert ip == 0

    def test_positive_bin(self):
        qQ, pQ, (iq, ip), _ = phase_space_quantize(1.3, 2.7, 1.0)
        assert qQ == 1.0
        assert pQ == 3.0
        assert iq == 1
        assert ip == 3

    def test_negative_bin(self):
        qQ, pQ, (iq, ip), _ = phase_space_quantize(-0.8, -1.2, 1.0)
        assert qQ == -1.0
        assert pQ == -1.0
        assert iq == -1
        assert ip == -1

    def test_fine_grid(self):
        qQ, pQ, (iq, ip), decision = phase_space_quantize(0.123, 0.456, 0.1)
        assert abs(qQ - 0.1) < 1e-12
        assert abs(pQ - 0.5) < 1e-12
        assert decision.domain == "phase_space"

    def test_error_estimate(self):
        _, _, _, decision = phase_space_quantize(0.3, 0.7, 1.0)
        assert decision.error_estimate >= 0.0

    def test_invalid_bin_width_raises(self):
        with pytest.raises(ValueError):
            phase_space_quantize(0.0, 0.0, 0.0)


# -----------------------------------------------------------------------
# 8. 100-run determinism
# -----------------------------------------------------------------------

class TestDeterminism:

    def test_uniform_quantize_determinism(self):
        x = np.linspace(-5, 5, 200)
        ref = uniform_quantize(x, 0.3)
        for _ in range(100):
            result = uniform_quantize(x, 0.3)
            np.testing.assert_array_equal(result, ref)

    def test_bit_depth_determinism(self):
        rng = np.random.RandomState(77)
        samples = rng.uniform(-1, 1, 300)
        ref_q, ref_n, ref_d = bit_depth_quantize(samples, bit_depth=8)
        for _ in range(100):
            q, n, d = bit_depth_quantize(samples, bit_depth=8)
            np.testing.assert_array_equal(q, ref_q)
            assert n == ref_n
            assert d.stable_hash == ref_d.stable_hash

    def test_weight_quantize_determinism(self):
        rng = np.random.RandomState(88)
        weights = rng.randn(200)
        ref_q, ref_mae, ref_mse, ref_d = weight_quantize(weights, bits=4)
        for _ in range(100):
            q, mae, mse, d = weight_quantize(weights, bits=4)
            np.testing.assert_array_equal(q, ref_q)
            assert mae == ref_mae
            assert mse == ref_mse
            assert d.stable_hash == ref_d.stable_hash

    def test_risk_band_determinism(self):
        ref_band, ref_d = risk_band_quantize(0.55)
        for _ in range(100):
            band, d = risk_band_quantize(0.55)
            assert band == ref_band
            assert d.stable_hash == ref_d.stable_hash

    def test_phase_space_determinism(self):
        ref_qQ, ref_pQ, ref_idx, ref_d = phase_space_quantize(1.23, -4.56, 0.5)
        for _ in range(100):
            qQ, pQ, idx, d = phase_space_quantize(1.23, -4.56, 0.5)
            assert qQ == ref_qQ
            assert pQ == ref_pQ
            assert idx == ref_idx
            assert d.stable_hash == ref_d.stable_hash


# -----------------------------------------------------------------------
# 9. Export equality
# -----------------------------------------------------------------------

class TestExportEquality:

    def _make_ledger(self) -> QuantizationLedger:
        decisions = []
        _, d1 = risk_band_quantize(0.3)
        decisions.append(d1)
        _, _, d2 = bit_depth_quantize(np.linspace(-1, 1, 50), bit_depth=16)
        decisions.append(d2)
        rng = np.random.RandomState(42)
        _, _, _, d3 = weight_quantize(rng.randn(50), bits=8)
        decisions.append(d3)
        _, _, _, d4 = phase_space_quantize(1.0, 2.0, 0.5)
        decisions.append(d4)
        return build_ledger(tuple(decisions))

    def test_export_is_valid_json(self):
        ledger = self._make_ledger()
        exported = export_quantization_bundle(ledger)
        parsed = json.loads(exported)
        assert "decisions" in parsed
        assert "stable_hash" in parsed
        assert parsed["version"] == CROSS_DOMAIN_QUANTIZATION_VERSION

    def test_export_byte_identical(self):
        """Two exports of the same ledger must be byte-identical."""
        ledger = self._make_ledger()
        a = export_quantization_bundle(ledger)
        b = export_quantization_bundle(ledger)
        assert a == b

    def test_different_ledgers_different_hash(self):
        _, d1 = risk_band_quantize(0.1)
        _, d2 = risk_band_quantize(0.9)
        l1 = build_ledger((d1,))
        l2 = build_ledger((d2,))
        assert l1.stable_hash != l2.stable_hash


# -----------------------------------------------------------------------
# 10. Ledger hash stability
# -----------------------------------------------------------------------

class TestLedgerHashStability:

    def test_ledger_hash_deterministic(self):
        """Same decisions -> same ledger hash, 100 runs."""
        _, d1 = risk_band_quantize(0.5)
        _, _, d2 = bit_depth_quantize(np.array([0.1, -0.2, 0.3]), bit_depth=8)
        ref = build_ledger((d1, d2))
        for _ in range(100):
            ledger = build_ledger((d1, d2))
            assert ledger.stable_hash == ref.stable_hash

    def test_order_matters(self):
        """Different ordering -> different hash."""
        _, d1 = risk_band_quantize(0.1)
        _, d2 = risk_band_quantize(0.9)
        l_ab = build_ledger((d1, d2))
        l_ba = build_ledger((d2, d1))
        assert l_ab.stable_hash != l_ba.stable_hash

    def test_empty_ledger(self):
        ledger = build_ledger(())
        assert isinstance(ledger.stable_hash, str)
        assert len(ledger.stable_hash) == 64  # SHA-256

    def test_frozen_decision(self):
        """QuantizationDecision must be immutable."""
        _, decision = risk_band_quantize(0.5)
        with pytest.raises(AttributeError):
            decision.domain = "hacked"  # type: ignore[misc]

    def test_frozen_ledger(self):
        """QuantizationLedger must be immutable."""
        ledger = build_ledger(())
        with pytest.raises(AttributeError):
            ledger.stable_hash = "hacked"  # type: ignore[misc]
