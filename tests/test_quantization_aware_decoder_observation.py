"""
Tests for quantization-aware decoder observation — v137.0.1

Covers:
    risk band mapping
    syndrome drift quantization
    phase-space bin correctness
    symbolic signature determinism
    compression metric correctness
    frozen immutability
    export equality
    stable ledger hashing
    100-run determinism
    no-regression against v136.10.0
"""

from __future__ import annotations

import json

import pytest

from qec.analysis.quantization_aware_decoder_observation import (
    DECODER_OBSERVATION_VERSION,
    DEFAULT_PHASE_BIN_WIDTH,
    DRIFT_BANDS,
    DRIFT_THRESHOLDS,
    STABILITY_BANDS,
    STABILITY_THRESHOLDS,
    DecoderObservationSignature,
    ObservationLedger,
    build_observation_ledger,
    export_decoder_observation_bundle,
    observe_decoder_quantization,
)
from qec.analysis.cross_domain_quantization import (
    RISK_BANDS,
    risk_band_quantize,
    phase_space_quantize,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_INPUTS = {
    "syndrome_drift": 0.35,
    "decoder_stability_score": 0.55,
    "phase_centroid_q": 1.1,
    "phase_centroid_p": -0.3,
    "risk_score": 0.45,
    "decoder_family": "bp_reference",
}


def _make_observation(**overrides):
    """Helper to create an observation with defaults."""
    kw = dict(SAMPLE_INPUTS)
    kw.update(overrides)
    return observe_decoder_quantization(**kw)


# ---------------------------------------------------------------------------
# Risk band mapping
# ---------------------------------------------------------------------------

class TestRiskBandMapping:
    """Verify risk band mapping matches v136.10.0 semantics."""

    def test_low(self):
        sig = _make_observation(risk_score=0.1)
        assert sig.symbolic_risk_lattice == "LOW"

    def test_watch(self):
        sig = _make_observation(risk_score=0.25)
        assert sig.symbolic_risk_lattice == "WATCH"

    def test_warning(self):
        sig = _make_observation(risk_score=0.45)
        assert sig.symbolic_risk_lattice == "WARNING"

    def test_critical(self):
        sig = _make_observation(risk_score=0.7)
        assert sig.symbolic_risk_lattice == "CRITICAL"

    def test_collapse_imminent(self):
        sig = _make_observation(risk_score=0.9)
        assert sig.symbolic_risk_lattice == "COLLAPSE_IMMINENT"

    def test_boundary_zero(self):
        sig = _make_observation(risk_score=0.0)
        assert sig.symbolic_risk_lattice == "LOW"

    def test_boundary_one(self):
        sig = _make_observation(risk_score=1.0)
        assert sig.symbolic_risk_lattice == "COLLAPSE_IMMINENT"

    def test_consistency_with_v136(self):
        """Risk bands must agree with v136.10.0 risk_band_quantize."""
        for score in [0.0, 0.15, 0.2, 0.39, 0.4, 0.59, 0.6, 0.79, 0.8, 1.0]:
            sig = _make_observation(risk_score=score)
            expected_band, _ = risk_band_quantize(score)
            assert sig.symbolic_risk_lattice == expected_band


# ---------------------------------------------------------------------------
# Syndrome drift quantization
# ---------------------------------------------------------------------------

class TestSyndromeDriftBand:
    """Verify syndrome drift band classification."""

    def test_low(self):
        sig = _make_observation(syndrome_drift=0.1)
        assert sig.syndrome_drift_band == "LOW"

    def test_mid(self):
        sig = _make_observation(syndrome_drift=0.35)
        assert sig.syndrome_drift_band == "MID"

    def test_high(self):
        sig = _make_observation(syndrome_drift=0.6)
        assert sig.syndrome_drift_band == "HIGH"

    def test_extreme(self):
        sig = _make_observation(syndrome_drift=0.8)
        assert sig.syndrome_drift_band == "EXTREME"

    def test_boundary_values(self):
        assert _make_observation(syndrome_drift=0.0).syndrome_drift_band == "LOW"
        assert _make_observation(syndrome_drift=0.25).syndrome_drift_band == "MID"
        assert _make_observation(syndrome_drift=0.5).syndrome_drift_band == "HIGH"
        assert _make_observation(syndrome_drift=0.75).syndrome_drift_band == "EXTREME"


# ---------------------------------------------------------------------------
# Phase-space bin correctness
# ---------------------------------------------------------------------------

class TestPhaseSpaceBin:
    """Verify phase-space quantization matches v136.10.0."""

    def test_default_bin(self):
        sig = _make_observation(phase_centroid_q=1.1, phase_centroid_p=-0.3)
        q_exp, p_exp, bin_exp, _ = phase_space_quantize(
            1.1, -0.3, DEFAULT_PHASE_BIN_WIDTH,
        )
        assert sig.phase_bin_index == bin_exp
        assert sig.phase_quantized_coords == (q_exp, p_exp)

    def test_origin(self):
        sig = _make_observation(phase_centroid_q=0.0, phase_centroid_p=0.0)
        assert sig.phase_bin_index == (0, 0)
        assert sig.phase_quantized_coords == (0.0, 0.0)

    def test_custom_bin_width(self):
        sig = observe_decoder_quantization(
            syndrome_drift=0.1,
            decoder_stability_score=0.5,
            phase_centroid_q=2.3,
            phase_centroid_p=-1.7,
            risk_score=0.3,
            phase_bin_width=1.0,
        )
        q_exp, p_exp, bin_exp, _ = phase_space_quantize(2.3, -1.7, 1.0)
        assert sig.phase_bin_index == bin_exp
        assert sig.phase_quantized_coords == (q_exp, p_exp)

    def test_negative_coords(self):
        sig = _make_observation(phase_centroid_q=-3.2, phase_centroid_p=-1.8)
        q_exp, p_exp, bin_exp, _ = phase_space_quantize(
            -3.2, -1.8, DEFAULT_PHASE_BIN_WIDTH,
        )
        assert sig.phase_bin_index == bin_exp


# ---------------------------------------------------------------------------
# Stability band
# ---------------------------------------------------------------------------

class TestStabilityBand:
    """Verify stability band classification."""

    def test_low(self):
        sig = _make_observation(decoder_stability_score=0.2)
        assert sig.stability_band == "LOW"

    def test_mid(self):
        sig = _make_observation(decoder_stability_score=0.55)
        assert sig.stability_band == "MID"

    def test_high(self):
        sig = _make_observation(decoder_stability_score=0.85)
        assert sig.stability_band == "HIGH"


# ---------------------------------------------------------------------------
# Symbolic signature determinism
# ---------------------------------------------------------------------------

class TestSymbolicSignature:
    """Verify symbolic signature format and determinism."""

    def test_format(self):
        sig = _make_observation()
        parts = sig.decoder_quantization_signature.split(" | ")
        assert len(parts) == 4
        assert parts[0].startswith("RISK:")
        assert parts[1].startswith("DRIFT:")
        assert parts[2].startswith("PHASE:")
        assert parts[3].startswith("STAB:")

    def test_known_value(self):
        sig = _make_observation(
            risk_score=0.45,
            syndrome_drift=0.35,
            phase_centroid_q=1.1,
            phase_centroid_p=-0.3,
            decoder_stability_score=0.55,
        )
        _, _, phase_bin, _ = phase_space_quantize(1.1, -0.3, DEFAULT_PHASE_BIN_WIDTH)
        expected = (
            f"RISK:WARNING | DRIFT:MID | "
            f"PHASE:({phase_bin[0]},{phase_bin[1]}) | STAB:MID"
        )
        assert sig.decoder_quantization_signature == expected

    def test_deterministic_across_calls(self):
        sig1 = _make_observation()
        sig2 = _make_observation()
        assert sig1.decoder_quantization_signature == sig2.decoder_quantization_signature


# ---------------------------------------------------------------------------
# Frozen immutability
# ---------------------------------------------------------------------------

class TestFrozenImmutability:
    """Verify frozen dataclass constraints."""

    def test_signature_immutable(self):
        sig = _make_observation()
        with pytest.raises(AttributeError):
            sig.symbolic_risk_lattice = "HACKED"

    def test_ledger_immutable(self):
        sig = _make_observation()
        ledger = build_observation_ledger((sig,))
        with pytest.raises(AttributeError):
            ledger.stable_hash = "HACKED"

    def test_signature_is_frozen_dataclass(self):
        sig = _make_observation()
        assert isinstance(sig, DecoderObservationSignature)
        assert sig.__dataclass_params__.frozen  # type: ignore[attr-defined]

    def test_ledger_is_frozen_dataclass(self):
        sig = _make_observation()
        ledger = build_observation_ledger((sig,))
        assert isinstance(ledger, ObservationLedger)
        assert ledger.__dataclass_params__.frozen  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Compression metrics
# ---------------------------------------------------------------------------

class TestCompressionMetrics:
    """Verify observability compression metrics."""

    def test_single_signature(self):
        sig = _make_observation()
        ledger = build_observation_ledger((sig,))
        assert ledger.unique_symbol_count == 1
        assert ledger.phase_bin_entropy_proxy == 0.0
        assert ledger.symbolic_compression_ratio == 1.0

    def test_identical_signatures(self):
        sig = _make_observation()
        ledger = build_observation_ledger((sig, sig, sig))
        assert ledger.unique_symbol_count == 1
        assert ledger.symbolic_compression_ratio == pytest.approx(1.0 / 3.0)

    def test_distinct_signatures(self):
        s1 = _make_observation(risk_score=0.1)
        s2 = _make_observation(risk_score=0.5)
        s3 = _make_observation(risk_score=0.9)
        ledger = build_observation_ledger((s1, s2, s3))
        assert ledger.unique_symbol_count == 3
        assert ledger.symbolic_compression_ratio == 1.0

    def test_phase_entropy_multiple_bins(self):
        s1 = _make_observation(phase_centroid_q=0.0, phase_centroid_p=0.0)
        s2 = _make_observation(phase_centroid_q=5.0, phase_centroid_p=5.0)
        ledger = build_observation_ledger((s1, s2))
        assert ledger.phase_bin_entropy_proxy > 0.0

    def test_empty_ledger(self):
        ledger = build_observation_ledger(())
        assert ledger.signature_count == 0
        assert ledger.unique_symbol_count == 0
        assert ledger.phase_bin_entropy_proxy == 0.0
        assert ledger.symbolic_compression_ratio == 0.0


# ---------------------------------------------------------------------------
# Export equality
# ---------------------------------------------------------------------------

class TestExportEquality:
    """Verify byte-identical JSON export on replay."""

    def test_single_export(self):
        sig = _make_observation()
        ledger = build_observation_ledger((sig,))
        j1 = export_decoder_observation_bundle(ledger)
        j2 = export_decoder_observation_bundle(ledger)
        assert j1 == j2

    def test_export_is_valid_json(self):
        sig = _make_observation()
        ledger = build_observation_ledger((sig,))
        j = export_decoder_observation_bundle(ledger)
        parsed = json.loads(j)
        assert "signatures" in parsed
        assert "stable_hash" in parsed
        assert "version" in parsed
        assert parsed["version"] == DECODER_OBSERVATION_VERSION

    def test_export_contains_metrics(self):
        sig = _make_observation()
        ledger = build_observation_ledger((sig,))
        parsed = json.loads(export_decoder_observation_bundle(ledger))
        metrics = parsed["compression_metrics"]
        assert "unique_symbol_count" in metrics
        assert "phase_bin_entropy_proxy" in metrics
        assert "symbolic_compression_ratio" in metrics

    def test_export_with_none_decoder_family(self):
        sig = _make_observation(decoder_family=None)
        ledger = build_observation_ledger((sig,))
        parsed = json.loads(export_decoder_observation_bundle(ledger))
        assert parsed["signatures"][0]["decoder_family"] == ""


# ---------------------------------------------------------------------------
# Stable ledger hashing
# ---------------------------------------------------------------------------

class TestLedgerHashStability:
    """Verify ledger hash stability across runs."""

    def test_same_content_same_hash(self):
        sig = _make_observation()
        l1 = build_observation_ledger((sig,))
        l2 = build_observation_ledger((sig,))
        assert l1.stable_hash == l2.stable_hash

    def test_different_content_different_hash(self):
        s1 = _make_observation(risk_score=0.1)
        s2 = _make_observation(risk_score=0.9)
        l1 = build_observation_ledger((s1,))
        l2 = build_observation_ledger((s2,))
        assert l1.stable_hash != l2.stable_hash

    def test_order_matters(self):
        s1 = _make_observation(risk_score=0.1)
        s2 = _make_observation(risk_score=0.9)
        l1 = build_observation_ledger((s1, s2))
        l2 = build_observation_ledger((s2, s1))
        assert l1.stable_hash != l2.stable_hash

    def test_signature_hash_stability(self):
        sig1 = _make_observation()
        sig2 = _make_observation()
        assert sig1.stable_hash == sig2.stable_hash
        assert len(sig1.stable_hash) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# 100-run determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Guarantee byte-identical results across 100 runs."""

    def test_observation_100_runs(self):
        ref = _make_observation()
        for _ in range(99):
            obs = _make_observation()
            assert obs == ref
            assert obs.stable_hash == ref.stable_hash

    def test_ledger_100_runs(self):
        s1 = _make_observation(risk_score=0.1)
        s2 = _make_observation(risk_score=0.5)
        ref_ledger = build_observation_ledger((s1, s2))
        ref_json = export_decoder_observation_bundle(ref_ledger)
        for _ in range(99):
            s1b = _make_observation(risk_score=0.1)
            s2b = _make_observation(risk_score=0.5)
            ledger = build_observation_ledger((s1b, s2b))
            assert ledger.stable_hash == ref_ledger.stable_hash
            assert export_decoder_observation_bundle(ledger) == ref_json


# ---------------------------------------------------------------------------
# No-regression against v136.10.0
# ---------------------------------------------------------------------------

class TestV136Regression:
    """Ensure observation layer agrees with v136.10.0 quantization outputs."""

    def test_risk_band_agreement(self):
        for score in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            sig = _make_observation(risk_score=score)
            expected, _ = risk_band_quantize(score)
            assert sig.symbolic_risk_lattice == expected

    def test_phase_space_agreement(self):
        coords = [
            (0.0, 0.0), (1.0, -1.0), (-2.5, 3.7), (0.25, 0.25),
        ]
        for q, p in coords:
            sig = _make_observation(phase_centroid_q=q, phase_centroid_p=p)
            q_exp, p_exp, bin_exp, _ = phase_space_quantize(
                q, p, DEFAULT_PHASE_BIN_WIDTH,
            )
            assert sig.phase_bin_index == bin_exp
            assert sig.phase_quantized_coords == (q_exp, p_exp)


# ---------------------------------------------------------------------------
# Decoder family optional
# ---------------------------------------------------------------------------

class TestDecoderFamily:
    """Verify decoder_family is optional and propagated correctly."""

    def test_default_none(self):
        sig = observe_decoder_quantization(
            syndrome_drift=0.1,
            decoder_stability_score=0.5,
            phase_centroid_q=0.0,
            phase_centroid_p=0.0,
            risk_score=0.3,
        )
        assert sig.decoder_family is None

    def test_explicit_family(self):
        sig = _make_observation(decoder_family="bp_experimental")
        assert sig.decoder_family == "bp_experimental"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Verify invalid inputs raise ValueError."""

    def test_risk_score_below_zero(self):
        with pytest.raises(ValueError, match="risk_score"):
            _make_observation(risk_score=-0.1)

    def test_risk_score_above_one(self):
        with pytest.raises(ValueError, match="risk_score"):
            _make_observation(risk_score=1.01)

    def test_syndrome_drift_below_zero(self):
        with pytest.raises(ValueError, match="syndrome_drift"):
            _make_observation(syndrome_drift=-0.01)

    def test_syndrome_drift_above_one(self):
        with pytest.raises(ValueError, match="syndrome_drift"):
            _make_observation(syndrome_drift=1.5)

    def test_stability_score_below_zero(self):
        with pytest.raises(ValueError, match="decoder_stability_score"):
            _make_observation(decoder_stability_score=-0.1)

    def test_stability_score_above_one(self):
        with pytest.raises(ValueError, match="decoder_stability_score"):
            _make_observation(decoder_stability_score=2.0)

    def test_phase_bin_width_zero(self):
        with pytest.raises(ValueError, match="phase_bin_width"):
            observe_decoder_quantization(
                syndrome_drift=0.1,
                decoder_stability_score=0.5,
                phase_centroid_q=0.0,
                phase_centroid_p=0.0,
                risk_score=0.3,
                phase_bin_width=0.0,
            )

    def test_phase_bin_width_negative(self):
        with pytest.raises(ValueError, match="phase_bin_width"):
            observe_decoder_quantization(
                syndrome_drift=0.1,
                decoder_stability_score=0.5,
                phase_centroid_q=0.0,
                phase_centroid_p=0.0,
                risk_score=0.3,
                phase_bin_width=-1.0,
            )
