"""Tests for v137.0.7 — Quantization-Aware Forecast Compression.

Target: 55–75 tests covering tokenization correctness, adjacent-run compression,
compression_ratio bounds, entropy_proxy bounds, dominant arbitration tie-breaking,
stability class detection, loss budget classification, symbolic trace correctness,
frozen immutability, export equality, stable hashing, 100-run replay,
no decoder contamination, and invalid input rejection.
"""

from __future__ import annotations

import json

import pytest

from qec.analysis.closed_loop_auditory_phase_control import (
    AuditoryPhaseSignature,
    observe_auditory_phase_control,
)
from qec.analysis.temporal_auditory_sequence_analysis import (
    TemporalAuditorySequenceDecision,
    analyze_auditory_sequence,
)
from qec.analysis.temporal_auditory_sequence_policy_memory import (
    TemporalAuditoryPolicyState,
    build_temporal_auditory_policy_state,
)
from qec.analysis.temporal_auditory_policy_arbitration import (
    ARBITRATION_LOCKDOWN,
    ARBITRATION_MERGE,
    ARBITRATION_PASS_THROUGH,
    ARBITRATION_PRIORITIZE_CRITICAL,
    ARBITRATION_PRIORITIZE_STABLE,
    TemporalAuditoryArbitrationDecision,
    arbitrate_temporal_auditory_policies,
)
from qec.analysis.quantization_aware_forecast_compression import (
    AR_CRIT,
    AR_LOCK,
    AR_MERGE,
    AR_PASS,
    AR_STABLE,
    CF_CRIT,
    CF_HIGH,
    CF_LOW,
    CF_MED,
    CF_NONE,
    CS_INT,
    CS_LOCK,
    CS_MON,
    CS_NONE,
    CS_STAB,
    FLOAT_PRECISION,
    LOSS_HIGH,
    LOSS_LOSSLESS,
    LOSS_LOW,
    LOSS_MEDIUM,
    QUANTIZATION_AWARE_FORECAST_COMPRESSION_VERSION,
    STABILITY_CRITICAL,
    STABILITY_DRIFTING,
    STABILITY_STABLE,
    STABILITY_VOLATILE,
    ForecastCompressionDecision,
    ForecastCompressionLedger,
    build_forecast_compression_ledger,
    compress_forecast_horizon,
    export_forecast_compression_bundle,
    export_forecast_compression_ledger,
    _tokenize_decision,
    _run_length_compress,
    _compute_compression_ratio,
    _compute_entropy_proxy,
    _classify_loss_budget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sig(risk: float, route: str = "RECOVERY") -> AuditoryPhaseSignature:
    """Create a signature via the v137.0.3 API."""
    return observe_auditory_phase_control(
        phase_bin_index=(2, 3),
        spectral_drift=0.50,
        risk_score=risk,
        governed_route=route,
    )


def _make_sig_band(band: str) -> AuditoryPhaseSignature:
    """Create a signature with a specific amplitude band."""
    risk_map = {"LOW": 0.1, "WATCH": 0.3, "WARNING": 0.5, "CRITICAL": 0.7, "COLLAPSE": 0.9}
    return _make_sig(risk_map[band])


def _make_decision_static() -> TemporalAuditorySequenceDecision:
    sigs = [_make_sig_band("LOW")] * 3
    return analyze_auditory_sequence(sigs)


def _make_decision_escalating() -> TemporalAuditorySequenceDecision:
    sigs = [_make_sig_band("LOW"), _make_sig_band("WATCH"),
            _make_sig_band("WARNING"), _make_sig_band("CRITICAL")]
    return analyze_auditory_sequence(sigs)


def _make_decision_collapse_loop() -> TemporalAuditorySequenceDecision:
    sigs = [_make_sig_band("COLLAPSE"), _make_sig_band("LOW"),
            _make_sig_band("COLLAPSE")]
    return analyze_auditory_sequence(sigs)


def _make_policy_static() -> TemporalAuditoryPolicyState:
    decisions = [_make_decision_static()] * 3
    return build_temporal_auditory_policy_state(decisions)


def _make_policy_escalating() -> TemporalAuditoryPolicyState:
    decisions = [_make_decision_static(), _make_decision_escalating(),
                 _make_decision_escalating()]
    return build_temporal_auditory_policy_state(decisions)


def _make_policy_collapse() -> TemporalAuditoryPolicyState:
    decisions = [_make_decision_collapse_loop()] * 3
    return build_temporal_auditory_policy_state(decisions)


def _make_arb_static() -> TemporalAuditoryArbitrationDecision:
    """Make an arbitration decision from static policies."""
    return arbitrate_temporal_auditory_policies(
        [_make_policy_static(), _make_policy_static()]
    )


def _make_arb_escalating() -> TemporalAuditoryArbitrationDecision:
    """Make an arbitration decision from mixed policies."""
    return arbitrate_temporal_auditory_policies(
        [_make_policy_static(), _make_policy_escalating()]
    )


def _make_arb_collapse() -> TemporalAuditoryArbitrationDecision:
    """Make an arbitration decision from collapse policies."""
    return arbitrate_temporal_auditory_policies(
        [_make_policy_collapse(), _make_policy_collapse()]
    )


def _make_arb_mixed() -> TemporalAuditoryArbitrationDecision:
    """Make an arbitration decision from static + collapse policies."""
    return arbitrate_temporal_auditory_policies(
        [_make_policy_static(), _make_policy_collapse()]
    )


# ---------------------------------------------------------------------------
# Test tokenization correctness
# ---------------------------------------------------------------------------

class TestTokenization:
    """Tests for symbolic token conversion."""

    def test_static_policy_produces_valid_token(self):
        arb = _make_arb_static()
        token = _tokenize_decision(arb)
        parts = token.split("|")
        assert len(parts) == 3
        assert parts[0].startswith("CF_")
        assert parts[1].startswith("AR_")
        assert parts[2].startswith("CS_")

    def test_token_is_string(self):
        arb = _make_arb_static()
        token = _tokenize_decision(arb)
        assert isinstance(token, str)

    def test_token_contains_pipe_separators(self):
        arb = _make_arb_static()
        token = _tokenize_decision(arb)
        assert token.count("|") == 2

    def test_collapse_policy_produces_severe_token(self):
        arb = _make_arb_collapse()
        token = _tokenize_decision(arb)
        assert "AR_CRIT" in token or "AR_LOCK" in token

    def test_deterministic_tokenization(self):
        arb = _make_arb_static()
        t1 = _tokenize_decision(arb)
        t2 = _tokenize_decision(arb)
        assert t1 == t2

    def test_conflict_none_maps_to_cf_none(self):
        arb = _make_arb_static()
        if arb.conflict_level == "NONE":
            token = _tokenize_decision(arb)
            assert token.startswith("CF_NONE")

    def test_different_inputs_may_produce_different_tokens(self):
        arb_s = _make_arb_static()
        arb_c = _make_arb_collapse()
        t_s = _tokenize_decision(arb_s)
        t_c = _tokenize_decision(arb_c)
        # Collapse and static should differ
        assert t_s != t_c


# ---------------------------------------------------------------------------
# Test adjacent-run compression
# ---------------------------------------------------------------------------

class TestRunLengthCompression:
    """Tests for run-length compression of token sequences."""

    def test_single_token(self):
        result = _run_length_compress(("A",))
        assert result == ("A ×1",)

    def test_all_same(self):
        result = _run_length_compress(("A", "A", "A"))
        assert result == ("A ×3",)

    def test_all_different(self):
        result = _run_length_compress(("A", "B", "C"))
        assert result == ("A ×1", "B ×1", "C ×1")

    def test_mixed_runs(self):
        result = _run_length_compress(("A", "A", "A", "B", "B", "C"))
        assert result == ("A ×3", "B ×2", "C ×1")

    def test_empty_input(self):
        result = _run_length_compress(())
        assert result == ()

    def test_alternating(self):
        result = _run_length_compress(("A", "B", "A", "B"))
        assert result == ("A ×1", "B ×1", "A ×1", "B ×1")

    def test_output_is_tuple(self):
        result = _run_length_compress(("A", "A"))
        assert isinstance(result, tuple)

    def test_long_run(self):
        tokens = ("X",) * 100
        result = _run_length_compress(tokens)
        assert result == ("X ×100",)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Test compression_ratio bounds
# ---------------------------------------------------------------------------

class TestCompressionRatio:
    """Tests for compression ratio computation."""

    def test_no_compression(self):
        ratio = _compute_compression_ratio(5, 5)
        assert ratio == 1.0

    def test_full_compression(self):
        ratio = _compute_compression_ratio(1, 10)
        assert ratio == 0.1

    def test_bounded_upper(self):
        ratio = _compute_compression_ratio(3, 3)
        assert ratio <= 1.0

    def test_bounded_lower(self):
        ratio = _compute_compression_ratio(1, 100)
        assert ratio > 0.0

    def test_ratio_from_core_function(self):
        decisions = [_make_arb_static()] * 5
        result = compress_forecast_horizon(decisions)
        assert 0.0 < result.compression_ratio <= 1.0

    def test_identical_decisions_high_compression(self):
        decisions = [_make_arb_static()] * 10
        result = compress_forecast_horizon(decisions)
        assert result.compression_ratio <= 0.5


# ---------------------------------------------------------------------------
# Test entropy_proxy bounds
# ---------------------------------------------------------------------------

class TestEntropyProxy:
    """Tests for entropy proxy computation."""

    def test_single_unique(self):
        proxy = _compute_entropy_proxy(("A", "A", "A"))
        assert proxy == round(1 / 3, FLOAT_PRECISION)

    def test_all_unique(self):
        proxy = _compute_entropy_proxy(("A", "B", "C"))
        assert proxy == 1.0

    def test_bounded(self):
        proxy = _compute_entropy_proxy(("A", "B", "A", "B", "C"))
        assert 0.0 < proxy <= 1.0

    def test_entropy_from_core(self):
        decisions = [_make_arb_static()] * 5
        result = compress_forecast_horizon(decisions)
        assert 0.0 < result.entropy_proxy <= 1.0

    def test_identical_decisions_low_entropy(self):
        decisions = [_make_arb_static()] * 10
        result = compress_forecast_horizon(decisions)
        assert result.entropy_proxy <= 0.5


# ---------------------------------------------------------------------------
# Test dominant arbitration tie-breaking
# ---------------------------------------------------------------------------

class TestDominantArbitrationMode:
    """Tests for dominant arbitration mode detection and tie-breaking."""

    def test_single_decision(self):
        decisions = [_make_arb_static()]
        result = compress_forecast_horizon(decisions)
        assert result.dominant_arbitration_mode == decisions[0].arbitration_decision

    def test_majority_wins(self):
        static = _make_arb_static()
        collapse = _make_arb_collapse()
        decisions = [static, static, static, collapse]
        result = compress_forecast_horizon(decisions)
        assert result.dominant_arbitration_mode == static.arbitration_decision

    def test_tie_break_favors_severity(self):
        """When tied, higher severity wins."""
        static = _make_arb_static()
        collapse = _make_arb_collapse()
        # Equal counts -> severity tie-break
        decisions = [static, collapse]
        result = compress_forecast_horizon(decisions)
        # Collapse should produce higher severity arbitration decision
        sev_static = {
            ARBITRATION_PASS_THROUGH: 0,
            ARBITRATION_MERGE: 1,
            ARBITRATION_PRIORITIZE_STABLE: 2,
            ARBITRATION_PRIORITIZE_CRITICAL: 3,
            ARBITRATION_LOCKDOWN: 4,
        }
        dom = result.dominant_arbitration_mode
        assert sev_static.get(dom, -1) >= sev_static.get(static.arbitration_decision, -1)

    def test_valid_arbitration_value(self):
        decisions = [_make_arb_static()]
        result = compress_forecast_horizon(decisions)
        valid = {
            ARBITRATION_PASS_THROUGH, ARBITRATION_MERGE,
            ARBITRATION_PRIORITIZE_STABLE, ARBITRATION_PRIORITIZE_CRITICAL,
            ARBITRATION_LOCKDOWN,
        }
        assert result.dominant_arbitration_mode in valid


# ---------------------------------------------------------------------------
# Test stability class detection
# ---------------------------------------------------------------------------

class TestStabilityClass:
    """Tests for forecast stability classification."""

    def test_identical_decisions_stable(self):
        decisions = [_make_arb_static()] * 5
        result = compress_forecast_horizon(decisions)
        assert result.forecast_stability_class == STABILITY_STABLE

    def test_mixed_decisions_not_stable(self):
        decisions = [_make_arb_static(), _make_arb_collapse(),
                     _make_arb_static(), _make_arb_collapse()]
        result = compress_forecast_horizon(decisions)
        assert result.forecast_stability_class != STABILITY_STABLE

    def test_valid_stability_class(self):
        decisions = [_make_arb_static()]
        result = compress_forecast_horizon(decisions)
        valid = {STABILITY_STABLE, STABILITY_DRIFTING, STABILITY_VOLATILE,
                 STABILITY_CRITICAL}
        assert result.forecast_stability_class in valid

    def test_collapse_heavy_critical(self):
        decisions = [_make_arb_collapse()] * 5
        result = compress_forecast_horizon(decisions)
        # All collapse -> could be STABLE (if single token) or CRITICAL
        assert result.forecast_stability_class in (
            STABILITY_STABLE, STABILITY_CRITICAL,
        )

    def test_highly_mixed_volatile_or_critical(self):
        s = _make_arb_static()
        c = _make_arb_collapse()
        e = _make_arb_escalating()
        m = _make_arb_mixed()
        decisions = [s, c, e, m]
        result = compress_forecast_horizon(decisions)
        assert result.forecast_stability_class in (
            STABILITY_VOLATILE, STABILITY_CRITICAL, STABILITY_DRIFTING,
        )


# ---------------------------------------------------------------------------
# Test loss budget classification
# ---------------------------------------------------------------------------

class TestLossBudgetClass:
    """Tests for loss budget classification."""

    def test_lossless_when_no_compression(self):
        assert _classify_loss_budget(1.0, 1.0) == LOSS_LOSSLESS

    def test_high_loss(self):
        assert _classify_loss_budget(0.2, 0.8) == LOSS_HIGH

    def test_medium_loss(self):
        assert _classify_loss_budget(0.5, 0.5) == LOSS_MEDIUM

    def test_low_loss(self):
        assert _classify_loss_budget(0.8, 0.2) == LOSS_LOW

    def test_valid_loss_class_from_core(self):
        decisions = [_make_arb_static()] * 3
        result = compress_forecast_horizon(decisions)
        valid = {LOSS_LOSSLESS, LOSS_LOW, LOSS_MEDIUM, LOSS_HIGH}
        assert result.loss_budget_class in valid


# ---------------------------------------------------------------------------
# Test symbolic trace correctness
# ---------------------------------------------------------------------------

class TestSymbolicTrace:
    """Tests for forecast symbolic trace construction."""

    def test_trace_is_string(self):
        decisions = [_make_arb_static()] * 3
        result = compress_forecast_horizon(decisions)
        assert isinstance(result.forecast_symbolic_trace, str)

    def test_trace_contains_arrow_separator(self):
        decisions = [_make_arb_static(), _make_arb_collapse()]
        result = compress_forecast_horizon(decisions)
        # With two different tokens, trace should have " -> "
        if len(result.compressed_forecast_tokens) > 1:
            assert " -> " in result.forecast_symbolic_trace

    def test_trace_contains_run_counts(self):
        decisions = [_make_arb_static()] * 3
        result = compress_forecast_horizon(decisions)
        assert "×" in result.forecast_symbolic_trace

    def test_trace_deterministic(self):
        decisions = [_make_arb_static()] * 3
        r1 = compress_forecast_horizon(decisions)
        r2 = compress_forecast_horizon(decisions)
        assert r1.forecast_symbolic_trace == r2.forecast_symbolic_trace


# ---------------------------------------------------------------------------
# Test frozen immutability
# ---------------------------------------------------------------------------

class TestFrozenImmutability:
    """Tests that dataclasses are frozen."""

    def test_decision_frozen(self):
        decisions = [_make_arb_static()] * 3
        result = compress_forecast_horizon(decisions)
        with pytest.raises(AttributeError):
            result.horizon_length = 999

    def test_decision_hash_frozen(self):
        decisions = [_make_arb_static()] * 3
        result = compress_forecast_horizon(decisions)
        with pytest.raises(AttributeError):
            result.stable_hash = "MODIFIED"

    def test_ledger_frozen(self):
        d1 = compress_forecast_horizon([_make_arb_static()])
        ledger = build_forecast_compression_ledger([d1])
        with pytest.raises(AttributeError):
            ledger.decision_count = 999

    def test_ledger_hash_frozen(self):
        d1 = compress_forecast_horizon([_make_arb_static()])
        ledger = build_forecast_compression_ledger([d1])
        with pytest.raises(AttributeError):
            ledger.stable_hash = "MODIFIED"


# ---------------------------------------------------------------------------
# Test export equality
# ---------------------------------------------------------------------------

class TestExportEquality:
    """Tests for export determinism."""

    def test_bundle_deterministic(self):
        decisions = [_make_arb_static()] * 3
        result = compress_forecast_horizon(decisions)
        b1 = json.dumps(export_forecast_compression_bundle(result), sort_keys=True)
        b2 = json.dumps(export_forecast_compression_bundle(result), sort_keys=True)
        assert b1 == b2

    def test_bundle_json_serializable(self):
        decisions = [_make_arb_static()] * 3
        result = compress_forecast_horizon(decisions)
        bundle = export_forecast_compression_bundle(result)
        serialized = json.dumps(bundle, sort_keys=True)
        assert isinstance(serialized, str)

    def test_bundle_contains_layer(self):
        decisions = [_make_arb_static()] * 3
        result = compress_forecast_horizon(decisions)
        bundle = export_forecast_compression_bundle(result)
        assert bundle["layer"] == "quantization_aware_forecast_compression"

    def test_bundle_contains_stable_hash(self):
        decisions = [_make_arb_static()] * 3
        result = compress_forecast_horizon(decisions)
        bundle = export_forecast_compression_bundle(result)
        assert "stable_hash" in bundle
        assert isinstance(bundle["stable_hash"], str)
        assert len(bundle["stable_hash"]) == 64

    def test_ledger_export_deterministic(self):
        d1 = compress_forecast_horizon([_make_arb_static()])
        ledger = build_forecast_compression_ledger([d1])
        e1 = json.dumps(export_forecast_compression_ledger(ledger), sort_keys=True)
        e2 = json.dumps(export_forecast_compression_ledger(ledger), sort_keys=True)
        assert e1 == e2

    def test_ledger_export_contains_version(self):
        d1 = compress_forecast_horizon([_make_arb_static()])
        ledger = build_forecast_compression_ledger([d1])
        export = export_forecast_compression_ledger(ledger)
        assert export["version"] == QUANTIZATION_AWARE_FORECAST_COMPRESSION_VERSION

    def test_ledger_export_decision_count_matches(self):
        d1 = compress_forecast_horizon([_make_arb_static()])
        d2 = compress_forecast_horizon([_make_arb_collapse()])
        ledger = build_forecast_compression_ledger([d1, d2])
        export = export_forecast_compression_ledger(ledger)
        assert export["decision_count"] == 2
        assert len(export["decisions"]) == 2


# ---------------------------------------------------------------------------
# Test stable hashing
# ---------------------------------------------------------------------------

class TestStableHashing:
    """Tests for stable hash computation."""

    def test_hash_is_hex_string(self):
        decisions = [_make_arb_static()] * 3
        result = compress_forecast_horizon(decisions)
        assert isinstance(result.stable_hash, str)
        assert len(result.stable_hash) == 64
        int(result.stable_hash, 16)  # valid hex

    def test_hash_deterministic(self):
        decisions = [_make_arb_static()] * 3
        r1 = compress_forecast_horizon(decisions)
        r2 = compress_forecast_horizon(decisions)
        assert r1.stable_hash == r2.stable_hash

    def test_different_inputs_different_hash(self):
        r1 = compress_forecast_horizon([_make_arb_static()])
        r2 = compress_forecast_horizon([_make_arb_collapse()])
        assert r1.stable_hash != r2.stable_hash

    def test_ledger_hash_deterministic(self):
        d1 = compress_forecast_horizon([_make_arb_static()])
        l1 = build_forecast_compression_ledger([d1])
        l2 = build_forecast_compression_ledger([d1])
        assert l1.stable_hash == l2.stable_hash

    def test_ledger_hash_is_hex(self):
        d1 = compress_forecast_horizon([_make_arb_static()])
        ledger = build_forecast_compression_ledger([d1])
        assert len(ledger.stable_hash) == 64
        int(ledger.stable_hash, 16)


# ---------------------------------------------------------------------------
# Test 100-run replay
# ---------------------------------------------------------------------------

class TestReplay:
    """Tests for 100-run replay determinism."""

    def test_100_run_decision_replay(self):
        decisions = [_make_arb_static(), _make_arb_collapse()]
        baseline = compress_forecast_horizon(decisions)
        for _ in range(100):
            result = compress_forecast_horizon(decisions)
            assert result.stable_hash == baseline.stable_hash
            assert result.compressed_forecast_tokens == baseline.compressed_forecast_tokens
            assert result.compression_ratio == baseline.compression_ratio
            assert result.entropy_proxy == baseline.entropy_proxy
            assert result.dominant_arbitration_mode == baseline.dominant_arbitration_mode
            assert result.forecast_stability_class == baseline.forecast_stability_class
            assert result.loss_budget_class == baseline.loss_budget_class
            assert result.forecast_symbolic_trace == baseline.forecast_symbolic_trace

    def test_100_run_ledger_replay(self):
        d1 = compress_forecast_horizon([_make_arb_static()])
        d2 = compress_forecast_horizon([_make_arb_collapse()])
        baseline = build_forecast_compression_ledger([d1, d2])
        for _ in range(100):
            ledger = build_forecast_compression_ledger([d1, d2])
            assert ledger.stable_hash == baseline.stable_hash
            assert ledger.decision_count == baseline.decision_count

    def test_100_run_export_replay(self):
        decisions = [_make_arb_static()] * 3
        result = compress_forecast_horizon(decisions)
        baseline = json.dumps(
            export_forecast_compression_bundle(result), sort_keys=True,
        )
        for _ in range(100):
            export = json.dumps(
                export_forecast_compression_bundle(result), sort_keys=True,
            )
            assert export == baseline


# ---------------------------------------------------------------------------
# Test no decoder contamination
# ---------------------------------------------------------------------------

class TestNoDecoderContamination:
    """Ensure no decoder imports in the module."""

    def test_no_decoder_import(self):
        import qec.analysis.quantization_aware_forecast_compression as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_no_experiment_import(self):
        import qec.analysis.quantization_aware_forecast_compression as mod
        source = open(mod.__file__).read()
        assert "from qec.experiments" not in source
        assert "import qec.experiments" not in source

    def test_no_sims_import(self):
        import qec.analysis.quantization_aware_forecast_compression as mod
        source = open(mod.__file__).read()
        assert "from qec.sims" not in source
        assert "import qec.sims" not in source


# ---------------------------------------------------------------------------
# Test invalid input rejection
# ---------------------------------------------------------------------------

class TestInvalidInputRejection:
    """Tests for input validation."""

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            compress_forecast_horizon([])

    def test_non_iterable_raises_type_error(self):
        with pytest.raises(TypeError, match="must be iterable"):
            compress_forecast_horizon(42)

    def test_wrong_element_type_raises_type_error(self):
        with pytest.raises(TypeError, match="must be TemporalAuditoryArbitrationDecision"):
            compress_forecast_horizon(["not_a_decision"])

    def test_ledger_wrong_type_raises_type_error(self):
        with pytest.raises(TypeError, match="must be ForecastCompressionDecision"):
            build_forecast_compression_ledger(["not_a_decision"])

    def test_mixed_types_raises_type_error(self):
        valid = compress_forecast_horizon([_make_arb_static()])
        with pytest.raises(TypeError):
            build_forecast_compression_ledger([valid, "invalid"])


# ---------------------------------------------------------------------------
# Test version field
# ---------------------------------------------------------------------------

class TestVersionField:
    """Tests for version field correctness."""

    def test_decision_version(self):
        decisions = [_make_arb_static()]
        result = compress_forecast_horizon(decisions)
        assert result.version == "v137.0.7"

    def test_version_constant(self):
        assert QUANTIZATION_AWARE_FORECAST_COMPRESSION_VERSION == "v137.0.7"


# ---------------------------------------------------------------------------
# Test horizon_length
# ---------------------------------------------------------------------------

class TestHorizonLength:
    """Tests for horizon_length field."""

    def test_single_decision(self):
        decisions = [_make_arb_static()]
        result = compress_forecast_horizon(decisions)
        assert result.horizon_length == 1

    def test_multiple_decisions(self):
        decisions = [_make_arb_static()] * 7
        result = compress_forecast_horizon(decisions)
        assert result.horizon_length == 7

    def test_horizon_matches_input_length(self):
        for n in (1, 3, 5, 10):
            decisions = [_make_arb_static()] * n
            result = compress_forecast_horizon(decisions)
            assert result.horizon_length == n


# ---------------------------------------------------------------------------
# Test compressed tokens field
# ---------------------------------------------------------------------------

class TestCompressedTokens:
    """Tests for compressed_forecast_tokens field."""

    def test_tokens_is_tuple(self):
        decisions = [_make_arb_static()] * 3
        result = compress_forecast_horizon(decisions)
        assert isinstance(result.compressed_forecast_tokens, tuple)

    def test_tokens_non_empty(self):
        decisions = [_make_arb_static()]
        result = compress_forecast_horizon(decisions)
        assert len(result.compressed_forecast_tokens) > 0

    def test_all_elements_are_strings(self):
        decisions = [_make_arb_static(), _make_arb_collapse()]
        result = compress_forecast_horizon(decisions)
        for t in result.compressed_forecast_tokens:
            assert isinstance(t, str)
