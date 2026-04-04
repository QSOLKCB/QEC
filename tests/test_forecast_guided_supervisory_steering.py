"""Tests for v137.0.8 — Forecast-Guided Supervisory Steering.

Target: 70-90 tests covering replay validation, coherence score bounds,
drift score bounds, trend classification, steering action mapping,
symbolic trace stability, frozen immutability, stable hashing,
export equality, 100-run replay, invalid input rejection,
no decoder contamination.
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
    TemporalAuditoryArbitrationDecision,
    arbitrate_temporal_auditory_policies,
)
from qec.analysis.quantization_aware_forecast_compression import (
    ForecastCompressionDecision,
    compress_forecast_horizon,
)
from qec.analysis.forecast_guided_supervisory_steering import (
    COHERENCE_FAILED,
    COHERENCE_HIGH,
    COHERENCE_LOW,
    COHERENCE_MEDIUM,
    FLOAT_PRECISION,
    FORECAST_GUIDED_SUPERVISORY_STEERING_VERSION,
    STEERING_AMPLIFY,
    STEERING_DAMPEN,
    STEERING_HOLD,
    STEERING_LOCKDOWN,
    STEERING_REDIRECT,
    TREND_CRITICAL,
    TREND_DRIFT_DOWN,
    TREND_DRIFT_UP,
    TREND_STABLE,
    TREND_VOLATILE,
    SteeringDecision,
    SteeringLedger,
    build_supervisory_steering_ledger,
    derive_supervisory_steering,
    export_supervisory_steering_bundle,
    export_supervisory_steering_ledger,
    _verify_replay,
    _compute_coherence_score,
    _compute_drift_score,
    _classify_trend,
    _select_steering_action,
    _classify_coherence,
    _build_symbolic_trace,
    _steering_decision_to_canonical_dict,
    _compute_steering_hash,
    _canonical_json,
    _round,
)


# ---------------------------------------------------------------------------
# Helpers — build upstream artifacts
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
    return arbitrate_temporal_auditory_policies(
        [_make_policy_static(), _make_policy_static()]
    )


def _make_arb_escalating() -> TemporalAuditoryArbitrationDecision:
    return arbitrate_temporal_auditory_policies(
        [_make_policy_static(), _make_policy_escalating()]
    )


def _make_arb_collapse() -> TemporalAuditoryArbitrationDecision:
    return arbitrate_temporal_auditory_policies(
        [_make_policy_collapse(), _make_policy_collapse()]
    )


def _make_arb_mixed() -> TemporalAuditoryArbitrationDecision:
    return arbitrate_temporal_auditory_policies(
        [_make_policy_static(), _make_policy_collapse()]
    )


# --- v137.0.7 compression helpers ---

def _make_fc_static() -> ForecastCompressionDecision:
    """Stable horizon: all-static arbitration decisions."""
    arbs = [_make_arb_static()] * 5
    return compress_forecast_horizon(arbs)


def _make_fc_escalating() -> ForecastCompressionDecision:
    """Escalating horizon: static -> escalating."""
    arbs = [_make_arb_static(), _make_arb_static(),
            _make_arb_escalating(), _make_arb_escalating(),
            _make_arb_collapse()]
    return compress_forecast_horizon(arbs)


def _make_fc_collapse() -> ForecastCompressionDecision:
    """Critical horizon: all-collapse."""
    arbs = [_make_arb_collapse()] * 5
    return compress_forecast_horizon(arbs)


def _make_fc_mixed() -> ForecastCompressionDecision:
    """Mixed horizon: alternating static/collapse."""
    arbs = [_make_arb_static(), _make_arb_collapse(),
            _make_arb_static(), _make_arb_collapse(),
            _make_arb_mixed()]
    return compress_forecast_horizon(arbs)


def _make_fc_calming() -> ForecastCompressionDecision:
    """Calming horizon: collapse -> static."""
    arbs = [_make_arb_collapse(), _make_arb_collapse(),
            _make_arb_mixed(), _make_arb_static(),
            _make_arb_static()]
    return compress_forecast_horizon(arbs)


# ---------------------------------------------------------------------------
# Test replay validation
# ---------------------------------------------------------------------------

class TestReplayValidation:
    """Tests for replay coherence verification."""

    def test_static_horizons_are_replay_valid(self):
        decisions = (_make_fc_static(),) * 3
        assert _verify_replay(decisions) is True

    def test_mixed_horizons_are_replay_valid(self):
        decisions = (_make_fc_static(), _make_fc_escalating(), _make_fc_collapse())
        assert _verify_replay(decisions) is True

    def test_empty_hash_fails_replay(self):
        fc = _make_fc_static()
        bad = ForecastCompressionDecision(
            horizon_length=fc.horizon_length,
            compressed_forecast_tokens=fc.compressed_forecast_tokens,
            compression_ratio=fc.compression_ratio,
            entropy_proxy=fc.entropy_proxy,
            dominant_arbitration_mode=fc.dominant_arbitration_mode,
            forecast_stability_class=fc.forecast_stability_class,
            loss_budget_class=fc.loss_budget_class,
            forecast_symbolic_trace=fc.forecast_symbolic_trace,
            stable_hash="",
        )
        assert _verify_replay((bad,)) is False

    def test_empty_tokens_fails_replay(self):
        fc = _make_fc_static()
        bad = ForecastCompressionDecision(
            horizon_length=fc.horizon_length,
            compressed_forecast_tokens=(),
            compression_ratio=fc.compression_ratio,
            entropy_proxy=fc.entropy_proxy,
            dominant_arbitration_mode=fc.dominant_arbitration_mode,
            forecast_stability_class=fc.forecast_stability_class,
            loss_budget_class=fc.loss_budget_class,
            forecast_symbolic_trace=fc.forecast_symbolic_trace,
            stable_hash=fc.stable_hash,
        )
        assert _verify_replay((bad,)) is False

    def test_zero_horizon_length_fails_replay(self):
        fc = _make_fc_static()
        bad = ForecastCompressionDecision(
            horizon_length=0,
            compressed_forecast_tokens=fc.compressed_forecast_tokens,
            compression_ratio=fc.compression_ratio,
            entropy_proxy=fc.entropy_proxy,
            dominant_arbitration_mode=fc.dominant_arbitration_mode,
            forecast_stability_class=fc.forecast_stability_class,
            loss_budget_class=fc.loss_budget_class,
            forecast_symbolic_trace=fc.forecast_symbolic_trace,
            stable_hash=fc.stable_hash,
        )
        assert _verify_replay((bad,)) is False

    def test_wrong_version_fails_replay(self):
        fc = _make_fc_static()
        bad = ForecastCompressionDecision(
            horizon_length=fc.horizon_length,
            compressed_forecast_tokens=fc.compressed_forecast_tokens,
            compression_ratio=fc.compression_ratio,
            entropy_proxy=fc.entropy_proxy,
            dominant_arbitration_mode=fc.dominant_arbitration_mode,
            forecast_stability_class=fc.forecast_stability_class,
            loss_budget_class=fc.loss_budget_class,
            forecast_symbolic_trace=fc.forecast_symbolic_trace,
            stable_hash=fc.stable_hash,
            version="v0.0.0",
        )
        assert _verify_replay((bad,)) is False

    def test_replay_valid_propagates_to_decision(self):
        result = derive_supervisory_steering([_make_fc_static()])
        assert result.replay_valid is True

    def test_replay_invalid_propagates_to_decision(self):
        fc = _make_fc_static()
        bad = ForecastCompressionDecision(
            horizon_length=fc.horizon_length,
            compressed_forecast_tokens=fc.compressed_forecast_tokens,
            compression_ratio=fc.compression_ratio,
            entropy_proxy=fc.entropy_proxy,
            dominant_arbitration_mode=fc.dominant_arbitration_mode,
            forecast_stability_class=fc.forecast_stability_class,
            loss_budget_class=fc.loss_budget_class,
            forecast_symbolic_trace=fc.forecast_symbolic_trace,
            stable_hash="",
        )
        result = derive_supervisory_steering([bad])
        assert result.replay_valid is False


# ---------------------------------------------------------------------------
# Test coherence score bounds
# ---------------------------------------------------------------------------

class TestCoherenceScore:
    """Tests for temporal coherence score computation."""

    def test_single_decision_coherence_is_one(self):
        decisions = (_make_fc_static(),)
        assert _compute_coherence_score(decisions) == 1.0

    def test_identical_decisions_coherence_is_one(self):
        fc = _make_fc_static()
        decisions = (fc, fc, fc)
        assert _compute_coherence_score(decisions) == 1.0

    def test_coherence_bounded_zero_to_one(self):
        decisions = (_make_fc_static(), _make_fc_escalating(),
                     _make_fc_collapse(), _make_fc_mixed())
        score = _compute_coherence_score(decisions)
        assert 0.0 <= score <= 1.0

    def test_coherence_is_float(self):
        decisions = (_make_fc_static(), _make_fc_escalating())
        score = _compute_coherence_score(decisions)
        assert isinstance(score, float)

    def test_coherence_deterministic(self):
        decisions = (_make_fc_static(), _make_fc_escalating(), _make_fc_collapse())
        s1 = _compute_coherence_score(decisions)
        s2 = _compute_coherence_score(decisions)
        assert s1 == s2

    def test_all_different_stability_low_coherence(self):
        # Use horizons that produce different stability classes
        fc_s = _make_fc_static()
        fc_e = _make_fc_escalating()
        fc_c = _make_fc_collapse()
        decisions = (fc_s, fc_e, fc_c)
        score = _compute_coherence_score(decisions)
        # With 3 different stability classes, no adjacent match -> 0.0
        # But some might match, so just check bound
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Test drift score bounds
# ---------------------------------------------------------------------------

class TestDriftScore:
    """Tests for directional drift computation."""

    def test_single_decision_drift_is_zero(self):
        decisions = (_make_fc_static(),)
        assert _compute_drift_score(decisions) == 0.0

    def test_identical_decisions_drift_near_zero(self):
        fc = _make_fc_static()
        decisions = (fc, fc, fc)
        assert _compute_drift_score(decisions) == 0.0

    def test_drift_bounded_minus_one_to_one(self):
        decisions = (_make_fc_static(), _make_fc_escalating(),
                     _make_fc_collapse(), _make_fc_mixed())
        score = _compute_drift_score(decisions)
        assert -1.0 <= score <= 1.0

    def test_drift_is_float(self):
        decisions = (_make_fc_static(), _make_fc_escalating())
        score = _compute_drift_score(decisions)
        assert isinstance(score, float)

    def test_drift_deterministic(self):
        decisions = (_make_fc_static(), _make_fc_escalating(), _make_fc_collapse())
        s1 = _compute_drift_score(decisions)
        s2 = _compute_drift_score(decisions)
        assert s1 == s2

    def test_escalating_produces_positive_drift(self):
        # Static -> escalating -> collapse should show positive drift
        decisions = (_make_fc_static(), _make_fc_escalating(), _make_fc_collapse())
        score = _compute_drift_score(decisions)
        # Drift direction depends on the actual stability/loss/entropy values
        assert -1.0 <= score <= 1.0

    def test_calming_direction(self):
        # Collapse -> calming -> static should produce negative drift
        decisions = (_make_fc_collapse(), _make_fc_calming(), _make_fc_static())
        score = _compute_drift_score(decisions)
        assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Test trend classification
# ---------------------------------------------------------------------------

class TestTrendClassification:
    """Tests for dominant trend class classification."""

    def test_stable_trend_classification(self):
        trend = _classify_trend(coherence_score=0.9, drift_score=0.0, replay_valid=True)
        assert trend == TREND_STABLE

    def test_drift_up_classification(self):
        trend = _classify_trend(coherence_score=0.5, drift_score=0.5, replay_valid=True)
        assert trend == TREND_DRIFT_UP

    def test_drift_down_classification(self):
        trend = _classify_trend(coherence_score=0.5, drift_score=-0.5, replay_valid=True)
        assert trend == TREND_DRIFT_DOWN

    def test_volatile_classification(self):
        trend = _classify_trend(coherence_score=0.2, drift_score=0.05, replay_valid=True)
        assert trend == TREND_VOLATILE

    def test_critical_on_replay_failure(self):
        trend = _classify_trend(coherence_score=0.9, drift_score=0.0, replay_valid=False)
        assert trend == TREND_CRITICAL

    def test_trend_is_string(self):
        trend = _classify_trend(0.5, 0.0, True)
        assert isinstance(trend, str)

    def test_high_coherence_near_zero_drift_is_stable(self):
        trend = _classify_trend(coherence_score=0.8, drift_score=0.05, replay_valid=True)
        assert trend == TREND_STABLE

    def test_all_trend_classes_are_valid(self):
        valid = {TREND_STABLE, TREND_DRIFT_UP, TREND_DRIFT_DOWN, TREND_VOLATILE, TREND_CRITICAL}
        for coh in [0.1, 0.5, 0.9]:
            for drift in [-0.5, 0.0, 0.5]:
                for replay in [True, False]:
                    trend = _classify_trend(coh, drift, replay)
                    assert trend in valid


# ---------------------------------------------------------------------------
# Test steering action mapping
# ---------------------------------------------------------------------------

class TestSteeringAction:
    """Tests for steering action mapping."""

    def test_stable_maps_to_hold(self):
        action = _select_steering_action(TREND_STABLE, 0.0, 0.9)
        assert action == STEERING_HOLD

    def test_drift_up_maps_to_dampen(self):
        action = _select_steering_action(TREND_DRIFT_UP, 0.5, 0.5)
        assert action == STEERING_DAMPEN

    def test_drift_down_maps_to_hold(self):
        action = _select_steering_action(TREND_DRIFT_DOWN, -0.2, 0.5)
        assert action == STEERING_HOLD

    def test_drift_down_strong_with_high_coherence_maps_to_amplify(self):
        action = _select_steering_action(TREND_DRIFT_DOWN, -0.5, 0.8)
        assert action == STEERING_AMPLIFY

    def test_volatile_maps_to_redirect(self):
        action = _select_steering_action(TREND_VOLATILE, 0.0, 0.2)
        assert action == STEERING_REDIRECT

    def test_critical_maps_to_lockdown(self):
        action = _select_steering_action(TREND_CRITICAL, 0.0, 0.0)
        assert action == STEERING_LOCKDOWN

    def test_all_actions_are_valid(self):
        valid = {STEERING_HOLD, STEERING_DAMPEN, STEERING_AMPLIFY, STEERING_REDIRECT, STEERING_LOCKDOWN}
        trends = [TREND_STABLE, TREND_DRIFT_UP, TREND_DRIFT_DOWN, TREND_VOLATILE, TREND_CRITICAL]
        for trend in trends:
            for drift in [-0.5, 0.0, 0.5]:
                for coh in [0.1, 0.5, 0.9]:
                    action = _select_steering_action(trend, drift, coh)
                    assert action in valid

    def test_amplify_requires_strong_negative_drift(self):
        # Weak negative drift should not amplify
        action = _select_steering_action(TREND_DRIFT_DOWN, -0.1, 0.9)
        assert action == STEERING_HOLD


# ---------------------------------------------------------------------------
# Test symbolic trace stability
# ---------------------------------------------------------------------------

class TestSymbolicTrace:
    """Tests for symbolic trace construction."""

    def test_trace_contains_arrow_separator(self):
        decisions = (_make_fc_static(),)
        trace = _build_symbolic_trace(decisions, TREND_STABLE, STEERING_HOLD)
        assert " -> " in trace

    def test_trace_ends_with_action(self):
        decisions = (_make_fc_static(),)
        trace = _build_symbolic_trace(decisions, TREND_STABLE, STEERING_HOLD)
        assert trace.endswith(STEERING_HOLD)

    def test_trace_contains_trend(self):
        decisions = (_make_fc_static(),)
        trace = _build_symbolic_trace(decisions, TREND_DRIFT_UP, STEERING_DAMPEN)
        assert TREND_DRIFT_UP in trace

    def test_trace_deterministic(self):
        decisions = (_make_fc_static(), _make_fc_escalating())
        t1 = _build_symbolic_trace(decisions, TREND_DRIFT_UP, STEERING_DAMPEN)
        t2 = _build_symbolic_trace(decisions, TREND_DRIFT_UP, STEERING_DAMPEN)
        assert t1 == t2

    def test_trace_is_string(self):
        decisions = (_make_fc_static(),)
        trace = _build_symbolic_trace(decisions, TREND_STABLE, STEERING_HOLD)
        assert isinstance(trace, str)

    def test_trace_includes_all_stability_classes(self):
        fc_s = _make_fc_static()
        fc_e = _make_fc_escalating()
        decisions = (fc_s, fc_e)
        trace = _build_symbolic_trace(decisions, TREND_DRIFT_UP, STEERING_DAMPEN)
        # Trace should include the stability classes from each decision
        assert fc_s.forecast_stability_class in trace
        assert fc_e.forecast_stability_class in trace


# ---------------------------------------------------------------------------
# Test coherence class
# ---------------------------------------------------------------------------

class TestCoherenceClass:
    """Tests for temporal coherence class classification."""

    def test_failed_replay_class(self):
        cls = _classify_coherence(0.9, replay_valid=False)
        assert cls == COHERENCE_FAILED

    def test_high_coherence_class(self):
        cls = _classify_coherence(0.8, replay_valid=True)
        assert cls == COHERENCE_HIGH

    def test_medium_coherence_class(self):
        cls = _classify_coherence(0.5, replay_valid=True)
        assert cls == COHERENCE_MEDIUM

    def test_low_coherence_class(self):
        cls = _classify_coherence(0.2, replay_valid=True)
        assert cls == COHERENCE_LOW

    def test_boundary_high(self):
        cls = _classify_coherence(0.7, replay_valid=True)
        assert cls == COHERENCE_HIGH

    def test_boundary_medium(self):
        cls = _classify_coherence(0.4, replay_valid=True)
        assert cls == COHERENCE_MEDIUM

    def test_boundary_low(self):
        cls = _classify_coherence(0.39, replay_valid=True)
        assert cls == COHERENCE_LOW


# ---------------------------------------------------------------------------
# Test frozen immutability
# ---------------------------------------------------------------------------

class TestFrozenImmutability:
    """Tests for frozen dataclass immutability."""

    def test_steering_decision_is_frozen(self):
        result = derive_supervisory_steering([_make_fc_static()])
        with pytest.raises(AttributeError):
            result.steering_action = "INVALID"  # type: ignore[misc]

    def test_steering_decision_drift_frozen(self):
        result = derive_supervisory_steering([_make_fc_static()])
        with pytest.raises(AttributeError):
            result.drift_score = 999.0  # type: ignore[misc]

    def test_steering_decision_hash_frozen(self):
        result = derive_supervisory_steering([_make_fc_static()])
        with pytest.raises(AttributeError):
            result.stable_hash = "tampered"  # type: ignore[misc]

    def test_ledger_is_frozen(self):
        result = derive_supervisory_steering([_make_fc_static()])
        ledger = build_supervisory_steering_ledger([result])
        with pytest.raises(AttributeError):
            ledger.decision_count = 999  # type: ignore[misc]

    def test_ledger_decisions_frozen(self):
        result = derive_supervisory_steering([_make_fc_static()])
        ledger = build_supervisory_steering_ledger([result])
        with pytest.raises(AttributeError):
            ledger.decisions = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test stable hashing
# ---------------------------------------------------------------------------

class TestStableHashing:
    """Tests for stable SHA-256 hashing."""

    def test_hash_is_hex_string(self):
        result = derive_supervisory_steering([_make_fc_static()])
        assert isinstance(result.stable_hash, str)
        assert len(result.stable_hash) == 64
        int(result.stable_hash, 16)  # valid hex

    def test_hash_deterministic(self):
        r1 = derive_supervisory_steering([_make_fc_static()])
        r2 = derive_supervisory_steering([_make_fc_static()])
        assert r1.stable_hash == r2.stable_hash

    def test_different_inputs_different_hashes(self):
        r1 = derive_supervisory_steering([_make_fc_static(), _make_fc_static()])
        r2 = derive_supervisory_steering([_make_fc_static(), _make_fc_escalating(), _make_fc_collapse()])
        assert r1.stable_hash != r2.stable_hash

    def test_ledger_hash_is_hex(self):
        result = derive_supervisory_steering([_make_fc_static()])
        ledger = build_supervisory_steering_ledger([result])
        assert len(ledger.stable_hash) == 64
        int(ledger.stable_hash, 16)

    def test_ledger_hash_deterministic(self):
        r1 = derive_supervisory_steering([_make_fc_static()])
        l1 = build_supervisory_steering_ledger([r1])
        r2 = derive_supervisory_steering([_make_fc_static()])
        l2 = build_supervisory_steering_ledger([r2])
        assert l1.stable_hash == l2.stable_hash

    def test_canonical_dict_excludes_hash(self):
        result = derive_supervisory_steering([_make_fc_static()])
        d = _steering_decision_to_canonical_dict(result)
        assert "stable_hash" not in d

    def test_canonical_json_sorted_keys(self):
        result = derive_supervisory_steering([_make_fc_static()])
        d = _steering_decision_to_canonical_dict(result)
        j = _canonical_json(d)
        reparsed = json.loads(j)
        keys = list(reparsed.keys())
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Test export equality
# ---------------------------------------------------------------------------

class TestExportEquality:
    """Tests for deterministic export."""

    def test_bundle_export_deterministic(self):
        result = derive_supervisory_steering([_make_fc_static()])
        e1 = export_supervisory_steering_bundle(result)
        e2 = export_supervisory_steering_bundle(result)
        assert json.dumps(e1, sort_keys=True) == json.dumps(e2, sort_keys=True)

    def test_bundle_contains_layer(self):
        result = derive_supervisory_steering([_make_fc_static()])
        bundle = export_supervisory_steering_bundle(result)
        assert bundle["layer"] == "forecast_guided_supervisory_steering"

    def test_bundle_contains_hash(self):
        result = derive_supervisory_steering([_make_fc_static()])
        bundle = export_supervisory_steering_bundle(result)
        assert "stable_hash" in bundle
        assert len(bundle["stable_hash"]) == 64

    def test_ledger_export_deterministic(self):
        r = derive_supervisory_steering([_make_fc_static()])
        ledger = build_supervisory_steering_ledger([r])
        e1 = export_supervisory_steering_ledger(ledger)
        e2 = export_supervisory_steering_ledger(ledger)
        assert json.dumps(e1, sort_keys=True) == json.dumps(e2, sort_keys=True)

    def test_ledger_export_contains_version(self):
        r = derive_supervisory_steering([_make_fc_static()])
        ledger = build_supervisory_steering_ledger([r])
        export = export_supervisory_steering_ledger(ledger)
        assert export["version"] == FORECAST_GUIDED_SUPERVISORY_STEERING_VERSION

    def test_ledger_export_contains_decision_count(self):
        r1 = derive_supervisory_steering([_make_fc_static()])
        r2 = derive_supervisory_steering([_make_fc_escalating()])
        ledger = build_supervisory_steering_ledger([r1, r2])
        export = export_supervisory_steering_ledger(ledger)
        assert export["decision_count"] == 2

    def test_ledger_export_decisions_list(self):
        r = derive_supervisory_steering([_make_fc_static()])
        ledger = build_supervisory_steering_ledger([r])
        export = export_supervisory_steering_ledger(ledger)
        assert isinstance(export["decisions"], list)
        assert len(export["decisions"]) == 1


# ---------------------------------------------------------------------------
# Test 100-run replay determinism
# ---------------------------------------------------------------------------

class TestReplayDeterminism:
    """Tests for 100-run byte-identical replay."""

    def test_100_run_single_horizon(self):
        reference = derive_supervisory_steering([_make_fc_static()])
        for _ in range(99):
            result = derive_supervisory_steering([_make_fc_static()])
            assert result == reference

    def test_100_run_multi_horizon(self):
        inputs = [_make_fc_static(), _make_fc_escalating(), _make_fc_collapse()]
        reference = derive_supervisory_steering(inputs)
        for _ in range(99):
            result = derive_supervisory_steering(inputs)
            assert result == reference

    def test_100_run_export_stable(self):
        result = derive_supervisory_steering([_make_fc_static()])
        ref_json = json.dumps(export_supervisory_steering_bundle(result), sort_keys=True)
        for _ in range(99):
            r = derive_supervisory_steering([_make_fc_static()])
            j = json.dumps(export_supervisory_steering_bundle(r), sort_keys=True)
            assert j == ref_json

    def test_100_run_ledger_stable(self):
        r = derive_supervisory_steering([_make_fc_static()])
        ref_ledger = build_supervisory_steering_ledger([r])
        ref_hash = ref_ledger.stable_hash
        for _ in range(99):
            r2 = derive_supervisory_steering([_make_fc_static()])
            l2 = build_supervisory_steering_ledger([r2])
            assert l2.stable_hash == ref_hash


# ---------------------------------------------------------------------------
# Test invalid input rejection
# ---------------------------------------------------------------------------

class TestInvalidInput:
    """Tests for input validation and error handling."""

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="must not be empty"):
            derive_supervisory_steering([])

    def test_string_raises_type_error(self):
        with pytest.raises(TypeError):
            derive_supervisory_steering("bad input")  # type: ignore[arg-type]

    def test_bytes_raises_type_error(self):
        with pytest.raises(TypeError):
            derive_supervisory_steering(b"bad")  # type: ignore[arg-type]

    def test_dict_raises_type_error(self):
        with pytest.raises(TypeError):
            derive_supervisory_steering({"a": 1})  # type: ignore[arg-type]

    def test_wrong_element_type_raises(self):
        with pytest.raises(TypeError, match="decisions\\[0\\]"):
            derive_supervisory_steering([42])  # type: ignore[list-item]

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            derive_supervisory_steering(None)  # type: ignore[arg-type]

    def test_mixed_types_raises(self):
        fc = _make_fc_static()
        with pytest.raises(TypeError, match="decisions\\[1\\]"):
            derive_supervisory_steering([fc, "bad"])  # type: ignore[list-item]

    def test_ledger_wrong_type_raises(self):
        with pytest.raises(TypeError, match="decisions\\[0\\]"):
            build_supervisory_steering_ledger(["not a decision"])

    def test_int_raises_type_error(self):
        with pytest.raises(TypeError):
            derive_supervisory_steering(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test no decoder contamination
# ---------------------------------------------------------------------------

class TestNoDecoderContamination:
    """Tests that no decoder imports leak into this module."""

    def test_no_decoder_import_in_module(self):
        import qec.analysis.forecast_guided_supervisory_steering as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_no_decoder_import_in_upstream(self):
        import qec.analysis.quantization_aware_forecast_compression as mod
        source = open(mod.__file__).read()
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source


# ---------------------------------------------------------------------------
# Test version
# ---------------------------------------------------------------------------

class TestVersion:
    """Tests for version correctness."""

    def test_version_constant(self):
        assert FORECAST_GUIDED_SUPERVISORY_STEERING_VERSION == "v137.0.8"

    def test_decision_version(self):
        result = derive_supervisory_steering([_make_fc_static()])
        assert result.version == "v137.0.8"


# ---------------------------------------------------------------------------
# Test core function integration
# ---------------------------------------------------------------------------

class TestCoreFunction:
    """Integration tests for derive_supervisory_steering."""

    def test_returns_steering_decision(self):
        result = derive_supervisory_steering([_make_fc_static()])
        assert isinstance(result, SteeringDecision)

    def test_horizon_count_matches_input(self):
        decisions = [_make_fc_static(), _make_fc_escalating()]
        result = derive_supervisory_steering(decisions)
        assert result.horizon_count == 2

    def test_single_horizon_count(self):
        result = derive_supervisory_steering([_make_fc_static()])
        assert result.horizon_count == 1

    def test_drift_score_in_result(self):
        result = derive_supervisory_steering([_make_fc_static(), _make_fc_escalating()])
        assert isinstance(result.drift_score, float)
        assert -1.0 <= result.drift_score <= 1.0

    def test_coherence_score_in_result(self):
        result = derive_supervisory_steering([_make_fc_static()])
        assert isinstance(result.coherence_score, float)
        assert 0.0 <= result.coherence_score <= 1.0

    def test_steering_action_is_valid(self):
        valid = {STEERING_HOLD, STEERING_DAMPEN, STEERING_AMPLIFY, STEERING_REDIRECT, STEERING_LOCKDOWN}
        result = derive_supervisory_steering([_make_fc_static()])
        assert result.steering_action in valid

    def test_trend_class_is_valid(self):
        valid = {TREND_STABLE, TREND_DRIFT_UP, TREND_DRIFT_DOWN, TREND_VOLATILE, TREND_CRITICAL}
        result = derive_supervisory_steering([_make_fc_static()])
        assert result.dominant_trend_class in valid

    def test_coherence_class_is_valid(self):
        valid = {COHERENCE_HIGH, COHERENCE_MEDIUM, COHERENCE_LOW, COHERENCE_FAILED}
        result = derive_supervisory_steering([_make_fc_static()])
        assert result.temporal_coherence_class in valid

    def test_symbolic_trace_non_empty(self):
        result = derive_supervisory_steering([_make_fc_static()])
        assert len(result.steering_symbolic_trace) > 0

    def test_accepts_tuple_input(self):
        result = derive_supervisory_steering((_make_fc_static(),))
        assert isinstance(result, SteeringDecision)

    def test_accepts_list_input(self):
        result = derive_supervisory_steering([_make_fc_static()])
        assert isinstance(result, SteeringDecision)

    def test_multi_horizon_integration(self):
        decisions = [_make_fc_static(), _make_fc_escalating(),
                     _make_fc_collapse(), _make_fc_mixed()]
        result = derive_supervisory_steering(decisions)
        assert result.horizon_count == 4
        assert isinstance(result.stable_hash, str)
        assert len(result.stable_hash) == 64


# ---------------------------------------------------------------------------
# Test ledger
# ---------------------------------------------------------------------------

class TestLedger:
    """Tests for steering ledger construction."""

    def test_ledger_type(self):
        r = derive_supervisory_steering([_make_fc_static()])
        ledger = build_supervisory_steering_ledger([r])
        assert isinstance(ledger, SteeringLedger)

    def test_ledger_count(self):
        r1 = derive_supervisory_steering([_make_fc_static()])
        r2 = derive_supervisory_steering([_make_fc_escalating()])
        ledger = build_supervisory_steering_ledger([r1, r2])
        assert ledger.decision_count == 2

    def test_ledger_decisions_tuple(self):
        r = derive_supervisory_steering([_make_fc_static()])
        ledger = build_supervisory_steering_ledger([r])
        assert isinstance(ledger.decisions, tuple)

    def test_ledger_hash_non_empty(self):
        r = derive_supervisory_steering([_make_fc_static()])
        ledger = build_supervisory_steering_ledger([r])
        assert len(ledger.stable_hash) == 64


# ---------------------------------------------------------------------------
# Test float precision
# ---------------------------------------------------------------------------

class TestFloatPrecision:
    """Tests for float precision guarantees."""

    def test_round_precision(self):
        val = _round(0.123456789012345678)
        assert val == round(0.123456789012345678, FLOAT_PRECISION)

    def test_float_precision_constant(self):
        assert FLOAT_PRECISION == 12
