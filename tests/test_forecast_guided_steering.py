"""
Tests for Forecast-Guided Adaptive Steering — v136.9.2

Covers:
  1. Bounded adaptive rollback [0,1]
  2. Escalation monotonicity
  3. Forecast override routing
  4. Low-risk preservation
  5. Critical proactive reroute
  6. Emergency dominance
  7. 100-run determinism
  8. Export equality
  9. Frozen dataclass immutability
  10. Ledger hash stability
"""

from __future__ import annotations

import json

import pytest

from qec.analysis.phase_space_decoder_steering import (
    ESCALATION_CRITICAL,
    ESCALATION_WARNING,
    ROUTE_ALTERNATE,
    ROUTE_EMERGENCY,
    ROUTE_PRIMARY,
    ROUTE_RECOVERY,
    route_decoder_from_phase_space,
)
from qec.analysis.spectral_attractor_forecasting import (
    LABEL_COLLAPSE_IMMINENT,
    LABEL_CRITICAL,
    LABEL_LOW,
    LABEL_WARNING,
    LABEL_WATCH,
    RECOVERY_EMERGENCY_REINIT,
    RECOVERY_HOLD_PRIMARY,
    RECOVERY_LATTICE_STABILIZE,
    RECOVERY_SHIFT_ALTERNATE,
    RECOVERY_SHIFT_RECOVERY,
    compute_forecast_decision,
)
from qec.analysis.forecast_guided_steering import (
    ADAPTIVE_STEERING_VERSION,
    AdaptiveSteeringDecision,
    AdaptiveSteeringLedger,
    append_adaptive_steering_decision,
    build_adaptive_steering_ledger,
    compute_adaptive_escalation,
    compute_adaptive_rollback_weight,
    export_adaptive_steering_bundle,
    route_with_forecast_guidance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_low_steering() -> PhaseSteeringDecision:
    """Create a low-risk steering decision."""
    return route_decoder_from_phase_space(
        centroid_q=0.05, centroid_p=0.03,
        negative_mass=0.02, drift_momentum=0.01,
    )


def _make_critical_steering() -> PhaseSteeringDecision:
    """Create a critical-risk steering decision."""
    return route_decoder_from_phase_space(
        centroid_q=0.9, centroid_p=0.85,
        negative_mass=0.9, drift_momentum=0.9,
    )


def _make_low_forecast() -> SpectralForecastDecision:
    """Create a low-risk forecast decision."""
    return compute_forecast_decision(
        spectral_drift=0.05, spectral_energy_delta=0.02,
        centroid_q=0.05, centroid_p=0.03,
        drift_momentum=0.01, negative_mass=0.02,
        prior_phase_risk_score=0.05, prior_escalation_level=0,
    )


def _make_critical_forecast() -> SpectralForecastDecision:
    """Create a critical/collapse forecast decision."""
    return compute_forecast_decision(
        spectral_drift=0.95, spectral_energy_delta=0.90,
        centroid_q=0.85, centroid_p=0.80,
        drift_momentum=0.90, negative_mass=0.85,
        prior_phase_risk_score=0.90, prior_escalation_level=3,
    )


def _make_watch_forecast() -> SpectralForecastDecision:
    """Create a WATCH-level forecast decision."""
    d = compute_forecast_decision(
        spectral_drift=0.35, spectral_energy_delta=0.30,
        centroid_q=0.20, centroid_p=0.15,
        drift_momentum=0.20, negative_mass=0.20,
        prior_phase_risk_score=0.25, prior_escalation_level=0,
    )
    assert d.risk_label == LABEL_WATCH, f"Expected WATCH, got {d.risk_label}"
    return d


def _make_warning_forecast() -> SpectralForecastDecision:
    """Create a WARNING-level forecast decision."""
    return compute_forecast_decision(
        spectral_drift=0.50, spectral_energy_delta=0.45,
        centroid_q=0.35, centroid_p=0.30,
        drift_momentum=0.40, negative_mass=0.35,
        prior_phase_risk_score=0.50, prior_escalation_level=1,
    )


# ---------------------------------------------------------------------------
# 1. Bounded adaptive rollback [0,1]
# ---------------------------------------------------------------------------

class TestBoundedAdaptiveRollback:
    """Verify adaptive rollback weight is always in [0, 1]."""

    def test_zero_inputs(self):
        w = compute_adaptive_rollback_weight(0.0, 0.0)
        assert 0.0 <= w <= 1.0

    def test_max_inputs(self):
        w = compute_adaptive_rollback_weight(1.0, 1.0)
        assert 0.0 <= w <= 1.0

    def test_extreme_inputs(self):
        w = compute_adaptive_rollback_weight(100.0, 100.0)
        assert 0.0 <= w <= 1.0

    def test_negative_inputs(self):
        w = compute_adaptive_rollback_weight(-5.0, -3.0)
        assert 0.0 <= w <= 1.0

    def test_forecast_modulation_increases_rollback(self):
        w_base = compute_adaptive_rollback_weight(0.3, 0.0)
        w_mod = compute_adaptive_rollback_weight(0.3, 0.8)
        assert w_mod >= w_base

    def test_full_decision_rollback_bounded(self):
        steering = _make_low_steering()
        forecast = _make_critical_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        assert 0.0 <= decision.adaptive_rollback_weight <= 1.0


# ---------------------------------------------------------------------------
# 2. Escalation monotonicity
# ---------------------------------------------------------------------------

class TestEscalationMonotonicity:
    """Verify escalation never decreases as forecast severity increases."""

    def test_escalation_floor_from_forecast_label(self):
        labels = [LABEL_LOW, LABEL_WATCH, LABEL_WARNING, LABEL_CRITICAL,
                  LABEL_COLLAPSE_IMMINENT]
        levels = []
        for label in labels:
            level = compute_adaptive_escalation(
                steering_escalation_level=0,
                forecast_label=label,
                precollapse_detected=False,
            )
            levels.append(level)
        for i in range(len(levels) - 1):
            assert levels[i] <= levels[i + 1], (
                f"Escalation decreased: {levels[i]} > {levels[i + 1]} "
                f"at labels {labels[i]}->{labels[i + 1]}"
            )

    def test_precollapse_steps_above_prior(self):
        level = compute_adaptive_escalation(
            steering_escalation_level=1,
            forecast_label=LABEL_LOW,
            precollapse_detected=True,
        )
        assert level >= 2  # at least one step above prior (1)

    def test_escalation_capped_at_critical(self):
        level = compute_adaptive_escalation(
            steering_escalation_level=3,
            forecast_label=LABEL_COLLAPSE_IMMINENT,
            precollapse_detected=True,
        )
        assert level == ESCALATION_CRITICAL

    def test_escalation_preserves_high_steering(self):
        level = compute_adaptive_escalation(
            steering_escalation_level=ESCALATION_WARNING,
            forecast_label=LABEL_LOW,
            precollapse_detected=False,
        )
        assert level >= ESCALATION_WARNING


# ---------------------------------------------------------------------------
# 3. Forecast override routing
# ---------------------------------------------------------------------------

class TestForecastOverrideRouting:
    """Verify forecast overrides steering route when severity is high."""

    def test_watch_nudges_primary_to_recovery(self):
        steering = _make_low_steering()
        forecast = _make_watch_forecast()
        assert forecast.risk_label == LABEL_WATCH
        decision = route_with_forecast_guidance(steering, forecast)
        assert decision.adaptive_recovery_route in (
            ROUTE_RECOVERY, ROUTE_ALTERNATE, ROUTE_EMERGENCY,
        )

    def test_warning_upgrades_route(self):
        steering = _make_low_steering()
        forecast = _make_warning_forecast()
        assert forecast.risk_label == LABEL_WARNING
        decision = route_with_forecast_guidance(steering, forecast)
        route_severity = {
            ROUTE_PRIMARY: 0, ROUTE_RECOVERY: 1,
            ROUTE_ALTERNATE: 2, ROUTE_EMERGENCY: 3,
        }
        assert route_severity[decision.adaptive_recovery_route] >= 1

    def test_critical_forces_alternate_or_higher(self):
        steering = _make_low_steering()
        forecast = _make_critical_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        assert decision.adaptive_recovery_route in (
            ROUTE_ALTERNATE, ROUTE_EMERGENCY,
        )

    def test_shift_recovery_suggestion_affects_route(self):
        steering = _make_low_steering()
        forecast = _make_warning_forecast()
        assert forecast.recovery_suggestion == RECOVERY_SHIFT_RECOVERY
        decision = route_with_forecast_guidance(steering, forecast)
        assert decision.adaptive_recovery_route in (
            ROUTE_RECOVERY, ROUTE_ALTERNATE, ROUTE_EMERGENCY,
        )

    def test_shift_recovery_suggestion_affects_bias(self):
        steering = _make_low_steering()
        forecast = _make_warning_forecast()
        assert forecast.recovery_suggestion == RECOVERY_SHIFT_RECOVERY
        decision = route_with_forecast_guidance(steering, forecast)
        assert "DECODE_PORTFOLIO_B" in decision.adaptive_decoder_bias


# ---------------------------------------------------------------------------
# 4. Low-risk preservation
# ---------------------------------------------------------------------------

class TestLowRiskPreservation:
    """Verify low-risk forecast preserves prior steering decisions."""

    def test_low_forecast_preserves_route(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        assert decision.adaptive_recovery_route == steering.recovery_route

    def test_low_forecast_preserves_decoder_bias(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        assert decision.adaptive_decoder_bias == steering.decoder_bias

    def test_low_forecast_minimal_rollback_increase(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        # Adaptive rollback should be close to original
        # (small forecast_risk^2 * 0.3 added)
        assert decision.adaptive_rollback_weight <= steering.rollback_weight + 0.05

    def test_low_forecast_preserves_escalation(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        assert decision.adaptive_escalation_level == steering.escalation_level


# ---------------------------------------------------------------------------
# 5. Critical proactive reroute
# ---------------------------------------------------------------------------

class TestCriticalProactiveReroute:
    """Verify critical forecast proactively reroutes the system."""

    def test_critical_shifts_bias(self):
        steering = _make_low_steering()
        forecast = _make_critical_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        # Bias should be shifted toward emergency/alternate decoders
        assert decision.adaptive_decoder_bias != steering.decoder_bias

    def test_critical_increases_rollback(self):
        steering = _make_low_steering()
        forecast = _make_critical_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        assert decision.adaptive_rollback_weight > steering.rollback_weight

    def test_critical_escalates(self):
        steering = _make_low_steering()
        forecast = _make_critical_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        assert decision.adaptive_escalation_level > steering.escalation_level

    def test_critical_reroutes(self):
        steering = _make_low_steering()
        forecast = _make_critical_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        assert decision.adaptive_recovery_route != ROUTE_PRIMARY


# ---------------------------------------------------------------------------
# 6. Emergency dominance
# ---------------------------------------------------------------------------

class TestEmergencyDominance:
    """Verify EMERGENCY_REINIT suggestion dominates routing."""

    def test_emergency_reinit_dominates_route(self):
        steering = _make_low_steering()
        forecast = _make_critical_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        if forecast.recovery_suggestion == RECOVERY_EMERGENCY_REINIT:
            assert decision.adaptive_recovery_route == ROUTE_EMERGENCY

    def test_collapse_imminent_forces_emergency(self):
        steering = _make_low_steering()
        forecast = _make_critical_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        if forecast.risk_label == LABEL_COLLAPSE_IMMINENT:
            assert decision.adaptive_recovery_route == ROUTE_EMERGENCY
            assert "REINIT_CODE_LATTICE" in decision.adaptive_decoder_bias

    def test_emergency_reinit_overrides_low_steering(self):
        steering = _make_low_steering()
        forecast = _make_critical_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        # Even with low steering, critical forecast dominates
        assert decision.adaptive_escalation_level >= ESCALATION_CRITICAL


# ---------------------------------------------------------------------------
# 7. 100-run determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Verify byte-identical outputs across 100 repeated runs."""

    def test_100_run_determinism_low(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        reference = route_with_forecast_guidance(steering, forecast)
        for _ in range(100):
            result = route_with_forecast_guidance(steering, forecast)
            assert result == reference
            assert result.stable_hash == reference.stable_hash

    def test_100_run_determinism_critical(self):
        steering = _make_critical_steering()
        forecast = _make_critical_forecast()
        reference = route_with_forecast_guidance(steering, forecast)
        for _ in range(100):
            result = route_with_forecast_guidance(steering, forecast)
            assert result == reference
            assert result.stable_hash == reference.stable_hash

    def test_100_run_export_determinism(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        ref_bundle = export_adaptive_steering_bundle(decision)
        ref_json = json.dumps(ref_bundle, sort_keys=True)
        for _ in range(100):
            bundle = export_adaptive_steering_bundle(decision)
            assert json.dumps(bundle, sort_keys=True) == ref_json


# ---------------------------------------------------------------------------
# 8. Export equality
# ---------------------------------------------------------------------------

class TestExportEquality:
    """Verify export produces byte-identical bundles."""

    def test_export_equality(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        b1 = export_adaptive_steering_bundle(decision)
        b2 = export_adaptive_steering_bundle(decision)
        assert b1 == b2

    def test_export_json_equality(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        b1 = export_adaptive_steering_bundle(decision)
        b2 = export_adaptive_steering_bundle(decision)
        j1 = json.dumps(b1, sort_keys=True, separators=(",", ":"))
        j2 = json.dumps(b2, sort_keys=True, separators=(",", ":"))
        assert j1 == j2

    def test_export_contains_version(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        bundle = export_adaptive_steering_bundle(decision)
        assert bundle["version"] == ADAPTIVE_STEERING_VERSION
        assert bundle["layer"] == "forecast_guided_steering"

    def test_export_contains_all_fields(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        bundle = export_adaptive_steering_bundle(decision)
        expected_keys = {
            "adaptive_decoder_bias", "adaptive_escalation_level",
            "adaptive_recovery_route", "adaptive_rollback_weight",
            "forecast_label", "forecast_risk_score", "layer",
            "precollapse_detected", "prior_decoder_bias",
            "prior_escalation_level", "prior_phase_risk_score",
            "prior_recovery_route", "prior_rollback_weight",
            "recovery_suggestion", "stable_hash", "version",
        }
        assert set(bundle.keys()) == expected_keys


# ---------------------------------------------------------------------------
# 9. Frozen dataclass immutability
# ---------------------------------------------------------------------------

class TestFrozenImmutability:
    """Verify frozen dataclasses reject mutation."""

    def test_decision_immutable(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        with pytest.raises(AttributeError):
            decision.adaptive_rollback_weight = 0.99  # type: ignore[misc]

    def test_decision_hash_immutable(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        with pytest.raises(AttributeError):
            decision.stable_hash = "tampered"  # type: ignore[misc]

    def test_ledger_immutable(self):
        ledger = build_adaptive_steering_ledger()
        with pytest.raises(AttributeError):
            ledger.decision_count = 999  # type: ignore[misc]

    def test_ledger_hash_immutable(self):
        ledger = build_adaptive_steering_ledger()
        with pytest.raises(AttributeError):
            ledger.stable_hash = "tampered"  # type: ignore[misc]

    def test_decision_is_frozen_dataclass(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        decision = route_with_forecast_guidance(steering, forecast)
        assert isinstance(decision, AdaptiveSteeringDecision)
        assert decision.__dataclass_params__.frozen is True  # type: ignore[attr-defined]

    def test_ledger_is_frozen_dataclass(self):
        ledger = build_adaptive_steering_ledger()
        assert isinstance(ledger, AdaptiveSteeringLedger)
        assert ledger.__dataclass_params__.frozen is True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 10. Ledger hash stability
# ---------------------------------------------------------------------------

class TestLedgerHashStability:
    """Verify ledger hashing is stable and ordering-sensitive."""

    def test_ledger_hash_stable(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        d1 = route_with_forecast_guidance(steering, forecast)
        d2 = route_with_forecast_guidance(
            _make_critical_steering(), _make_critical_forecast(),
        )
        ledger1 = build_adaptive_steering_ledger((d1, d2))
        ledger2 = build_adaptive_steering_ledger((d1, d2))
        assert ledger1.stable_hash == ledger2.stable_hash

    def test_ledger_hash_changes_with_order(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        d1 = route_with_forecast_guidance(steering, forecast)
        d2 = route_with_forecast_guidance(
            _make_critical_steering(), _make_critical_forecast(),
        )
        ledger_ab = build_adaptive_steering_ledger((d1, d2))
        ledger_ba = build_adaptive_steering_ledger((d2, d1))
        assert ledger_ab.stable_hash != ledger_ba.stable_hash

    def test_append_preserves_order(self):
        d1 = route_with_forecast_guidance(
            _make_low_steering(), _make_low_forecast(),
        )
        d2 = route_with_forecast_guidance(
            _make_critical_steering(), _make_critical_forecast(),
        )
        ledger = build_adaptive_steering_ledger((d1,))
        ledger = append_adaptive_steering_decision(d2, ledger)
        assert ledger.decisions == (d1, d2)
        assert ledger.decision_count == 2

    def test_100_run_ledger_hash_determinism(self):
        d1 = route_with_forecast_guidance(
            _make_low_steering(), _make_low_forecast(),
        )
        d2 = route_with_forecast_guidance(
            _make_critical_steering(), _make_critical_forecast(),
        )
        ref_ledger = build_adaptive_steering_ledger((d1, d2))
        for _ in range(100):
            ledger = build_adaptive_steering_ledger((d1, d2))
            assert ledger.stable_hash == ref_ledger.stable_hash
