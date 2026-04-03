"""
Tests for Spectral Attractor Forecasting — v136.9.1

Covers:
  1. Bounded forecast score [0,1]
  2. 100-run determinism
  3. Stable ordering
  4. Low-risk forecast case
  5. Critical collapse forecast case
  6. Export equality
  7. Frozen dataclass immutability
  8. Monotonic risk increase scenario
  9. Pre-collapse signature detection
  10. Recovery suggestion routing
"""

from __future__ import annotations

import json

import pytest

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
    ForecastLedger,
    SpectralForecastDecision,
    append_forecast_decision,
    build_forecast_ledger,
    compute_basin_switch_risk,
    compute_forecast_decision,
    detect_precollapse_signature,
    export_forecast_bundle,
    suggest_recovery_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _low_risk_inputs():
    """Inputs that produce a LOW risk forecast."""
    return dict(
        spectral_drift=0.05,
        spectral_energy_delta=0.02,
        centroid_q=0.05,
        centroid_p=0.03,
        drift_momentum=0.01,
        negative_mass=0.02,
        prior_phase_risk_score=0.05,
        prior_escalation_level=0,
    )


def _critical_risk_inputs():
    """Inputs that produce a CRITICAL or COLLAPSE_IMMINENT forecast."""
    return dict(
        spectral_drift=0.95,
        spectral_energy_delta=0.90,
        centroid_q=0.85,
        centroid_p=0.80,
        drift_momentum=0.90,
        negative_mass=0.85,
        prior_phase_risk_score=0.90,
        prior_escalation_level=3,
    )


# ---------------------------------------------------------------------------
# 1. Bounded forecast score [0,1]
# ---------------------------------------------------------------------------

class TestBoundedForecastScore:
    """Verify forecast risk score is always in [0, 1]."""

    def test_all_zero_inputs(self):
        score = compute_basin_switch_risk(
            spectral_drift=0.0, spectral_energy_delta=0.0,
            centroid_q=0.0, centroid_p=0.0,
            drift_momentum=0.0, negative_mass=0.0,
            prior_phase_risk_score=0.0, prior_escalation_level=0,
        )
        assert 0.0 <= score <= 1.0

    def test_all_max_inputs(self):
        score = compute_basin_switch_risk(
            spectral_drift=1.0, spectral_energy_delta=1.0,
            centroid_q=1.0, centroid_p=1.0,
            drift_momentum=1.0, negative_mass=1.0,
            prior_phase_risk_score=1.0, prior_escalation_level=3,
        )
        assert 0.0 <= score <= 1.0

    def test_extreme_inputs(self):
        score = compute_basin_switch_risk(
            spectral_drift=100.0, spectral_energy_delta=100.0,
            centroid_q=100.0, centroid_p=100.0,
            drift_momentum=100.0, negative_mass=100.0,
            prior_phase_risk_score=100.0, prior_escalation_level=100,
        )
        assert 0.0 <= score <= 1.0

    def test_negative_inputs(self):
        score = compute_basin_switch_risk(
            spectral_drift=-5.0, spectral_energy_delta=-3.0,
            centroid_q=-1.0, centroid_p=-1.0,
            drift_momentum=-2.0, negative_mass=-1.0,
            prior_phase_risk_score=-0.5, prior_escalation_level=0,
        )
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 2. 100-run determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Verify byte-identical outputs across 100 repeated runs."""

    def test_100_run_determinism(self):
        inputs = _low_risk_inputs()
        reference = compute_forecast_decision(**inputs)
        for _ in range(100):
            result = compute_forecast_decision(**inputs)
            assert result == reference
            assert result.stable_hash == reference.stable_hash
            assert result.forecast_risk == reference.forecast_risk

    def test_100_run_export_determinism(self):
        inputs = _low_risk_inputs()
        decision = compute_forecast_decision(**inputs)
        reference_bundle = export_forecast_bundle(decision)
        reference_json = json.dumps(reference_bundle, sort_keys=True)
        for _ in range(100):
            bundle = export_forecast_bundle(decision)
            assert json.dumps(bundle, sort_keys=True) == reference_json

    def test_100_run_critical_determinism(self):
        inputs = _critical_risk_inputs()
        reference = compute_forecast_decision(**inputs)
        for _ in range(100):
            result = compute_forecast_decision(**inputs)
            assert result == reference


# ---------------------------------------------------------------------------
# 3. Stable ordering
# ---------------------------------------------------------------------------

class TestStableOrdering:
    """Verify ledger ordering and tuple stability."""

    def test_ledger_ordering_preserved(self):
        d1 = compute_forecast_decision(**_low_risk_inputs())
        d2 = compute_forecast_decision(**_critical_risk_inputs())
        ledger = build_forecast_ledger((d1, d2))
        assert ledger.decisions[0] == d1
        assert ledger.decisions[1] == d2
        assert ledger.decision_count == 2

    def test_append_preserves_order(self):
        d1 = compute_forecast_decision(**_low_risk_inputs())
        d2 = compute_forecast_decision(**_critical_risk_inputs())
        ledger = build_forecast_ledger((d1,))
        ledger = append_forecast_decision(d2, ledger)
        assert ledger.decisions == (d1, d2)

    def test_ledger_hash_stable(self):
        d1 = compute_forecast_decision(**_low_risk_inputs())
        d2 = compute_forecast_decision(**_critical_risk_inputs())
        ledger1 = build_forecast_ledger((d1, d2))
        ledger2 = build_forecast_ledger((d1, d2))
        assert ledger1.stable_hash == ledger2.stable_hash

    def test_ledger_hash_changes_with_order(self):
        d1 = compute_forecast_decision(**_low_risk_inputs())
        d2 = compute_forecast_decision(**_critical_risk_inputs())
        ledger_ab = build_forecast_ledger((d1, d2))
        ledger_ba = build_forecast_ledger((d2, d1))
        assert ledger_ab.stable_hash != ledger_ba.stable_hash


# ---------------------------------------------------------------------------
# 4. Low-risk forecast case
# ---------------------------------------------------------------------------

class TestLowRiskForecast:
    """Verify low-risk inputs produce expected outcomes."""

    def test_low_risk_label(self):
        decision = compute_forecast_decision(**_low_risk_inputs())
        assert decision.risk_label == LABEL_LOW

    def test_low_risk_score(self):
        decision = compute_forecast_decision(**_low_risk_inputs())
        assert decision.forecast_risk <= 0.20

    def test_low_risk_recovery(self):
        decision = compute_forecast_decision(**_low_risk_inputs())
        assert decision.recovery_suggestion == RECOVERY_HOLD_PRIMARY

    def test_low_risk_no_precollapse(self):
        decision = compute_forecast_decision(**_low_risk_inputs())
        assert decision.precollapse_detected is False

    def test_low_risk_collapse_probability(self):
        decision = compute_forecast_decision(**_low_risk_inputs())
        assert decision.collapse_probability < 0.1


# ---------------------------------------------------------------------------
# 5. Critical collapse forecast case
# ---------------------------------------------------------------------------

class TestCriticalCollapseForecast:
    """Verify high-risk inputs produce critical/collapse outcomes."""

    def test_critical_label(self):
        decision = compute_forecast_decision(**_critical_risk_inputs())
        assert decision.risk_label in (LABEL_CRITICAL, LABEL_COLLAPSE_IMMINENT)

    def test_critical_high_score(self):
        decision = compute_forecast_decision(**_critical_risk_inputs())
        assert decision.forecast_risk > 0.70

    def test_critical_precollapse_detected(self):
        decision = compute_forecast_decision(**_critical_risk_inputs())
        assert decision.precollapse_detected is True

    def test_critical_emergency_recovery(self):
        decision = compute_forecast_decision(**_critical_risk_inputs())
        assert decision.recovery_suggestion == RECOVERY_EMERGENCY_REINIT

    def test_critical_collapse_probability(self):
        decision = compute_forecast_decision(**_critical_risk_inputs())
        assert decision.collapse_probability > 0.5


# ---------------------------------------------------------------------------
# 6. Export equality
# ---------------------------------------------------------------------------

class TestExportEquality:
    """Verify export produces byte-identical bundles."""

    def test_export_equality(self):
        decision = compute_forecast_decision(**_low_risk_inputs())
        b1 = export_forecast_bundle(decision)
        b2 = export_forecast_bundle(decision)
        assert b1 == b2

    def test_export_json_equality(self):
        decision = compute_forecast_decision(**_low_risk_inputs())
        b1 = export_forecast_bundle(decision)
        b2 = export_forecast_bundle(decision)
        j1 = json.dumps(b1, sort_keys=True, separators=(",", ":"))
        j2 = json.dumps(b2, sort_keys=True, separators=(",", ":"))
        assert j1 == j2

    def test_export_contains_version(self):
        decision = compute_forecast_decision(**_low_risk_inputs())
        bundle = export_forecast_bundle(decision)
        assert bundle["version"] == "v136.9.1"
        assert bundle["layer"] == "spectral_attractor_forecasting"

    def test_export_contains_all_fields(self):
        decision = compute_forecast_decision(**_low_risk_inputs())
        bundle = export_forecast_bundle(decision)
        expected_keys = {
            "basin_switch_risk", "centroid_p", "centroid_q",
            "collapse_probability", "drift_momentum", "forecast_risk",
            "layer", "negative_mass", "phase_radius",
            "precollapse_detected", "prior_escalation_level",
            "prior_phase_risk_score", "recovery_suggestion",
            "risk_label", "spectral_drift", "spectral_energy_delta",
            "stable_hash", "version",
        }
        assert set(bundle.keys()) == expected_keys


# ---------------------------------------------------------------------------
# 7. Frozen dataclass immutability
# ---------------------------------------------------------------------------

class TestFrozenImmutability:
    """Verify frozen dataclasses reject mutation."""

    def test_decision_immutable(self):
        decision = compute_forecast_decision(**_low_risk_inputs())
        with pytest.raises(AttributeError):
            decision.forecast_risk = 0.99  # type: ignore[misc]

    def test_decision_hash_immutable(self):
        decision = compute_forecast_decision(**_low_risk_inputs())
        with pytest.raises(AttributeError):
            decision.stable_hash = "tampered"  # type: ignore[misc]

    def test_ledger_immutable(self):
        ledger = build_forecast_ledger()
        with pytest.raises(AttributeError):
            ledger.decision_count = 999  # type: ignore[misc]

    def test_ledger_hash_immutable(self):
        ledger = build_forecast_ledger()
        with pytest.raises(AttributeError):
            ledger.stable_hash = "tampered"  # type: ignore[misc]

    def test_decision_is_frozen_dataclass(self):
        decision = compute_forecast_decision(**_low_risk_inputs())
        assert isinstance(decision, SpectralForecastDecision)
        assert decision.__dataclass_params__.frozen is True  # type: ignore[attr-defined]

    def test_ledger_is_frozen_dataclass(self):
        ledger = build_forecast_ledger()
        assert isinstance(ledger, ForecastLedger)
        assert ledger.__dataclass_params__.frozen is True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 8. Monotonic risk increase scenario
# ---------------------------------------------------------------------------

class TestMonotonicRiskIncrease:
    """Verify monotonically increasing inputs produce increasing risk."""

    def test_monotonic_risk_increase(self):
        scores = []
        for i in range(5):
            factor = i * 0.2
            score = compute_basin_switch_risk(
                spectral_drift=factor,
                spectral_energy_delta=factor,
                centroid_q=factor,
                centroid_p=factor,
                drift_momentum=factor,
                negative_mass=factor,
                prior_phase_risk_score=factor,
                prior_escalation_level=i,
            )
            scores.append(score)
        # Verify monotonically non-decreasing
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], (
                f"Risk did not increase: {scores[i]} > {scores[i + 1]}"
            )

    def test_labels_escalate_with_risk(self):
        labels = []
        for i in range(5):
            factor = i * 0.25
            decision = compute_forecast_decision(
                spectral_drift=factor,
                spectral_energy_delta=factor,
                centroid_q=factor,
                centroid_p=factor,
                drift_momentum=factor,
                negative_mass=factor,
                prior_phase_risk_score=factor,
                prior_escalation_level=min(i, 3),
            )
            labels.append(decision.risk_label)
        # First label should be lowest severity
        assert labels[0] == LABEL_LOW
        # Last label should be high severity
        assert labels[-1] in (LABEL_CRITICAL, LABEL_COLLAPSE_IMMINENT)


# ---------------------------------------------------------------------------
# 9. Pre-collapse signature detection
# ---------------------------------------------------------------------------

class TestPrecollapse:
    """Verify pre-collapse signature detection logic."""

    def test_drift_and_negative_mass_trigger(self):
        result = detect_precollapse_signature(
            spectral_drift=0.6,
            negative_mass=0.5,
            centroid_q=0.1,
            centroid_p=0.1,
            prior_escalation_level=0,
        )
        assert result is True

    def test_centroid_displacement_trigger(self):
        result = detect_precollapse_signature(
            spectral_drift=0.1,
            negative_mass=0.1,
            centroid_q=0.5,
            centroid_p=0.5,
            prior_escalation_level=0,
        )
        # sqrt(0.5^2 + 0.5^2) = 0.707 > 0.6
        assert result is True

    def test_escalation_trigger(self):
        result = detect_precollapse_signature(
            spectral_drift=0.1,
            negative_mass=0.1,
            centroid_q=0.1,
            centroid_p=0.1,
            prior_escalation_level=2,
        )
        assert result is True

    def test_monotonic_window_trigger(self):
        # Build 3 decisions with strictly increasing risk
        decisions = []
        for i in range(3):
            factor = 0.1 + i * 0.15
            d = compute_forecast_decision(
                spectral_drift=factor,
                spectral_energy_delta=factor,
                centroid_q=factor * 0.5,
                centroid_p=factor * 0.3,
                drift_momentum=factor,
                negative_mass=factor,
                prior_phase_risk_score=factor,
                prior_escalation_level=0,
            )
            decisions.append(d)
        # Confirm monotonic increase in our decisions
        assert all(
            decisions[i].forecast_risk < decisions[i + 1].forecast_risk
            for i in range(len(decisions) - 1)
        )
        result = detect_precollapse_signature(
            spectral_drift=0.2,
            negative_mass=0.1,
            centroid_q=0.1,
            centroid_p=0.1,
            prior_escalation_level=0,
            prior_decisions=tuple(decisions),
        )
        assert result is True

    def test_no_trigger_low_signals(self):
        result = detect_precollapse_signature(
            spectral_drift=0.1,
            negative_mass=0.1,
            centroid_q=0.1,
            centroid_p=0.1,
            prior_escalation_level=0,
        )
        assert result is False


# ---------------------------------------------------------------------------
# 10. Recovery suggestion routing
# ---------------------------------------------------------------------------

class TestRecoverySuggestion:
    """Verify recovery suggestion deterministic routing."""

    def test_low_holds_primary(self):
        result = suggest_recovery_path(
            risk_label=LABEL_LOW,
            spectral_drift=0.1,
            negative_mass=0.1,
            drift_momentum=0.1,
            precollapse_detected=False,
        )
        assert result == RECOVERY_HOLD_PRIMARY

    def test_watch_holds_primary(self):
        result = suggest_recovery_path(
            risk_label=LABEL_WATCH,
            spectral_drift=0.3,
            negative_mass=0.1,
            drift_momentum=0.1,
            precollapse_detected=False,
        )
        assert result == RECOVERY_HOLD_PRIMARY

    def test_watch_high_drift_shifts_recovery(self):
        result = suggest_recovery_path(
            risk_label=LABEL_WATCH,
            spectral_drift=0.7,
            negative_mass=0.1,
            drift_momentum=0.1,
            precollapse_detected=False,
        )
        assert result == RECOVERY_SHIFT_RECOVERY

    def test_warning_shifts_recovery(self):
        result = suggest_recovery_path(
            risk_label=LABEL_WARNING,
            spectral_drift=0.5,
            negative_mass=0.3,
            drift_momentum=0.3,
            precollapse_detected=False,
        )
        assert result == RECOVERY_SHIFT_RECOVERY

    def test_warning_high_negative_mass_stabilize(self):
        result = suggest_recovery_path(
            risk_label=LABEL_WARNING,
            spectral_drift=0.5,
            negative_mass=0.6,
            drift_momentum=0.3,
            precollapse_detected=False,
        )
        assert result == RECOVERY_LATTICE_STABILIZE

    def test_warning_high_drift_momentum_alternate(self):
        result = suggest_recovery_path(
            risk_label=LABEL_WARNING,
            spectral_drift=0.5,
            negative_mass=0.3,
            drift_momentum=0.7,
            precollapse_detected=False,
        )
        assert result == RECOVERY_SHIFT_ALTERNATE

    def test_critical_shifts_alternate(self):
        result = suggest_recovery_path(
            risk_label=LABEL_CRITICAL,
            spectral_drift=0.8,
            negative_mass=0.5,
            drift_momentum=0.8,
            precollapse_detected=False,
        )
        assert result == RECOVERY_SHIFT_ALTERNATE

    def test_critical_high_negative_mass_stabilize(self):
        result = suggest_recovery_path(
            risk_label=LABEL_CRITICAL,
            spectral_drift=0.8,
            negative_mass=0.8,
            drift_momentum=0.8,
            precollapse_detected=False,
        )
        assert result == RECOVERY_LATTICE_STABILIZE

    def test_critical_with_precollapse_emergency(self):
        result = suggest_recovery_path(
            risk_label=LABEL_CRITICAL,
            spectral_drift=0.8,
            negative_mass=0.5,
            drift_momentum=0.8,
            precollapse_detected=True,
        )
        assert result == RECOVERY_EMERGENCY_REINIT

    def test_collapse_imminent_emergency(self):
        result = suggest_recovery_path(
            risk_label=LABEL_COLLAPSE_IMMINENT,
            spectral_drift=0.95,
            negative_mass=0.9,
            drift_momentum=0.9,
            precollapse_detected=True,
        )
        assert result == RECOVERY_EMERGENCY_REINIT
