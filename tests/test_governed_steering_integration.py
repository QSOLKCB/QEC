"""
Tests for Governed Steering Integration — v136.9.4

Covers:
  1. Disabled governance passthrough
  2. Enabled governance mutation
  3. route_mutated correctness
  4. Raw decision preservation
  5. Policy state passthrough
  6. 100-run determinism
  7. Export equality
  8. Frozen immutability
  9. Ledger hash stability
  10. No regression against v136.9.2 / v136.9.3
"""

from __future__ import annotations

import json

import pytest

from qec.analysis.phase_space_decoder_steering import (
    ROUTE_ALTERNATE,
    ROUTE_EMERGENCY,
    ROUTE_PRIMARY,
    ROUTE_RECOVERY,
    route_decoder_from_phase_space,
)
from qec.analysis.spectral_attractor_forecasting import (
    compute_forecast_decision,
)
from qec.analysis.forecast_guided_steering import (
    AdaptiveSteeringDecision,
    route_with_forecast_guidance,
)
from qec.analysis.adaptive_steering_policy_memory import (
    PolicyMemoryState,
    update_policy_memory,
    make_initial_policy_memory_state,
)
from qec.analysis.governed_steering_integration import (
    GOVERNED_STEERING_VERSION,
    GovernedSteeringBundle,
    GovernedSteeringLedger,
    append_governed_bundle,
    build_governed_ledger,
    compute_governed_steering,
    export_governed_steering_bundle,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_low_steering():
    return route_decoder_from_phase_space(
        centroid_q=0.05, centroid_p=0.03,
        negative_mass=0.02, drift_momentum=0.01,
    )


def _make_critical_steering():
    return route_decoder_from_phase_space(
        centroid_q=0.9, centroid_p=0.85,
        negative_mass=0.9, drift_momentum=0.9,
    )


def _make_low_forecast():
    return compute_forecast_decision(
        spectral_drift=0.05, spectral_energy_delta=0.02,
        centroid_q=0.05, centroid_p=0.03,
        drift_momentum=0.01, negative_mass=0.02,
        prior_phase_risk_score=0.05, prior_escalation_level=0,
    )


def _make_critical_forecast():
    return compute_forecast_decision(
        spectral_drift=0.95, spectral_energy_delta=0.90,
        centroid_q=0.85, centroid_p=0.80,
        drift_momentum=0.90, negative_mass=0.85,
        prior_phase_risk_score=0.90, prior_escalation_level=3,
    )


def _make_watch_forecast():
    return compute_forecast_decision(
        spectral_drift=0.35, spectral_energy_delta=0.30,
        centroid_q=0.20, centroid_p=0.15,
        drift_momentum=0.20, negative_mass=0.20,
        prior_phase_risk_score=0.25, prior_escalation_level=0,
    )


def _make_warning_forecast():
    return compute_forecast_decision(
        spectral_drift=0.50, spectral_energy_delta=0.45,
        centroid_q=0.35, centroid_p=0.30,
        drift_momentum=0.40, negative_mass=0.35,
        prior_phase_risk_score=0.50, prior_escalation_level=1,
    )


def _memory_in_recovery(consecutive_low=0, consecutive_recovery=0,
                         oscillation_count=0, cycle_index=1):
    """Helper: memory state in RECOVERY route."""
    return PolicyMemoryState(
        current_route=ROUTE_RECOVERY,
        prior_route=ROUTE_PRIMARY,
        consecutive_low_risk_cycles=consecutive_low,
        consecutive_recovery_cycles=consecutive_recovery,
        oscillation_count=oscillation_count,
        last_escalation_level=1,
        policy_cycle_index=cycle_index,
    )


# ---------------------------------------------------------------------------
# 1. Disabled governance passthrough
# ---------------------------------------------------------------------------

class TestDisabledGovernancePassthrough:
    """When enable_policy_memory=False, governed route == raw route."""

    def test_governed_route_equals_raw_route(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=False,
        )
        assert bundle.governed_route == bundle.raw_decision.adaptive_recovery_route

    def test_governance_applied_false(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=False,
        )
        assert bundle.governance_applied is False

    def test_policy_state_is_none(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=False,
        )
        assert bundle.policy_state is None

    def test_route_mutated_false(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=False,
        )
        assert bundle.route_mutated is False

    def test_cycle_index_zero_without_prior(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=False,
        )
        assert bundle.cycle_index == 0

    def test_cycle_index_increments_with_prior(self):
        prior = _memory_in_recovery(cycle_index=5)
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            prior_policy_state=prior,
            enable_policy_memory=False,
        )
        assert bundle.cycle_index == 6

    def test_critical_passthrough(self):
        bundle = compute_governed_steering(
            _make_critical_steering(), _make_critical_forecast(),
            enable_policy_memory=False,
        )
        assert bundle.governed_route == bundle.raw_decision.adaptive_recovery_route
        assert bundle.route_mutated is False


# ---------------------------------------------------------------------------
# 2. Enabled governance mutation
# ---------------------------------------------------------------------------

class TestEnabledGovernanceMutation:
    """When enable_policy_memory=True, policy memory governs the route."""

    def test_governance_applied_true(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=True,
        )
        assert bundle.governance_applied is True

    def test_policy_state_populated(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=True,
        )
        assert bundle.policy_state is not None
        assert isinstance(bundle.policy_state, PolicyMemoryState)

    def test_hysteresis_blocks_downgrade(self):
        """From RECOVERY, low-risk should be blocked by hysteresis on first cycle."""
        prior = _memory_in_recovery(consecutive_low=0, cycle_index=1)
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            prior_policy_state=prior,
            enable_policy_memory=True,
        )
        # Raw decision wants PRIMARY, but hysteresis blocks -> RECOVERY
        assert bundle.governed_route == ROUTE_RECOVERY
        assert bundle.route_mutated is True

    def test_escalation_always_allowed(self):
        """Escalation to higher severity is always allowed."""
        bundle = compute_governed_steering(
            _make_critical_steering(), _make_critical_forecast(),
            enable_policy_memory=True,
        )
        # Critical forecast should produce EMERGENCY route
        assert bundle.governed_route == bundle.raw_decision.adaptive_recovery_route


# ---------------------------------------------------------------------------
# 3. route_mutated correctness
# ---------------------------------------------------------------------------

class TestRouteMutatedCorrectness:
    """route_mutated must exactly reflect governance mutation."""

    def test_no_mutation_when_routes_match(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=True,
        )
        expected = (bundle.governed_route != bundle.raw_decision.adaptive_recovery_route)
        assert bundle.route_mutated == expected

    def test_mutation_when_hysteresis_blocks(self):
        prior = _memory_in_recovery(consecutive_low=0, cycle_index=1)
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            prior_policy_state=prior,
            enable_policy_memory=True,
        )
        assert bundle.route_mutated is True

    def test_no_mutation_on_critical_escalation(self):
        """Critical escalation shouldn't be mutated by governance."""
        bundle = compute_governed_steering(
            _make_critical_steering(), _make_critical_forecast(),
            enable_policy_memory=True,
        )
        assert bundle.route_mutated is False

    def test_mutation_flag_matches_route_comparison(self):
        """route_mutated == (governed_route != raw_decision.adaptive_recovery_route)"""
        for steering_fn, forecast_fn in [
            (_make_low_steering, _make_low_forecast),
            (_make_critical_steering, _make_critical_forecast),
            (_make_low_steering, _make_watch_forecast),
            (_make_low_steering, _make_warning_forecast),
        ]:
            bundle = compute_governed_steering(
                steering_fn(), forecast_fn(),
                enable_policy_memory=True,
            )
            expected = (bundle.governed_route != bundle.raw_decision.adaptive_recovery_route)
            assert bundle.route_mutated == expected


# ---------------------------------------------------------------------------
# 4. Raw decision preservation
# ---------------------------------------------------------------------------

class TestRawDecisionPreservation:
    """Raw decision must be identical to standalone v136.9.2 output."""

    def test_raw_matches_standalone_low(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        standalone = route_with_forecast_guidance(steering, forecast)
        bundle = compute_governed_steering(
            steering, forecast, enable_policy_memory=True,
        )
        assert bundle.raw_decision == standalone

    def test_raw_matches_standalone_critical(self):
        steering = _make_critical_steering()
        forecast = _make_critical_forecast()
        standalone = route_with_forecast_guidance(steering, forecast)
        bundle = compute_governed_steering(
            steering, forecast, enable_policy_memory=True,
        )
        assert bundle.raw_decision == standalone

    def test_raw_matches_standalone_disabled(self):
        steering = _make_low_steering()
        forecast = _make_watch_forecast()
        standalone = route_with_forecast_guidance(steering, forecast)
        bundle = compute_governed_steering(
            steering, forecast, enable_policy_memory=False,
        )
        assert bundle.raw_decision == standalone


# ---------------------------------------------------------------------------
# 5. Policy state passthrough
# ---------------------------------------------------------------------------

class TestPolicyStatePassthrough:
    """Policy state must match standalone v136.9.3 output."""

    def test_state_matches_standalone(self):
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        raw_decision = route_with_forecast_guidance(steering, forecast)
        standalone_state, standalone_route = update_policy_memory(raw_decision)

        bundle = compute_governed_steering(
            steering, forecast, enable_policy_memory=True,
        )
        assert bundle.policy_state == standalone_state
        assert bundle.governed_route == standalone_route

    def test_state_matches_with_prior(self):
        prior = _memory_in_recovery(consecutive_low=1, cycle_index=3)
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        raw_decision = route_with_forecast_guidance(steering, forecast)
        standalone_state, standalone_route = update_policy_memory(
            raw_decision, prior,
        )

        bundle = compute_governed_steering(
            steering, forecast,
            prior_policy_state=prior,
            enable_policy_memory=True,
        )
        assert bundle.policy_state == standalone_state
        assert bundle.governed_route == standalone_route

    def test_cycle_index_from_policy_state(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=True,
        )
        assert bundle.cycle_index == bundle.policy_state.policy_cycle_index


# ---------------------------------------------------------------------------
# 6. 100-run determinism
# ---------------------------------------------------------------------------

class TestDeterminism100Run:
    """100 identical runs must produce byte-identical output."""

    def test_100_run_determinism_enabled(self):
        ref = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=True,
        )
        for _ in range(99):
            result = compute_governed_steering(
                _make_low_steering(), _make_low_forecast(),
                enable_policy_memory=True,
            )
            assert result == ref
            assert result.stable_hash == ref.stable_hash

    def test_100_run_determinism_disabled(self):
        ref = compute_governed_steering(
            _make_critical_steering(), _make_critical_forecast(),
            enable_policy_memory=False,
        )
        for _ in range(99):
            result = compute_governed_steering(
                _make_critical_steering(), _make_critical_forecast(),
                enable_policy_memory=False,
            )
            assert result == ref
            assert result.stable_hash == ref.stable_hash

    def test_100_run_determinism_with_prior(self):
        prior = _memory_in_recovery(consecutive_low=2, cycle_index=10)
        ref = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            prior_policy_state=prior,
            enable_policy_memory=True,
        )
        for _ in range(99):
            result = compute_governed_steering(
                _make_low_steering(), _make_low_forecast(),
                prior_policy_state=prior,
                enable_policy_memory=True,
            )
            assert result == ref


# ---------------------------------------------------------------------------
# 7. Export equality
# ---------------------------------------------------------------------------

class TestExportEquality:
    """Exported dicts must be identical across runs."""

    def test_export_determinism(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=True,
        )
        export1 = export_governed_steering_bundle(bundle)
        export2 = export_governed_steering_bundle(bundle)
        assert export1 == export2
        assert json.dumps(export1, sort_keys=True) == json.dumps(export2, sort_keys=True)

    def test_export_contains_required_fields(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=True,
        )
        export = export_governed_steering_bundle(bundle)
        required = {
            "adaptive_recovery_route", "governed_route", "route_mutated",
            "governance_applied", "cycle_index", "stable_hash",
            "version", "layer", "forecast_label", "forecast_risk_score",
            "adaptive_escalation_level", "adaptive_rollback_weight",
            "adaptive_decoder_bias", "precollapse_detected",
            "policy_consecutive_low_risk_cycles",
            "policy_consecutive_recovery_cycles",
            "policy_oscillation_count",
        }
        assert required.issubset(set(export.keys()))

    def test_export_version(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=True,
        )
        export = export_governed_steering_bundle(bundle)
        assert export["version"] == GOVERNED_STEERING_VERSION

    def test_export_disabled_has_zero_counters(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=False,
        )
        export = export_governed_steering_bundle(bundle)
        assert export["policy_consecutive_low_risk_cycles"] == 0
        assert export["policy_consecutive_recovery_cycles"] == 0
        assert export["policy_oscillation_count"] == 0

    def test_export_json_serializable(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=True,
        )
        export = export_governed_steering_bundle(bundle)
        serialized = json.dumps(export, sort_keys=True, separators=(",", ":"))
        assert isinstance(serialized, str)
        roundtrip = json.loads(serialized)
        assert roundtrip == export


# ---------------------------------------------------------------------------
# 8. Frozen immutability
# ---------------------------------------------------------------------------

class TestFrozenImmutability:
    """Frozen dataclasses must reject mutation."""

    def test_bundle_frozen(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=True,
        )
        with pytest.raises(AttributeError):
            bundle.governed_route = ROUTE_EMERGENCY  # type: ignore[misc]

    def test_bundle_route_mutated_frozen(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=True,
        )
        with pytest.raises(AttributeError):
            bundle.route_mutated = True  # type: ignore[misc]

    def test_ledger_frozen(self):
        ledger = build_governed_ledger()
        with pytest.raises(AttributeError):
            ledger.bundle_count = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 9. Ledger hash stability
# ---------------------------------------------------------------------------

class TestLedgerHashStability:
    """Ledger hashes must be stable across runs."""

    def test_empty_ledger_hash_stable(self):
        l1 = build_governed_ledger()
        l2 = build_governed_ledger()
        assert l1.stable_hash == l2.stable_hash
        assert l1.bundle_count == 0

    def test_single_bundle_ledger_hash(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=True,
        )
        l1 = build_governed_ledger((bundle,))
        l2 = build_governed_ledger((bundle,))
        assert l1.stable_hash == l2.stable_hash
        assert l1.bundle_count == 1

    def test_append_changes_hash(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=True,
        )
        l1 = build_governed_ledger()
        l2 = append_governed_bundle(bundle, l1)
        assert l1.stable_hash != l2.stable_hash
        assert l2.bundle_count == 1

    def test_100_run_ledger_determinism(self):
        bundle = compute_governed_steering(
            _make_low_steering(), _make_low_forecast(),
            enable_policy_memory=True,
        )
        ref = build_governed_ledger((bundle,))
        for _ in range(99):
            result = build_governed_ledger((bundle,))
            assert result.stable_hash == ref.stable_hash


# ---------------------------------------------------------------------------
# 10. No regression against v136.9.2 / v136.9.3
# ---------------------------------------------------------------------------

class TestNoRegression:
    """Integration must not alter upstream v136.9.2 or v136.9.3 behavior."""

    def test_v136_9_2_raw_decision_unchanged(self):
        """Raw decision from integration matches standalone v136.9.2."""
        steering = _make_low_steering()
        forecast = _make_watch_forecast()
        standalone = route_with_forecast_guidance(steering, forecast)
        bundle = compute_governed_steering(
            steering, forecast, enable_policy_memory=True,
        )
        assert bundle.raw_decision.adaptive_rollback_weight == standalone.adaptive_rollback_weight
        assert bundle.raw_decision.adaptive_escalation_level == standalone.adaptive_escalation_level
        assert bundle.raw_decision.adaptive_recovery_route == standalone.adaptive_recovery_route
        assert bundle.raw_decision.adaptive_decoder_bias == standalone.adaptive_decoder_bias
        assert bundle.raw_decision.stable_hash == standalone.stable_hash

    def test_v136_9_3_policy_state_unchanged(self):
        """Policy state from integration matches standalone v136.9.3."""
        steering = _make_low_steering()
        forecast = _make_low_forecast()
        raw = route_with_forecast_guidance(steering, forecast)
        prior = _memory_in_recovery(consecutive_low=2, cycle_index=5)

        standalone_state, standalone_route = update_policy_memory(raw, prior)
        bundle = compute_governed_steering(
            steering, forecast,
            prior_policy_state=prior,
            enable_policy_memory=True,
        )
        assert bundle.policy_state == standalone_state
        assert bundle.governed_route == standalone_route

    def test_critical_escalation_regression(self):
        """Critical scenario still routes to EMERGENCY unchanged."""
        steering = _make_critical_steering()
        forecast = _make_critical_forecast()
        standalone = route_with_forecast_guidance(steering, forecast)
        bundle = compute_governed_steering(
            steering, forecast, enable_policy_memory=True,
        )
        assert bundle.raw_decision.adaptive_recovery_route == standalone.adaptive_recovery_route
        assert bundle.raw_decision.adaptive_recovery_route == ROUTE_EMERGENCY

    def test_warning_steering_regression(self):
        """Warning scenario preserves v136.9.2 adaptive recovery route."""
        steering = _make_low_steering()
        forecast = _make_warning_forecast()
        standalone = route_with_forecast_guidance(steering, forecast)
        bundle = compute_governed_steering(
            steering, forecast, enable_policy_memory=False,
        )
        assert bundle.raw_decision == standalone
        assert bundle.governed_route == standalone.adaptive_recovery_route
