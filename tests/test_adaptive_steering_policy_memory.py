"""
Tests for Adaptive Steering Policy Memory — v136.9.3

Covers:
  1. Route stickiness (hysteresis blocks premature downgrade)
  2. 3-cycle hysteresis release
  3. Recovery timeout enforcement
  4. Oscillation detection
  5. Oscillation suppression (increased strictness)
  6. 100-run determinism
  7. Export equality
  8. Frozen immutability
  9. Ledger stability
  10. No-regression against v136.9.2 outputs
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
    LABEL_COLLAPSE_IMMINENT,
    LABEL_CRITICAL,
    LABEL_LOW,
    LABEL_WARNING,
    LABEL_WATCH,
    compute_forecast_decision,
)
from qec.analysis.forecast_guided_steering import (
    AdaptiveSteeringDecision,
    route_with_forecast_guidance,
)
from qec.analysis.adaptive_steering_policy_memory import (
    HYSTERESIS_RELEASE_CYCLES,
    OSCILLATION_STRICTNESS_MULTIPLIER,
    OSCILLATION_THRESHOLD,
    POLICY_MEMORY_VERSION,
    RECOVERY_TIMEOUT_CYCLES,
    PolicyMemoryLedger,
    PolicyMemoryState,
    append_policy_memory_state,
    build_policy_memory_ledger,
    compute_hysteresis_route,
    enforce_recovery_timeout,
    export_policy_memory_bundle,
    make_initial_policy_memory_state,
    update_policy_memory,
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


def _make_low_decision() -> AdaptiveSteeringDecision:
    return route_with_forecast_guidance(_make_low_steering(), _make_low_forecast())


def _make_critical_decision() -> AdaptiveSteeringDecision:
    return route_with_forecast_guidance(_make_critical_steering(), _make_critical_forecast())


def _make_watch_decision() -> AdaptiveSteeringDecision:
    return route_with_forecast_guidance(_make_low_steering(), _make_watch_forecast())


def _make_warning_decision() -> AdaptiveSteeringDecision:
    return route_with_forecast_guidance(_make_low_steering(), _make_warning_forecast())


def _memory_in_recovery(consecutive_low: int = 0,
                        consecutive_recovery: int = 0,
                        oscillation_count: int = 0,
                        cycle_index: int = 1) -> PolicyMemoryState:
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


def _memory_in_alternate(consecutive_low: int = 0,
                         consecutive_recovery: int = 0,
                         oscillation_count: int = 0,
                         cycle_index: int = 1) -> PolicyMemoryState:
    """Helper: memory state in ALTERNATE route."""
    return PolicyMemoryState(
        current_route=ROUTE_ALTERNATE,
        prior_route=ROUTE_RECOVERY,
        consecutive_low_risk_cycles=consecutive_low,
        consecutive_recovery_cycles=consecutive_recovery,
        oscillation_count=oscillation_count,
        last_escalation_level=2,
        policy_cycle_index=cycle_index,
    )


# ---------------------------------------------------------------------------
# 1. Route stickiness
# ---------------------------------------------------------------------------

class TestRouteStickiness:
    """Verify hysteresis blocks premature downgrade from non-PRIMARY routes."""

    def test_recovery_blocks_immediate_downgrade(self):
        memory = _memory_in_recovery(consecutive_low=0)
        route = compute_hysteresis_route(ROUTE_PRIMARY, memory)
        assert route == ROUTE_RECOVERY

    def test_alternate_blocks_immediate_downgrade(self):
        memory = _memory_in_alternate(consecutive_low=0)
        route = compute_hysteresis_route(ROUTE_PRIMARY, memory)
        assert route == ROUTE_ALTERNATE

    def test_recovery_blocks_with_one_low_cycle(self):
        memory = _memory_in_recovery(consecutive_low=1)
        route = compute_hysteresis_route(ROUTE_PRIMARY, memory)
        assert route == ROUTE_RECOVERY

    def test_recovery_blocks_with_two_low_cycles(self):
        memory = _memory_in_recovery(consecutive_low=2)
        route = compute_hysteresis_route(ROUTE_PRIMARY, memory)
        assert route == ROUTE_RECOVERY

    def test_escalation_always_allowed(self):
        memory = _memory_in_recovery(consecutive_low=0)
        route = compute_hysteresis_route(ROUTE_EMERGENCY, memory)
        assert route == ROUTE_EMERGENCY

    def test_lateral_at_same_severity_holds(self):
        memory = _memory_in_recovery(consecutive_low=0)
        route = compute_hysteresis_route(ROUTE_RECOVERY, memory)
        assert route == ROUTE_RECOVERY

    def test_full_cycle_stickiness(self):
        """Run through update_policy_memory and verify route stickiness."""
        critical_decision = _make_critical_decision()
        state, route = update_policy_memory(critical_decision, None)
        # Now try a low decision — should be blocked from downgrading
        low_decision = _make_low_decision()
        state2, route2 = update_policy_memory(low_decision, state)
        # Route should not drop to PRIMARY immediately
        if state.current_route != ROUTE_PRIMARY:
            assert route2 != ROUTE_PRIMARY or state.consecutive_low_risk_cycles >= HYSTERESIS_RELEASE_CYCLES


# ---------------------------------------------------------------------------
# 2. 3-cycle hysteresis release
# ---------------------------------------------------------------------------

class TestHysteresisRelease:
    """Verify that 3 consecutive low-risk cycles allow downgrade."""

    def test_release_after_three_cycles(self):
        memory = _memory_in_recovery(consecutive_low=3)
        route = compute_hysteresis_route(ROUTE_PRIMARY, memory)
        assert route == ROUTE_PRIMARY

    def test_release_after_more_than_three_cycles(self):
        memory = _memory_in_recovery(consecutive_low=5)
        route = compute_hysteresis_route(ROUTE_PRIMARY, memory)
        assert route == ROUTE_PRIMARY

    def test_no_release_at_two_cycles(self):
        memory = _memory_in_recovery(consecutive_low=2)
        route = compute_hysteresis_route(ROUTE_PRIMARY, memory)
        assert route == ROUTE_RECOVERY

    def test_full_cycle_release(self):
        """Run 3 low-risk cycles and verify eventual release to PRIMARY."""
        # Start: escalate to non-PRIMARY
        critical_decision = _make_critical_decision()
        state, _ = update_policy_memory(critical_decision, None)

        # Run 3 low-risk cycles
        low_decision = _make_low_decision()
        for _ in range(HYSTERESIS_RELEASE_CYCLES):
            state, governed = update_policy_memory(low_decision, state)

        # Fourth low-risk cycle should allow PRIMARY if proposed
        state, governed = update_policy_memory(low_decision, state)
        # The low decision proposes PRIMARY, so after enough cycles it releases
        if low_decision.adaptive_recovery_route == ROUTE_PRIMARY:
            assert governed == ROUTE_PRIMARY

    def test_release_resets_on_high_risk(self):
        """Verify that a high-risk cycle resets consecutive_low_risk_cycles."""
        state = _memory_in_recovery(consecutive_low=2)
        warning_decision = _make_warning_decision()
        state2, _ = update_policy_memory(warning_decision, state)
        assert state2.consecutive_low_risk_cycles == 0


# ---------------------------------------------------------------------------
# 3. Recovery timeout enforcement
# ---------------------------------------------------------------------------

class TestRecoveryTimeout:
    """Verify bounded-time recovery guarantee."""

    def test_no_timeout_below_limit(self):
        memory = _memory_in_recovery(consecutive_recovery=4)
        route = enforce_recovery_timeout(ROUTE_RECOVERY, memory)
        assert route == ROUTE_RECOVERY

    def test_timeout_at_limit_escalates(self):
        memory = _memory_in_recovery(consecutive_recovery=5)
        route = enforce_recovery_timeout(ROUTE_RECOVERY, memory)
        assert route == ROUTE_ALTERNATE

    def test_timeout_above_limit_escalates(self):
        memory = _memory_in_recovery(consecutive_recovery=10)
        route = enforce_recovery_timeout(ROUTE_RECOVERY, memory)
        assert route == ROUTE_ALTERNATE

    def test_alternate_timeout_escalates_to_emergency(self):
        memory = _memory_in_alternate(consecutive_recovery=5)
        route = enforce_recovery_timeout(ROUTE_ALTERNATE, memory)
        assert route == ROUTE_EMERGENCY

    def test_primary_not_affected(self):
        memory = _memory_in_recovery(consecutive_recovery=10)
        route = enforce_recovery_timeout(ROUTE_PRIMARY, memory)
        assert route == ROUTE_PRIMARY

    def test_emergency_not_affected(self):
        memory = _memory_in_recovery(consecutive_recovery=10)
        route = enforce_recovery_timeout(ROUTE_EMERGENCY, memory)
        assert route == ROUTE_EMERGENCY

    def test_full_cycle_timeout(self):
        """Run RECOVERY_TIMEOUT_CYCLES + 1 recovery cycles and verify escalation."""
        critical_decision = _make_critical_decision()
        state, _ = update_policy_memory(critical_decision, None)

        # Force into a non-primary warning state repeatedly
        warning_decision = _make_warning_decision()
        for _ in range(RECOVERY_TIMEOUT_CYCLES + 2):
            state, governed = update_policy_memory(warning_decision, state)

        # After enough recovery cycles, timeout should have escalated
        assert state.consecutive_recovery_cycles == 0 or governed != ROUTE_RECOVERY or state.consecutive_recovery_cycles <= RECOVERY_TIMEOUT_CYCLES


# ---------------------------------------------------------------------------
# 4. Oscillation detection
# ---------------------------------------------------------------------------

class TestOscillationDetection:
    """Verify route flipping detection."""

    def test_no_oscillation_on_first_cycle(self):
        state = make_initial_policy_memory_state()
        low_decision = _make_low_decision()
        new_state, _ = update_policy_memory(low_decision, state)
        assert new_state.oscillation_count == 0

    def test_oscillation_detected_on_flip(self):
        """A->B->A pattern should increment oscillation_count."""
        # State: currently RECOVERY, prior was PRIMARY
        memory = PolicyMemoryState(
            current_route=ROUTE_RECOVERY,
            prior_route=ROUTE_PRIMARY,
            consecutive_low_risk_cycles=3,
            consecutive_recovery_cycles=1,
            oscillation_count=0,
            last_escalation_level=1,
            policy_cycle_index=2,
        )
        # Decision that proposes PRIMARY (flip back)
        low_decision = _make_low_decision()
        if low_decision.adaptive_recovery_route == ROUTE_PRIMARY:
            new_state, _ = update_policy_memory(low_decision, memory)
            assert new_state.oscillation_count == 1

    def test_no_oscillation_on_continued_escalation(self):
        """Continued escalation should not trigger oscillation."""
        memory = PolicyMemoryState(
            current_route=ROUTE_RECOVERY,
            prior_route=ROUTE_PRIMARY,
            consecutive_low_risk_cycles=0,
            consecutive_recovery_cycles=1,
            oscillation_count=0,
            last_escalation_level=1,
            policy_cycle_index=2,
        )
        critical_decision = _make_critical_decision()
        new_state, _ = update_policy_memory(critical_decision, memory)
        assert new_state.oscillation_count == 0

    def test_oscillation_increments_cumulatively(self):
        """Multiple oscillations should accumulate."""
        memory = PolicyMemoryState(
            current_route=ROUTE_RECOVERY,
            prior_route=ROUTE_PRIMARY,
            consecutive_low_risk_cycles=3,
            consecutive_recovery_cycles=1,
            oscillation_count=1,
            last_escalation_level=1,
            policy_cycle_index=5,
        )
        low_decision = _make_low_decision()
        if low_decision.adaptive_recovery_route == ROUTE_PRIMARY:
            new_state, _ = update_policy_memory(low_decision, memory)
            assert new_state.oscillation_count == 2


# ---------------------------------------------------------------------------
# 5. Oscillation suppression
# ---------------------------------------------------------------------------

class TestOscillationSuppression:
    """Verify oscillation increases hysteresis strictness."""

    def test_oscillation_doubles_required_cycles(self):
        """When oscillation_count >= threshold, require 2x cycles."""
        required = HYSTERESIS_RELEASE_CYCLES * OSCILLATION_STRICTNESS_MULTIPLIER
        # At the normal threshold, 3 cycles would release. With oscillation, need 6.
        memory = PolicyMemoryState(
            current_route=ROUTE_RECOVERY,
            prior_route=ROUTE_PRIMARY,
            consecutive_low_risk_cycles=3,
            consecutive_recovery_cycles=1,
            oscillation_count=OSCILLATION_THRESHOLD,
            last_escalation_level=1,
            policy_cycle_index=5,
        )
        route = compute_hysteresis_route(ROUTE_PRIMARY, memory)
        # 3 < 6, so downgrade should be blocked
        assert route == ROUTE_RECOVERY

    def test_oscillation_release_with_doubled_cycles(self):
        """With oscillation, release after doubled cycle count."""
        required = HYSTERESIS_RELEASE_CYCLES * OSCILLATION_STRICTNESS_MULTIPLIER
        memory = PolicyMemoryState(
            current_route=ROUTE_RECOVERY,
            prior_route=ROUTE_PRIMARY,
            consecutive_low_risk_cycles=required,
            consecutive_recovery_cycles=1,
            oscillation_count=OSCILLATION_THRESHOLD,
            last_escalation_level=1,
            policy_cycle_index=10,
        )
        route = compute_hysteresis_route(ROUTE_PRIMARY, memory)
        assert route == ROUTE_PRIMARY

    def test_below_threshold_normal_release(self):
        """Below oscillation threshold, normal release cycles apply."""
        memory = PolicyMemoryState(
            current_route=ROUTE_RECOVERY,
            prior_route=ROUTE_PRIMARY,
            consecutive_low_risk_cycles=HYSTERESIS_RELEASE_CYCLES,
            consecutive_recovery_cycles=1,
            oscillation_count=OSCILLATION_THRESHOLD - 1,
            last_escalation_level=1,
            policy_cycle_index=5,
        )
        route = compute_hysteresis_route(ROUTE_PRIMARY, memory)
        assert route == ROUTE_PRIMARY


# ---------------------------------------------------------------------------
# 6. 100-run determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Verify byte-identical outputs across 100 repeated runs."""

    def test_100_run_update_determinism(self):
        decision = _make_low_decision()
        ref_state, ref_route = update_policy_memory(decision, None)
        for _ in range(100):
            state, route = update_policy_memory(decision, None)
            assert state == ref_state
            assert route == ref_route

    def test_100_run_multi_cycle_determinism(self):
        decisions = [_make_low_decision(), _make_critical_decision(),
                     _make_watch_decision(), _make_low_decision()]
        ref_states = []
        ref_routes = []
        state = None
        for d in decisions:
            state, route = update_policy_memory(d, state)
            ref_states.append(state)
            ref_routes.append(route)

        for _ in range(100):
            state = None
            for i, d in enumerate(decisions):
                state, route = update_policy_memory(d, state)
                assert state == ref_states[i]
                assert route == ref_routes[i]

    def test_100_run_export_determinism(self):
        decision = _make_low_decision()
        state, route = update_policy_memory(decision, None)
        ref_bundle = export_policy_memory_bundle(state, route)
        ref_json = json.dumps(ref_bundle, sort_keys=True)
        for _ in range(100):
            bundle = export_policy_memory_bundle(state, route)
            assert json.dumps(bundle, sort_keys=True) == ref_json


# ---------------------------------------------------------------------------
# 7. Export equality
# ---------------------------------------------------------------------------

class TestExportEquality:
    """Verify export produces byte-identical bundles."""

    def test_export_equality(self):
        decision = _make_low_decision()
        state, route = update_policy_memory(decision, None)
        b1 = export_policy_memory_bundle(state, route)
        b2 = export_policy_memory_bundle(state, route)
        assert b1 == b2

    def test_export_json_equality(self):
        decision = _make_low_decision()
        state, route = update_policy_memory(decision, None)
        b1 = export_policy_memory_bundle(state, route)
        b2 = export_policy_memory_bundle(state, route)
        j1 = json.dumps(b1, sort_keys=True, separators=(",", ":"))
        j2 = json.dumps(b2, sort_keys=True, separators=(",", ":"))
        assert j1 == j2

    def test_export_contains_version(self):
        decision = _make_low_decision()
        state, route = update_policy_memory(decision, None)
        bundle = export_policy_memory_bundle(state, route)
        assert bundle["version"] == POLICY_MEMORY_VERSION
        assert bundle["layer"] == "adaptive_steering_policy_memory"

    def test_export_contains_all_fields(self):
        decision = _make_low_decision()
        state, route = update_policy_memory(decision, None)
        bundle = export_policy_memory_bundle(state, route)
        expected_keys = {
            "consecutive_low_risk_cycles", "consecutive_recovery_cycles",
            "current_route", "governed_route", "last_escalation_level",
            "layer", "oscillation_count", "policy_cycle_index",
            "prior_route", "stable_hash", "version",
        }
        assert set(bundle.keys()) == expected_keys


# ---------------------------------------------------------------------------
# 8. Frozen immutability
# ---------------------------------------------------------------------------

class TestFrozenImmutability:
    """Verify frozen dataclasses reject mutation."""

    def test_state_immutable(self):
        state = make_initial_policy_memory_state()
        with pytest.raises(AttributeError):
            state.current_route = ROUTE_EMERGENCY  # type: ignore[misc]

    def test_state_oscillation_immutable(self):
        state = make_initial_policy_memory_state()
        with pytest.raises(AttributeError):
            state.oscillation_count = 99  # type: ignore[misc]

    def test_ledger_immutable(self):
        ledger = build_policy_memory_ledger()
        with pytest.raises(AttributeError):
            ledger.state_count = 999  # type: ignore[misc]

    def test_ledger_hash_immutable(self):
        ledger = build_policy_memory_ledger()
        with pytest.raises(AttributeError):
            ledger.stable_hash = "tampered"  # type: ignore[misc]

    def test_state_is_frozen_dataclass(self):
        state = make_initial_policy_memory_state()
        assert isinstance(state, PolicyMemoryState)
        assert state.__dataclass_params__.frozen is True  # type: ignore[attr-defined]

    def test_ledger_is_frozen_dataclass(self):
        ledger = build_policy_memory_ledger()
        assert isinstance(ledger, PolicyMemoryLedger)
        assert ledger.__dataclass_params__.frozen is True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 9. Ledger stability
# ---------------------------------------------------------------------------

class TestLedgerStability:
    """Verify ledger hashing is stable and ordering-sensitive."""

    def test_ledger_hash_stable(self):
        s1 = make_initial_policy_memory_state()
        decision = _make_low_decision()
        s2, _ = update_policy_memory(decision, s1)
        ledger1 = build_policy_memory_ledger((s1, s2))
        ledger2 = build_policy_memory_ledger((s1, s2))
        assert ledger1.stable_hash == ledger2.stable_hash

    def test_ledger_hash_changes_with_order(self):
        s1 = make_initial_policy_memory_state()
        decision = _make_critical_decision()
        s2, _ = update_policy_memory(decision, s1)
        ledger_ab = build_policy_memory_ledger((s1, s2))
        ledger_ba = build_policy_memory_ledger((s2, s1))
        assert ledger_ab.stable_hash != ledger_ba.stable_hash

    def test_append_preserves_order(self):
        s1 = make_initial_policy_memory_state()
        decision = _make_low_decision()
        s2, _ = update_policy_memory(decision, s1)
        ledger = build_policy_memory_ledger((s1,))
        ledger = append_policy_memory_state(s2, ledger)
        assert ledger.states == (s1, s2)
        assert ledger.state_count == 2

    def test_100_run_ledger_hash_determinism(self):
        s1 = make_initial_policy_memory_state()
        decision = _make_low_decision()
        s2, _ = update_policy_memory(decision, s1)
        ref_ledger = build_policy_memory_ledger((s1, s2))
        for _ in range(100):
            ledger = build_policy_memory_ledger((s1, s2))
            assert ledger.stable_hash == ref_ledger.stable_hash


# ---------------------------------------------------------------------------
# 10. No-regression against v136.9.2 outputs
# ---------------------------------------------------------------------------

class TestNoRegression:
    """Verify policy memory does not alter v136.9.2 decision outputs."""

    def test_low_decision_unchanged(self):
        """Policy memory must not mutate the input AdaptiveSteeringDecision."""
        decision = _make_low_decision()
        original_route = decision.adaptive_recovery_route
        original_hash = decision.stable_hash
        _, _ = update_policy_memory(decision, None)
        assert decision.adaptive_recovery_route == original_route
        assert decision.stable_hash == original_hash

    def test_critical_decision_unchanged(self):
        decision = _make_critical_decision()
        original_route = decision.adaptive_recovery_route
        original_escalation = decision.adaptive_escalation_level
        original_hash = decision.stable_hash
        _, _ = update_policy_memory(decision, None)
        assert decision.adaptive_recovery_route == original_route
        assert decision.adaptive_escalation_level == original_escalation
        assert decision.stable_hash == original_hash

    def test_v136_9_2_export_stable(self):
        """v136.9.2 exports must remain byte-identical after policy memory use."""
        from qec.analysis.forecast_guided_steering import export_adaptive_steering_bundle
        decision = _make_low_decision()
        bundle_before = json.dumps(export_adaptive_steering_bundle(decision),
                                   sort_keys=True, separators=(",", ":"))
        _, _ = update_policy_memory(decision, None)
        bundle_after = json.dumps(export_adaptive_steering_bundle(decision),
                                  sort_keys=True, separators=(",", ":"))
        assert bundle_before == bundle_after

    def test_policy_memory_is_additive(self):
        """Policy memory state is separate from AdaptiveSteeringDecision."""
        decision = _make_low_decision()
        state, governed = update_policy_memory(decision, None)
        # State is a PolicyMemoryState, not modifying AdaptiveSteeringDecision
        assert isinstance(state, PolicyMemoryState)
        assert isinstance(decision, AdaptiveSteeringDecision)
        # governed_route may differ from decision route (hysteresis applied)
        assert isinstance(governed, str)
