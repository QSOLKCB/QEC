"""Tests for v137.0.9 — Supervisory Steering Policy Memory.

Covers:
  - dominant action computation & tie-breaking
  - cumulative drift score bounds
  - oscillation detection
  - cooldown window logic
  - hysteresis state persistence
  - policy memory class precedence
  - symbolic trace determinism
  - frozen immutability
  - stable hashing
  - export equality
  - 100-run replay
  - invalid input rejection
  - no decoder contamination
  - ledger construction

Target: 80–100 tests.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from qec.analysis.forecast_guided_supervisory_steering import (
    STEERING_HOLD,
    STEERING_DAMPEN,
    STEERING_AMPLIFY,
    STEERING_REDIRECT,
    STEERING_LOCKDOWN,
    SteeringDecision,
)
from qec.analysis.supervisory_steering_policy_memory import (
    SUPERVISORY_STEERING_POLICY_MEMORY_VERSION,
    POLICY_STABLE,
    POLICY_HYSTERETIC,
    POLICY_OSCILLATING,
    POLICY_LOCKED,
    POLICY_COOLDOWN,
    HYSTERESIS_NEUTRAL,
    HYSTERESIS_PERSIST_HOLD,
    HYSTERESIS_PERSIST_DAMPEN,
    HYSTERESIS_PERSIST_AMPLIFY,
    HYSTERESIS_PERSIST_REDIRECT,
    HYSTERESIS_LOCKDOWN_MEMORY,
    COOLDOWN_HORIZON_WINDOW,
    FLOAT_PRECISION,
    SteeringPolicyState,
    SteeringPolicyLedger,
    build_supervisory_policy_memory,
    build_supervisory_policy_memory_ledger,
    export_supervisory_policy_memory_bundle,
    export_supervisory_policy_memory_ledger,
    _compute_dominant_action,
    _compute_cumulative_drift_score,
    _detect_oscillation,
    _compute_cooldown,
    _compute_hysteresis_state,
    _classify_policy_memory,
    _build_symbolic_trace,
    _canonical_json,
    _round,
    _policy_state_to_canonical_dict,
    _compute_state_hash,
    _compute_ledger_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_decision(
    action: str = STEERING_HOLD,
    drift: float = 0.0,
    coherence: float = 0.8,
    trend: str = "STABLE_TREND",
    replay_valid: bool = True,
    coherence_class: str = "HIGH_COHERENCE",
) -> SteeringDecision:
    """Build a synthetic SteeringDecision for testing."""
    return SteeringDecision(
        horizon_count=5,
        dominant_trend_class=trend,
        steering_action=action,
        drift_score=drift,
        coherence_score=coherence,
        replay_valid=replay_valid,
        temporal_coherence_class=coherence_class,
        steering_symbolic_trace=f"trace->{action}",
        stable_hash=hashlib.sha256(
            f"{action}-{drift}-{coherence}".encode()
        ).hexdigest(),
    )


def _make_decisions(*actions_and_drifts):
    """Make decisions from (action, drift) pairs."""
    return tuple(
        _make_decision(action=a, drift=d)
        for a, d in actions_and_drifts
    )


# ---------------------------------------------------------------------------
# TestDominantAction
# ---------------------------------------------------------------------------


class TestDominantAction:
    """Tests for dominant action computation and tie-breaking."""

    def test_single_decision(self):
        ds = (_make_decision(action=STEERING_HOLD),)
        assert _compute_dominant_action(ds) == STEERING_HOLD

    def test_clear_majority(self):
        ds = (
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_HOLD),
        )
        assert _compute_dominant_action(ds) == STEERING_DAMPEN

    def test_tie_lockdown_wins_over_hold(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_LOCKDOWN),
        )
        assert _compute_dominant_action(ds) == STEERING_LOCKDOWN

    def test_tie_redirect_wins_over_dampen(self):
        ds = (
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_REDIRECT),
        )
        assert _compute_dominant_action(ds) == STEERING_REDIRECT

    def test_tie_dampen_wins_over_amplify(self):
        ds = (
            _make_decision(action=STEERING_AMPLIFY),
            _make_decision(action=STEERING_DAMPEN),
        )
        assert _compute_dominant_action(ds) == STEERING_DAMPEN

    def test_tie_amplify_wins_over_hold(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_AMPLIFY),
        )
        assert _compute_dominant_action(ds) == STEERING_AMPLIFY

    def test_three_way_tie(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_REDIRECT),
        )
        assert _compute_dominant_action(ds) == STEERING_REDIRECT

    def test_lockdown_wins_five_way_tie(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_AMPLIFY),
            _make_decision(action=STEERING_REDIRECT),
            _make_decision(action=STEERING_LOCKDOWN),
        )
        assert _compute_dominant_action(ds) == STEERING_LOCKDOWN

    def test_majority_overrides_severity(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_LOCKDOWN),
        )
        assert _compute_dominant_action(ds) == STEERING_HOLD

    def test_deterministic_across_runs(self):
        ds = (
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_REDIRECT),
        )
        results = [_compute_dominant_action(ds) for _ in range(100)]
        assert len(set(results)) == 1


# ---------------------------------------------------------------------------
# TestCumulativeDriftScore
# ---------------------------------------------------------------------------


class TestCumulativeDriftScore:
    """Tests for cumulative drift score computation and bounds."""

    def test_zero_drift(self):
        ds = (_make_decision(drift=0.0),)
        assert _compute_cumulative_drift_score(ds) == 0.0

    def test_positive_drift(self):
        ds = (
            _make_decision(drift=0.5),
            _make_decision(drift=0.3),
        )
        result = _compute_cumulative_drift_score(ds)
        assert result == _round(0.4)

    def test_negative_drift(self):
        ds = (
            _make_decision(drift=-0.5),
            _make_decision(drift=-0.3),
        )
        result = _compute_cumulative_drift_score(ds)
        assert result == _round(-0.4)

    def test_bounded_above(self):
        ds = (
            _make_decision(drift=1.0),
            _make_decision(drift=1.0),
        )
        assert _compute_cumulative_drift_score(ds) <= 1.0

    def test_bounded_below(self):
        ds = (
            _make_decision(drift=-1.0),
            _make_decision(drift=-1.0),
        )
        assert _compute_cumulative_drift_score(ds) >= -1.0

    def test_mixed_drift_averages(self):
        ds = (
            _make_decision(drift=0.6),
            _make_decision(drift=-0.2),
        )
        result = _compute_cumulative_drift_score(ds)
        assert result == _round(0.2)

    def test_single_decision_returns_itself(self):
        ds = (_make_decision(drift=0.7),)
        assert _compute_cumulative_drift_score(ds) == _round(0.7)

    def test_canonical_rounding(self):
        ds = (
            _make_decision(drift=1.0 / 3.0),
            _make_decision(drift=1.0 / 3.0),
            _make_decision(drift=1.0 / 3.0),
        )
        result = _compute_cumulative_drift_score(ds)
        assert result == round(1.0 / 3.0, FLOAT_PRECISION)

    def test_deterministic_across_runs(self):
        ds = (
            _make_decision(drift=0.123456789012),
            _make_decision(drift=-0.987654321098),
        )
        results = [_compute_cumulative_drift_score(ds) for _ in range(100)]
        assert len(set(results)) == 1


# ---------------------------------------------------------------------------
# TestOscillationDetection
# ---------------------------------------------------------------------------


class TestOscillationDetection:
    """Tests for oscillation detection."""

    def test_no_oscillation_single(self):
        ds = (_make_decision(action=STEERING_HOLD),)
        detected, count = _detect_oscillation(ds)
        assert not detected
        assert count == 0

    def test_no_oscillation_two_decisions(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_DAMPEN),
        )
        detected, count = _detect_oscillation(ds)
        assert not detected
        assert count == 0

    def test_no_oscillation_monotonic(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_HOLD),
        )
        detected, count = _detect_oscillation(ds)
        assert not detected
        assert count == 0

    def test_single_alternation_not_oscillation(self):
        ds = (
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_DAMPEN),
        )
        detected, count = _detect_oscillation(ds)
        assert not detected
        assert count == 1

    def test_two_alternations_oscillation(self):
        ds = (
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_HOLD),
        )
        detected, count = _detect_oscillation(ds)
        assert detected
        assert count == 2

    def test_three_alternations(self):
        ds = (
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_DAMPEN),
        )
        detected, count = _detect_oscillation(ds)
        assert detected
        assert count == 3

    def test_redirect_hold_oscillation(self):
        ds = (
            _make_decision(action=STEERING_REDIRECT),
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_REDIRECT),
            _make_decision(action=STEERING_HOLD),
        )
        detected, count = _detect_oscillation(ds)
        assert detected
        assert count == 2

    def test_no_oscillation_progressive(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_REDIRECT),
            _make_decision(action=STEERING_LOCKDOWN),
        )
        detected, count = _detect_oscillation(ds)
        assert not detected
        assert count == 0

    def test_deterministic_oscillation_count(self):
        ds = (
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_HOLD),
        )
        results = [_detect_oscillation(ds) for _ in range(100)]
        assert all(r == (True, 2) for r in results)


# ---------------------------------------------------------------------------
# TestCooldownEnforcement
# ---------------------------------------------------------------------------


class TestCooldownEnforcement:
    """Tests for cooldown window logic."""

    def test_no_cooldown_hold_only(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_HOLD),
        )
        active, remaining = _compute_cooldown(ds)
        assert not active
        assert remaining == 0

    def test_cooldown_from_lockdown(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_LOCKDOWN),
        )
        active, remaining = _compute_cooldown(ds)
        assert active
        assert remaining == 1

    def test_cooldown_from_redirect(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_REDIRECT),
        )
        active, remaining = _compute_cooldown(ds)
        assert active
        assert remaining == 1

    def test_cooldown_both_trailing(self):
        ds = (
            _make_decision(action=STEERING_LOCKDOWN),
            _make_decision(action=STEERING_REDIRECT),
        )
        active, remaining = _compute_cooldown(ds)
        assert active
        assert remaining == 2

    def test_cooldown_expired(self):
        ds = (
            _make_decision(action=STEERING_LOCKDOWN),
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_HOLD),
        )
        active, remaining = _compute_cooldown(ds)
        assert not active
        assert remaining == 0

    def test_cooldown_window_size(self):
        assert COOLDOWN_HORIZON_WINDOW == 2

    def test_cooldown_dampen_not_trigger(self):
        ds = (
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_DAMPEN),
        )
        active, remaining = _compute_cooldown(ds)
        assert not active

    def test_cooldown_amplify_not_trigger(self):
        ds = (
            _make_decision(action=STEERING_AMPLIFY),
            _make_decision(action=STEERING_AMPLIFY),
        )
        active, remaining = _compute_cooldown(ds)
        assert not active

    def test_single_decision_lockdown(self):
        ds = (_make_decision(action=STEERING_LOCKDOWN),)
        active, remaining = _compute_cooldown(ds)
        assert active
        assert remaining == 1


# ---------------------------------------------------------------------------
# TestHysteresisState
# ---------------------------------------------------------------------------


class TestHysteresisState:
    """Tests for hysteresis state persistence."""

    def test_single_hold(self):
        ds = (_make_decision(action=STEERING_HOLD),)
        assert _compute_hysteresis_state(ds) == HYSTERESIS_PERSIST_HOLD

    def test_single_dampen(self):
        ds = (_make_decision(action=STEERING_DAMPEN),)
        assert _compute_hysteresis_state(ds) == HYSTERESIS_PERSIST_DAMPEN

    def test_repeated_hold(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_HOLD),
        )
        assert _compute_hysteresis_state(ds) == HYSTERESIS_PERSIST_HOLD

    def test_repeated_dampen(self):
        ds = (
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_DAMPEN),
        )
        assert _compute_hysteresis_state(ds) == HYSTERESIS_PERSIST_DAMPEN

    def test_repeated_amplify(self):
        ds = (
            _make_decision(action=STEERING_AMPLIFY),
            _make_decision(action=STEERING_AMPLIFY),
        )
        assert _compute_hysteresis_state(ds) == HYSTERESIS_PERSIST_AMPLIFY

    def test_repeated_redirect(self):
        ds = (
            _make_decision(action=STEERING_REDIRECT),
            _make_decision(action=STEERING_REDIRECT),
        )
        assert _compute_hysteresis_state(ds) == HYSTERESIS_PERSIST_REDIRECT

    def test_repeated_lockdown(self):
        ds = (
            _make_decision(action=STEERING_LOCKDOWN),
            _make_decision(action=STEERING_LOCKDOWN),
        )
        assert _compute_hysteresis_state(ds) == HYSTERESIS_LOCKDOWN_MEMORY

    def test_different_actions_neutral(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_DAMPEN),
        )
        assert _compute_hysteresis_state(ds) == HYSTERESIS_NEUTRAL

    def test_lockdown_last_overrides_neutral(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_LOCKDOWN),
        )
        assert _compute_hysteresis_state(ds) == HYSTERESIS_LOCKDOWN_MEMORY

    def test_redirect_then_hold_neutral(self):
        ds = (
            _make_decision(action=STEERING_REDIRECT),
            _make_decision(action=STEERING_HOLD),
        )
        assert _compute_hysteresis_state(ds) == HYSTERESIS_NEUTRAL


# ---------------------------------------------------------------------------
# TestPolicyMemoryClass
# ---------------------------------------------------------------------------


class TestPolicyMemoryClass:
    """Tests for policy memory class precedence."""

    def test_oscillation_highest_priority(self):
        result = _classify_policy_memory(
            oscillation_detected=True,
            cooldown_active=True,
            hysteresis_state=HYSTERESIS_LOCKDOWN_MEMORY,
            dominant_action=STEERING_LOCKDOWN,
            cumulative_drift_score=0.0,
        )
        assert result == POLICY_OSCILLATING

    def test_cooldown_second_priority(self):
        result = _classify_policy_memory(
            oscillation_detected=False,
            cooldown_active=True,
            hysteresis_state=HYSTERESIS_LOCKDOWN_MEMORY,
            dominant_action=STEERING_LOCKDOWN,
            cumulative_drift_score=0.0,
        )
        assert result == POLICY_COOLDOWN

    def test_locked_third_priority(self):
        result = _classify_policy_memory(
            oscillation_detected=False,
            cooldown_active=False,
            hysteresis_state=HYSTERESIS_PERSIST_DAMPEN,
            dominant_action=STEERING_LOCKDOWN,
            cumulative_drift_score=0.0,
        )
        assert result == POLICY_LOCKED

    def test_hysteretic_fourth_priority(self):
        result = _classify_policy_memory(
            oscillation_detected=False,
            cooldown_active=False,
            hysteresis_state=HYSTERESIS_PERSIST_DAMPEN,
            dominant_action=STEERING_DAMPEN,
            cumulative_drift_score=0.5,
        )
        assert result == POLICY_HYSTERETIC

    def test_stable_hold_low_drift(self):
        result = _classify_policy_memory(
            oscillation_detected=False,
            cooldown_active=False,
            hysteresis_state=HYSTERESIS_NEUTRAL,
            dominant_action=STEERING_HOLD,
            cumulative_drift_score=0.05,
        )
        assert result == POLICY_STABLE

    def test_stable_fallback(self):
        result = _classify_policy_memory(
            oscillation_detected=False,
            cooldown_active=False,
            hysteresis_state=HYSTERESIS_NEUTRAL,
            dominant_action=STEERING_DAMPEN,
            cumulative_drift_score=0.05,
        )
        assert result == POLICY_STABLE

    def test_stable_at_drift_boundary(self):
        result = _classify_policy_memory(
            oscillation_detected=False,
            cooldown_active=False,
            hysteresis_state=HYSTERESIS_NEUTRAL,
            dominant_action=STEERING_HOLD,
            cumulative_drift_score=0.1,
        )
        assert result == POLICY_STABLE


# ---------------------------------------------------------------------------
# TestSymbolicTrace
# ---------------------------------------------------------------------------


class TestSymbolicTrace:
    """Tests for symbolic trace generation."""

    def test_single_action(self):
        ds = (_make_decision(action=STEERING_HOLD),)
        trace = _build_symbolic_trace(ds, POLICY_STABLE)
        assert trace == "HOLD -> STABLE_POLICY"

    def test_multiple_actions(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_DAMPEN),
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_DAMPEN),
        )
        trace = _build_symbolic_trace(ds, POLICY_OSCILLATING)
        assert trace == "HOLD -> DAMPEN -> HOLD -> DAMPEN -> OSCILLATING_POLICY"

    def test_lockdown_trace(self):
        ds = (
            _make_decision(action=STEERING_LOCKDOWN),
            _make_decision(action=STEERING_LOCKDOWN),
        )
        trace = _build_symbolic_trace(ds, POLICY_LOCKED)
        assert trace == "LOCKDOWN -> LOCKDOWN -> LOCKED_POLICY"

    def test_trace_deterministic(self):
        ds = (
            _make_decision(action=STEERING_REDIRECT),
            _make_decision(action=STEERING_HOLD),
        )
        results = [_build_symbolic_trace(ds, POLICY_COOLDOWN) for _ in range(100)]
        assert len(set(results)) == 1

    def test_trace_format_arrow_separator(self):
        ds = (
            _make_decision(action=STEERING_AMPLIFY),
            _make_decision(action=STEERING_DAMPEN),
        )
        trace = _build_symbolic_trace(ds, POLICY_HYSTERETIC)
        assert " -> " in trace
        parts = trace.split(" -> ")
        assert len(parts) == 3


# ---------------------------------------------------------------------------
# TestFrozenImmutability
# ---------------------------------------------------------------------------


class TestFrozenImmutability:
    """Tests for frozen dataclass immutability."""

    def test_policy_state_frozen(self):
        ds = (_make_decision(),)
        state = build_supervisory_policy_memory(ds)
        with pytest.raises(AttributeError):
            state.history_length = 999

    def test_policy_state_frozen_dominant_action(self):
        ds = (_make_decision(),)
        state = build_supervisory_policy_memory(ds)
        with pytest.raises(AttributeError):
            state.dominant_action = "INVALID"

    def test_policy_state_frozen_hash(self):
        ds = (_make_decision(),)
        state = build_supervisory_policy_memory(ds)
        with pytest.raises(AttributeError):
            state.stable_hash = "tampered"

    def test_policy_state_frozen_version(self):
        ds = (_make_decision(),)
        state = build_supervisory_policy_memory(ds)
        with pytest.raises(AttributeError):
            state.version = "wrong"

    def test_ledger_frozen(self):
        ds = (_make_decision(),)
        state = build_supervisory_policy_memory(ds)
        ledger = build_supervisory_policy_memory_ledger([state])
        with pytest.raises(AttributeError):
            ledger.state_count = 999

    def test_ledger_frozen_hash(self):
        ds = (_make_decision(),)
        state = build_supervisory_policy_memory(ds)
        ledger = build_supervisory_policy_memory_ledger([state])
        with pytest.raises(AttributeError):
            ledger.stable_hash = "tampered"


# ---------------------------------------------------------------------------
# TestStableHashing
# ---------------------------------------------------------------------------


class TestStableHashing:
    """Tests for SHA-256 stable hashing."""

    def test_hash_is_hex_string(self):
        ds = (_make_decision(),)
        state = build_supervisory_policy_memory(ds)
        assert all(c in "0123456789abcdef" for c in state.stable_hash)

    def test_hash_length_64(self):
        ds = (_make_decision(),)
        state = build_supervisory_policy_memory(ds)
        assert len(state.stable_hash) == 64

    def test_hash_deterministic(self):
        ds = (_make_decision(),)
        h1 = build_supervisory_policy_memory(ds).stable_hash
        h2 = build_supervisory_policy_memory(ds).stable_hash
        assert h1 == h2

    def test_different_inputs_different_hashes(self):
        d1 = (_make_decision(action=STEERING_HOLD),)
        d2 = (_make_decision(action=STEERING_LOCKDOWN),)
        h1 = build_supervisory_policy_memory(d1).stable_hash
        h2 = build_supervisory_policy_memory(d2).stable_hash
        assert h1 != h2

    def test_ledger_hash_is_hex(self):
        state = build_supervisory_policy_memory((_make_decision(),))
        ledger = build_supervisory_policy_memory_ledger([state])
        assert all(c in "0123456789abcdef" for c in ledger.stable_hash)

    def test_ledger_hash_length_64(self):
        state = build_supervisory_policy_memory((_make_decision(),))
        ledger = build_supervisory_policy_memory_ledger([state])
        assert len(ledger.stable_hash) == 64

    def test_ledger_hash_deterministic(self):
        state = build_supervisory_policy_memory((_make_decision(),))
        l1 = build_supervisory_policy_memory_ledger([state])
        l2 = build_supervisory_policy_memory_ledger([state])
        assert l1.stable_hash == l2.stable_hash

    def test_canonical_json_sorted_keys(self):
        obj = {"b": 2, "a": 1}
        result = _canonical_json(obj)
        assert result == '{"a":1,"b":2}'

    def test_canonical_json_compact(self):
        obj = {"x": [1, 2, 3]}
        result = _canonical_json(obj)
        assert " " not in result


# ---------------------------------------------------------------------------
# TestExport
# ---------------------------------------------------------------------------


class TestExport:
    """Tests for export functions."""

    def test_bundle_contains_all_fields(self):
        state = build_supervisory_policy_memory((_make_decision(),))
        bundle = export_supervisory_policy_memory_bundle(state)
        assert "history_length" in bundle
        assert "dominant_action" in bundle
        assert "hysteresis_state" in bundle
        assert "cooldown_active" in bundle
        assert "cooldown_remaining" in bundle
        assert "oscillation_detected" in bundle
        assert "oscillation_count" in bundle
        assert "cumulative_drift_score" in bundle
        assert "policy_memory_class" in bundle
        assert "policy_symbolic_trace" in bundle
        assert "stable_hash" in bundle
        assert "layer" in bundle
        assert "version" in bundle

    def test_bundle_layer(self):
        state = build_supervisory_policy_memory((_make_decision(),))
        bundle = export_supervisory_policy_memory_bundle(state)
        assert bundle["layer"] == "supervisory_steering_policy_memory"

    def test_bundle_deterministic(self):
        state = build_supervisory_policy_memory((_make_decision(),))
        b1 = export_supervisory_policy_memory_bundle(state)
        b2 = export_supervisory_policy_memory_bundle(state)
        assert _canonical_json(b1) == _canonical_json(b2)

    def test_ledger_export_contains_fields(self):
        state = build_supervisory_policy_memory((_make_decision(),))
        ledger = build_supervisory_policy_memory_ledger([state])
        export = export_supervisory_policy_memory_ledger(ledger)
        assert "state_count" in export
        assert "states" in export
        assert "stable_hash" in export
        assert "layer" in export
        assert "version" in export

    def test_ledger_export_version(self):
        state = build_supervisory_policy_memory((_make_decision(),))
        ledger = build_supervisory_policy_memory_ledger([state])
        export = export_supervisory_policy_memory_ledger(ledger)
        assert export["version"] == SUPERVISORY_STEERING_POLICY_MEMORY_VERSION

    def test_ledger_export_state_count(self):
        state = build_supervisory_policy_memory((_make_decision(),))
        ledger = build_supervisory_policy_memory_ledger([state, state])
        export = export_supervisory_policy_memory_ledger(ledger)
        assert export["state_count"] == 2
        assert len(export["states"]) == 2

    def test_ledger_export_deterministic(self):
        state = build_supervisory_policy_memory((_make_decision(),))
        ledger = build_supervisory_policy_memory_ledger([state])
        e1 = export_supervisory_policy_memory_ledger(ledger)
        e2 = export_supervisory_policy_memory_ledger(ledger)
        assert _canonical_json(e1) == _canonical_json(e2)

    def test_export_json_serializable(self):
        state = build_supervisory_policy_memory((_make_decision(),))
        bundle = export_supervisory_policy_memory_bundle(state)
        # Must not raise
        json.dumps(bundle)

    def test_ledger_export_json_serializable(self):
        state = build_supervisory_policy_memory((_make_decision(),))
        ledger = build_supervisory_policy_memory_ledger([state])
        export = export_supervisory_policy_memory_ledger(ledger)
        # Must not raise
        json.dumps(export)


# ---------------------------------------------------------------------------
# TestReplayDeterminism
# ---------------------------------------------------------------------------


class TestReplayDeterminism:
    """100-run replay determinism tests."""

    def test_100_run_single_decision(self):
        ds = (_make_decision(action=STEERING_HOLD, drift=0.1),)
        baseline = build_supervisory_policy_memory(ds)
        for _ in range(99):
            state = build_supervisory_policy_memory(ds)
            assert state == baseline

    def test_100_run_oscillating(self):
        ds = (
            _make_decision(action=STEERING_DAMPEN, drift=0.3),
            _make_decision(action=STEERING_HOLD, drift=-0.1),
            _make_decision(action=STEERING_DAMPEN, drift=0.3),
            _make_decision(action=STEERING_HOLD, drift=-0.1),
        )
        baseline = build_supervisory_policy_memory(ds)
        for _ in range(99):
            state = build_supervisory_policy_memory(ds)
            assert state == baseline

    def test_100_run_export_bytes(self):
        ds = (
            _make_decision(action=STEERING_REDIRECT, drift=-0.5),
            _make_decision(action=STEERING_LOCKDOWN, drift=-0.8),
        )
        state = build_supervisory_policy_memory(ds)
        baseline = _canonical_json(export_supervisory_policy_memory_bundle(state))
        for _ in range(99):
            s = build_supervisory_policy_memory(ds)
            result = _canonical_json(export_supervisory_policy_memory_bundle(s))
            assert result == baseline

    def test_100_run_ledger(self):
        ds = (_make_decision(action=STEERING_HOLD),)
        state = build_supervisory_policy_memory(ds)
        baseline = build_supervisory_policy_memory_ledger([state])
        for _ in range(99):
            s = build_supervisory_policy_memory(ds)
            ledger = build_supervisory_policy_memory_ledger([s])
            assert ledger == baseline


# ---------------------------------------------------------------------------
# TestInputValidation
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Tests for invalid input rejection."""

    def test_reject_empty_list(self):
        with pytest.raises(ValueError, match="must not be empty"):
            build_supervisory_policy_memory([])

    def test_reject_empty_tuple(self):
        with pytest.raises(ValueError, match="must not be empty"):
            build_supervisory_policy_memory(())

    def test_reject_none(self):
        with pytest.raises(TypeError, match="must be a list or tuple"):
            build_supervisory_policy_memory(None)

    def test_reject_string(self):
        with pytest.raises(TypeError, match="must be a list or tuple"):
            build_supervisory_policy_memory("not a sequence")

    def test_reject_wrong_type_in_list(self):
        with pytest.raises(TypeError, match="must be SteeringDecision"):
            build_supervisory_policy_memory(["not a decision"])

    def test_reject_mixed_types(self):
        with pytest.raises(TypeError, match="must be SteeringDecision"):
            build_supervisory_policy_memory([_make_decision(), "bad"])

    def test_reject_int(self):
        with pytest.raises(TypeError, match="must be a list or tuple"):
            build_supervisory_policy_memory(42)

    def test_ledger_reject_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            build_supervisory_policy_memory_ledger([])

    def test_ledger_reject_none(self):
        with pytest.raises(TypeError, match="must be a list or tuple"):
            build_supervisory_policy_memory_ledger(None)

    def test_ledger_reject_wrong_type(self):
        with pytest.raises(TypeError, match="must be SteeringPolicyState"):
            build_supervisory_policy_memory_ledger(["not a state"])


# ---------------------------------------------------------------------------
# TestNoDecoderContamination
# ---------------------------------------------------------------------------


class TestNoDecoderContamination:
    """Verify no decoder imports."""

    def test_no_decoder_import(self):
        import inspect
        import qec.analysis.supervisory_steering_policy_memory as mod

        source = inspect.getsource(mod)
        assert "from qec.decoder" not in source
        assert "import qec.decoder" not in source

    def test_module_layer(self):
        import qec.analysis.supervisory_steering_policy_memory as mod

        assert hasattr(mod, "SUPERVISORY_STEERING_POLICY_MEMORY_VERSION")
        assert mod.SUPERVISORY_STEERING_POLICY_MEMORY_VERSION == "v137.0.9"


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests."""

    def test_stable_policy_all_hold(self):
        ds = [_make_decision(action=STEERING_HOLD, drift=0.0) for _ in range(5)]
        state = build_supervisory_policy_memory(ds)
        assert state.policy_memory_class == POLICY_STABLE
        assert state.dominant_action == STEERING_HOLD
        assert not state.oscillation_detected
        assert not state.cooldown_active

    def test_oscillating_policy(self):
        ds = [
            _make_decision(action=STEERING_DAMPEN, drift=0.3),
            _make_decision(action=STEERING_HOLD, drift=-0.1),
            _make_decision(action=STEERING_DAMPEN, drift=0.3),
            _make_decision(action=STEERING_HOLD, drift=-0.1),
        ]
        state = build_supervisory_policy_memory(ds)
        assert state.policy_memory_class == POLICY_OSCILLATING
        assert state.oscillation_detected
        assert state.oscillation_count == 2

    def test_cooldown_policy(self):
        ds = [
            _make_decision(action=STEERING_HOLD, drift=0.0),
            _make_decision(action=STEERING_HOLD, drift=0.0),
            _make_decision(action=STEERING_LOCKDOWN, drift=-0.9),
        ]
        state = build_supervisory_policy_memory(ds)
        assert state.cooldown_active
        assert state.cooldown_remaining >= 1

    def test_locked_policy(self):
        ds = [
            _make_decision(action=STEERING_LOCKDOWN, drift=-0.9),
            _make_decision(action=STEERING_LOCKDOWN, drift=-0.8),
            _make_decision(action=STEERING_LOCKDOWN, drift=-0.7),
        ]
        state = build_supervisory_policy_memory(ds)
        assert state.dominant_action == STEERING_LOCKDOWN
        assert state.hysteresis_state == HYSTERESIS_LOCKDOWN_MEMORY

    def test_hysteretic_policy_dampen(self):
        ds = [
            _make_decision(action=STEERING_DAMPEN, drift=0.3),
            _make_decision(action=STEERING_DAMPEN, drift=0.2),
            _make_decision(action=STEERING_DAMPEN, drift=0.1),
        ]
        state = build_supervisory_policy_memory(ds)
        assert state.hysteresis_state == HYSTERESIS_PERSIST_DAMPEN
        assert state.dominant_action == STEERING_DAMPEN

    def test_version_field(self):
        ds = (_make_decision(),)
        state = build_supervisory_policy_memory(ds)
        assert state.version == "v137.0.9"

    def test_history_length(self):
        ds = [_make_decision() for _ in range(7)]
        state = build_supervisory_policy_memory(ds)
        assert state.history_length == 7

    def test_accepts_list_input(self):
        ds = [_make_decision(), _make_decision()]
        state = build_supervisory_policy_memory(ds)
        assert state.history_length == 2

    def test_accepts_tuple_input(self):
        ds = (_make_decision(), _make_decision())
        state = build_supervisory_policy_memory(ds)
        assert state.history_length == 2

    def test_ledger_state_count(self):
        s1 = build_supervisory_policy_memory((_make_decision(),))
        s2 = build_supervisory_policy_memory(
            (_make_decision(action=STEERING_DAMPEN),)
        )
        ledger = build_supervisory_policy_memory_ledger([s1, s2])
        assert ledger.state_count == 2
        assert len(ledger.states) == 2

    def test_symbolic_trace_in_state(self):
        ds = (
            _make_decision(action=STEERING_HOLD),
            _make_decision(action=STEERING_DAMPEN),
        )
        state = build_supervisory_policy_memory(ds)
        assert "HOLD" in state.policy_symbolic_trace
        assert "DAMPEN" in state.policy_symbolic_trace
        assert " -> " in state.policy_symbolic_trace

    def test_cumulative_drift_in_state(self):
        ds = (
            _make_decision(drift=0.5),
            _make_decision(drift=0.3),
        )
        state = build_supervisory_policy_memory(ds)
        assert state.cumulative_drift_score == _round(0.4)

    def test_full_pipeline(self):
        ds = [
            _make_decision(action=STEERING_HOLD, drift=0.0),
            _make_decision(action=STEERING_DAMPEN, drift=0.2),
            _make_decision(action=STEERING_HOLD, drift=-0.1),
            _make_decision(action=STEERING_DAMPEN, drift=0.3),
            _make_decision(action=STEERING_HOLD, drift=-0.05),
        ]
        state = build_supervisory_policy_memory(ds)
        ledger = build_supervisory_policy_memory_ledger([state])
        bundle = export_supervisory_policy_memory_bundle(state)
        ledger_export = export_supervisory_policy_memory_ledger(ledger)

        assert state.history_length == 5
        assert len(state.stable_hash) == 64
        assert ledger.state_count == 1
        assert len(ledger.stable_hash) == 64
        assert bundle["layer"] == "supervisory_steering_policy_memory"
        assert ledger_export["version"] == "v137.0.9"
