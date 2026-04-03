"""
Tests for Phase-Space Decoder Steering — v136.9.0

Covers:
  - risk score bounded [0, 1]
  - 100-run determinism check
  - stable ordering check
  - low-risk route test
  - high-risk route test
  - export equality across repeated runs
  - input non-mutation
"""

from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.phase_space_decoder_steering import (
    ESCALATION_ADVISORY,
    ESCALATION_CRITICAL,
    ESCALATION_NONE,
    ESCALATION_WARNING,
    ROUTE_ALTERNATE,
    ROUTE_EMERGENCY,
    ROUTE_PRIMARY,
    ROUTE_RECOVERY,
    STEERING_VERSION,
    PhaseSteeringDecision,
    SteeringLedger,
    append_steering_decision,
    build_steering_ledger,
    compute_phase_risk_score,
    export_phase_steering_bundle,
    route_decoder_from_phase_space,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Low-risk scenario: small displacement, low negative mass, low drift
LOW_RISK = {
    "centroid_q": 0.05,
    "centroid_p": 0.02,
    "negative_mass": 0.01,
    "drift_momentum": 0.01,
}

# Medium-risk scenario
MEDIUM_RISK = {
    "centroid_q": 0.4,
    "centroid_p": 0.3,
    "negative_mass": 0.35,
    "drift_momentum": 0.4,
}

# High-risk scenario: large displacement, high negative mass, high drift
HIGH_RISK = {
    "centroid_q": 0.8,
    "centroid_p": 0.7,
    "negative_mass": 0.8,
    "drift_momentum": 0.9,
}

# Critical-risk scenario: extreme values
CRITICAL_RISK = {
    "centroid_q": 1.0,
    "centroid_p": 1.0,
    "negative_mass": 1.0,
    "drift_momentum": 1.0,
}


# ---------------------------------------------------------------------------
# Test: risk score bounded [0, 1]
# ---------------------------------------------------------------------------

class TestRiskScoreBounded:
    """Risk score must always be in [0, 1]."""

    def test_zero_inputs(self) -> None:
        score = compute_phase_risk_score(0.0, 0.0, 0.0, 0.0)
        assert 0.0 <= score <= 1.0
        assert score == 0.0

    def test_low_risk_bounded(self) -> None:
        score = compute_phase_risk_score(**LOW_RISK)
        assert 0.0 <= score <= 1.0

    def test_medium_risk_bounded(self) -> None:
        score = compute_phase_risk_score(**MEDIUM_RISK)
        assert 0.0 <= score <= 1.0

    def test_high_risk_bounded(self) -> None:
        score = compute_phase_risk_score(**HIGH_RISK)
        assert 0.0 <= score <= 1.0

    def test_critical_risk_bounded(self) -> None:
        score = compute_phase_risk_score(**CRITICAL_RISK)
        assert 0.0 <= score <= 1.0

    def test_extreme_positive(self) -> None:
        score = compute_phase_risk_score(10.0, 10.0, 10.0, 10.0)
        assert 0.0 <= score <= 1.0

    def test_extreme_negative(self) -> None:
        score = compute_phase_risk_score(-5.0, -5.0, 0.0, -5.0)
        assert 0.0 <= score <= 1.0

    def test_negative_mass_clamped(self) -> None:
        score = compute_phase_risk_score(0.0, 0.0, 5.0, 0.0)
        assert 0.0 <= score <= 1.0

    def test_monotonic_with_radius(self) -> None:
        s1 = compute_phase_risk_score(0.1, 0.1, 0.0, 0.0)
        s2 = compute_phase_risk_score(0.5, 0.5, 0.0, 0.0)
        s3 = compute_phase_risk_score(0.9, 0.9, 0.0, 0.0)
        assert s1 <= s2 <= s3


# ---------------------------------------------------------------------------
# Test: 100-run determinism check
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same inputs must produce identical outputs across 100 runs."""

    def test_risk_score_determinism_100(self) -> None:
        scores = [compute_phase_risk_score(**MEDIUM_RISK) for _ in range(100)]
        assert len(set(scores)) == 1

    def test_decision_determinism_100(self) -> None:
        decisions = [route_decoder_from_phase_space(**MEDIUM_RISK) for _ in range(100)]
        hashes = [d.stable_hash for d in decisions]
        assert len(set(hashes)) == 1

    def test_export_determinism_100(self) -> None:
        decision = route_decoder_from_phase_space(**HIGH_RISK)
        exports = [json.dumps(export_phase_steering_bundle(decision),
                              sort_keys=True) for _ in range(100)]
        assert len(set(exports)) == 1

    def test_low_risk_determinism_100(self) -> None:
        decisions = [route_decoder_from_phase_space(**LOW_RISK) for _ in range(100)]
        hashes = [d.stable_hash for d in decisions]
        assert len(set(hashes)) == 1

    def test_critical_determinism_100(self) -> None:
        decisions = [route_decoder_from_phase_space(**CRITICAL_RISK) for _ in range(100)]
        hashes = [d.stable_hash for d in decisions]
        assert len(set(hashes)) == 1


# ---------------------------------------------------------------------------
# Test: stable ordering check
# ---------------------------------------------------------------------------

class TestStableOrdering:
    """decoder_bias tuples must maintain stable ordering."""

    def test_low_risk_bias_is_tuple(self) -> None:
        d = route_decoder_from_phase_space(**LOW_RISK)
        assert isinstance(d.decoder_bias, tuple)

    def test_high_risk_bias_is_tuple(self) -> None:
        d = route_decoder_from_phase_space(**HIGH_RISK)
        assert isinstance(d.decoder_bias, tuple)

    def test_bias_ordering_stable_across_runs(self) -> None:
        biases = [route_decoder_from_phase_space(**MEDIUM_RISK).decoder_bias
                   for _ in range(50)]
        assert all(b == biases[0] for b in biases)

    def test_export_keys_sorted(self) -> None:
        d = route_decoder_from_phase_space(**MEDIUM_RISK)
        bundle = export_phase_steering_bundle(d)
        keys = list(bundle.keys())
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Test: low-risk route
# ---------------------------------------------------------------------------

class TestLowRiskRoute:
    """Low-risk scenario must produce conservative primary route."""

    def test_low_risk_label(self) -> None:
        d = route_decoder_from_phase_space(**LOW_RISK)
        assert d.risk_label == "low"

    def test_low_risk_primary_route(self) -> None:
        d = route_decoder_from_phase_space(**LOW_RISK)
        assert d.recovery_route == ROUTE_PRIMARY

    def test_low_risk_no_escalation(self) -> None:
        d = route_decoder_from_phase_space(**LOW_RISK)
        assert d.escalation_level == ESCALATION_NONE

    def test_low_risk_low_rollback(self) -> None:
        d = route_decoder_from_phase_space(**LOW_RISK)
        assert d.rollback_weight < 0.3

    def test_low_risk_primary_decoder_first(self) -> None:
        d = route_decoder_from_phase_space(**LOW_RISK)
        assert d.decoder_bias[0] == "DECODE_PORTFOLIO_A"


# ---------------------------------------------------------------------------
# Test: high-risk route
# ---------------------------------------------------------------------------

class TestHighRiskRoute:
    """High-risk scenario must produce escalated route."""

    def test_high_risk_label(self) -> None:
        d = route_decoder_from_phase_space(**HIGH_RISK)
        assert d.risk_label in ("high", "critical")

    def test_high_risk_escalation(self) -> None:
        d = route_decoder_from_phase_space(**HIGH_RISK)
        assert d.escalation_level >= ESCALATION_WARNING

    def test_high_risk_rollback_elevated(self) -> None:
        d = route_decoder_from_phase_space(**HIGH_RISK)
        assert d.rollback_weight >= 0.5

    def test_high_risk_not_primary_route(self) -> None:
        d = route_decoder_from_phase_space(**HIGH_RISK)
        assert d.recovery_route != ROUTE_PRIMARY

    def test_critical_emergency_route(self) -> None:
        d = route_decoder_from_phase_space(**CRITICAL_RISK)
        assert d.recovery_route == ROUTE_EMERGENCY

    def test_critical_max_escalation(self) -> None:
        d = route_decoder_from_phase_space(**CRITICAL_RISK)
        assert d.escalation_level == ESCALATION_CRITICAL

    def test_high_negative_mass_drift_escalation(self) -> None:
        """High negative mass + drift → escalation + stronger rollback."""
        d = route_decoder_from_phase_space(
            centroid_q=0.3, centroid_p=0.3,
            negative_mass=0.8, drift_momentum=0.9,
        )
        assert d.escalation_level >= ESCALATION_WARNING
        assert d.rollback_weight > 0.5

    def test_strong_displacement_alternate_bias(self) -> None:
        """Strong displacement → alternate portfolio bias."""
        d = route_decoder_from_phase_space(
            centroid_q=0.9, centroid_p=0.8,
            negative_mass=0.7, drift_momentum=0.6,
        )
        assert "DECODE_PORTFOLIO_A" not in d.decoder_bias[:1]

    def test_medium_drift_recovery_preferred(self) -> None:
        """Medium drift → recovery preferred."""
        d = route_decoder_from_phase_space(
            centroid_q=0.2, centroid_p=0.2,
            negative_mass=0.3, drift_momentum=0.6,
        )
        assert d.recovery_route in (ROUTE_RECOVERY, ROUTE_PRIMARY)


# ---------------------------------------------------------------------------
# Test: export equality across repeated runs
# ---------------------------------------------------------------------------

class TestExportEquality:
    """Export must be byte-identical across repeated runs."""

    def test_export_json_equality(self) -> None:
        d = route_decoder_from_phase_space(**MEDIUM_RISK)
        json1 = json.dumps(export_phase_steering_bundle(d), sort_keys=True)
        json2 = json.dumps(export_phase_steering_bundle(d), sort_keys=True)
        assert json1 == json2

    def test_export_contains_version(self) -> None:
        d = route_decoder_from_phase_space(**LOW_RISK)
        bundle = export_phase_steering_bundle(d)
        assert bundle["version"] == STEERING_VERSION

    def test_export_contains_all_fields(self) -> None:
        d = route_decoder_from_phase_space(**LOW_RISK)
        bundle = export_phase_steering_bundle(d)
        expected_keys = {
            "centroid_p", "centroid_q", "decoder_bias", "drift_momentum",
            "escalation_level", "layer", "negative_mass", "phase_radius",
            "phase_risk_score", "recovery_route", "risk_label",
            "rollback_weight", "stable_hash", "version",
        }
        assert set(bundle.keys()) == expected_keys

    def test_export_hash_matches_decision(self) -> None:
        d = route_decoder_from_phase_space(**HIGH_RISK)
        bundle = export_phase_steering_bundle(d)
        assert bundle["stable_hash"] == d.stable_hash

    def test_export_repeated_equality(self) -> None:
        """10 exports produce identical JSON bytes."""
        d = route_decoder_from_phase_space(**CRITICAL_RISK)
        exports = [json.dumps(export_phase_steering_bundle(d),
                              sort_keys=True, separators=(",", ":"))
                   for _ in range(10)]
        assert len(set(exports)) == 1


# ---------------------------------------------------------------------------
# Test: input non-mutation
# ---------------------------------------------------------------------------

class TestInputNonMutation:
    """Inputs must not be mutated by any function."""

    def test_risk_score_no_mutation(self) -> None:
        inputs = copy.deepcopy(MEDIUM_RISK)
        compute_phase_risk_score(**inputs)
        assert inputs == MEDIUM_RISK

    def test_route_no_mutation(self) -> None:
        inputs = copy.deepcopy(HIGH_RISK)
        route_decoder_from_phase_space(**inputs)
        assert inputs == HIGH_RISK

    def test_export_no_mutation(self) -> None:
        d = route_decoder_from_phase_space(**LOW_RISK)
        original_hash = d.stable_hash
        export_phase_steering_bundle(d)
        assert d.stable_hash == original_hash

    def test_frozen_decision(self) -> None:
        d = route_decoder_from_phase_space(**LOW_RISK)
        with pytest.raises(FrozenInstanceError):
            d.phase_risk_score = 0.99  # type: ignore[misc]

    def test_frozen_ledger(self) -> None:
        ledger = build_steering_ledger()
        with pytest.raises(FrozenInstanceError):
            ledger.decision_count = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: ledger operations
# ---------------------------------------------------------------------------

class TestSteeringLedger:
    """Steering ledger must maintain immutability and determinism."""

    def test_empty_ledger(self) -> None:
        ledger = build_steering_ledger()
        assert ledger.decision_count == 0
        assert ledger.decisions == ()
        assert len(ledger.stable_hash) == 64

    def test_append_decision(self) -> None:
        d = route_decoder_from_phase_space(**LOW_RISK)
        ledger = build_steering_ledger()
        updated = append_steering_decision(d, ledger)
        assert updated.decision_count == 1
        assert updated.decisions[0] is d
        assert updated.stable_hash != ledger.stable_hash

    def test_ledger_determinism(self) -> None:
        d1 = route_decoder_from_phase_space(**LOW_RISK)
        d2 = route_decoder_from_phase_space(**HIGH_RISK)
        ledger_a = build_steering_ledger()
        ledger_a = append_steering_decision(d1, ledger_a)
        ledger_a = append_steering_decision(d2, ledger_a)

        ledger_b = build_steering_ledger()
        ledger_b = append_steering_decision(d1, ledger_b)
        ledger_b = append_steering_decision(d2, ledger_b)

        assert ledger_a.stable_hash == ledger_b.stable_hash

    def test_ledger_immutable(self) -> None:
        ledger = build_steering_ledger()
        with pytest.raises(FrozenInstanceError):
            ledger.decisions = ()  # type: ignore[misc]
