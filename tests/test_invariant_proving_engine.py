"""Tests for invariant_proving_engine.py — v120.0.0."""

from __future__ import annotations

from qec.analysis.invariant_proving_engine import (
    INVARIANT_CONFIDENCE_CEILING,
    INVARIANT_CONFIDENCE_FLOOR,
    MAX_PROOF_DEPTH,
    compute_proof_confidence,
    run_invariant_proving_engine,
)


def _valid_snapshot() -> dict:
    return {
        "warning_score": 0.2,
        "risk_score": 0.4,
        "timer_state": {
            "critical_cycles": 12,
            "observer_cycles": 3,
        },
        "controller_state": "timed_control",
        "supervisory_state": "stable",
    }


def test_all_invariants_valid():
    result = run_invariant_proving_engine(_valid_snapshot())
    assert result["proof_status"] == "valid"
    assert result["violations_detected"] == 0
    assert result["violation_triggered"] is False


def test_warning_score_violation():
    snapshot = _valid_snapshot()
    snapshot["warning_score"] = 1.2
    result = run_invariant_proving_engine(snapshot)
    assert result["proof_status"] == "partial"
    assert result["violations_detected"] == 1
    assert result["counterexample_trace"][0][0] == "bounded_warning_score"


def test_risk_score_violation():
    snapshot = _valid_snapshot()
    snapshot["risk_score"] = -0.1
    result = run_invariant_proving_engine(snapshot)
    assert result["violations_detected"] == 1
    assert result["counterexample_trace"][0][0] == "bounded_risk_score"


def test_timer_overflow_violation():
    snapshot = _valid_snapshot()
    snapshot["timer_state"] = {"critical_cycles": 33}
    result = run_invariant_proving_engine(snapshot)
    assert result["violations_detected"] == 1
    assert result["counterexample_trace"] == [
        ("bounded_timer_values", "critical_cycles", 33),
    ]


def test_invalid_state_violation():
    snapshot = _valid_snapshot()
    snapshot["controller_state"] = "unknown_state"
    result = run_invariant_proving_engine(snapshot)
    assert result["violations_detected"] == 1
    assert result["counterexample_trace"][0][0] == "valid_controller_state"


def test_proof_confidence_bounds():
    assert compute_proof_confidence(0, 0) == INVARIANT_CONFIDENCE_FLOOR
    assert INVARIANT_CONFIDENCE_FLOOR <= compute_proof_confidence(3, 5) <= INVARIANT_CONFIDENCE_CEILING
    assert compute_proof_confidence(10, 5) == INVARIANT_CONFIDENCE_CEILING


def test_proof_depth_bounds():
    result = run_invariant_proving_engine(_valid_snapshot())
    assert 0 <= result["proof_depth"] <= MAX_PROOF_DEPTH


def test_counterexample_trace_stability():
    snapshot = _valid_snapshot()
    snapshot["warning_score"] = 2.0
    snapshot["risk_score"] = -1.0
    result = run_invariant_proving_engine(snapshot)
    assert result["counterexample_trace"] == [
        ("bounded_risk_score", "risk_score", -1.0),
        ("bounded_warning_score", "warning_score", 2.0),
    ]


def test_deterministic_repeatability():
    snapshot = _valid_snapshot()
    first = run_invariant_proving_engine(snapshot)
    second = run_invariant_proving_engine(snapshot)
    assert first == second


def test_empty_snapshot_handling():
    result = run_invariant_proving_engine({})
    assert result["invariants_checked"] == 5
    assert result["proof_status"] == "violated"
    assert result["violation_triggered"] is True
