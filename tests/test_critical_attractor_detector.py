"""Tests for v118.0.0 deterministic critical attractor detector layer."""

from __future__ import annotations

from qec.analysis.critical_attractor_detector import (
    classify_attractor_state,
    compare_attractors,
    compute_critical_risk_score,
    compute_cycle_signature,
    detect_basin_lock,
    detect_cycles,
    run_critical_attractor_detector,
)


def test_cycle_detection_simple_repeat() -> None:
    detected, cycle = detect_cycles([1, 2, 3, 1, 2, 3])
    assert detected is True
    assert cycle == [1, 2, 3]


def test_no_cycle_detection() -> None:
    detected, cycle = detect_cycles([1, 2, 3, 4, 5])
    assert detected is False
    assert cycle == []


def test_signature_stability() -> None:
    signature_a = compute_cycle_signature([1, 2, 3])
    signature_b = compute_cycle_signature([1, 2, 3])
    assert signature_a == signature_b


def test_exact_attractor_match() -> None:
    assert compare_attractors([1, 2, 3], [1, 2, 3]) == 1.0


def test_partial_similarity() -> None:
    score = compare_attractors([1, 2, 3], [1, 9, 3])
    assert score == (2.0 / 3.0)


def test_risk_score_bounds() -> None:
    assert 0.0 <= compute_critical_risk_score(0.0, 0, 0.0) <= 1.0
    assert 0.0 <= compute_critical_risk_score(1.0, 10_000, 1.0) <= 1.0


def test_basin_lock_detection() -> None:
    assert detect_basin_lock(2, 0.8) is True
    assert detect_basin_lock(1, 0.99) is False


def test_nominal_classification() -> None:
    assert classify_attractor_state(0.2) == "nominal"


def test_elevated_classification() -> None:
    assert classify_attractor_state(0.5) == "elevated"


def test_critical_classification() -> None:
    assert classify_attractor_state(0.9) == "critical"


def test_deterministic_repeatability() -> None:
    sequence = [1, 2, 1, 2, 1, 2]
    first = run_critical_attractor_detector(sequence, warning_score=0.6, baseline_cycle=[1, 2])
    second = run_critical_attractor_detector(sequence, warning_score=0.6, baseline_cycle=[1, 2])
    assert first == second


def test_empty_sequence_handling() -> None:
    result = run_critical_attractor_detector([], warning_score=0.4)
    assert result["cycle_detected"] is False
    assert result["cycle_length"] == 0
    assert result["cycle_signature"] == ()
    assert result["attractor_similarity_score"] == 0.0
    assert result["critical_risk_score"] == 0.0
    assert result["risk_triggered"] is False


def test_singleton_sequence_handling() -> None:
    result = run_critical_attractor_detector([7], warning_score=0.4)
    assert result["cycle_detected"] is False
    assert result["cycle_length"] == 0
    assert result["cycle_signature"] == ()
    assert result["risk_triggered"] is False


def test_no_cycle_high_warning_non_triggered() -> None:
    result = run_critical_attractor_detector([1, 2, 3, 4], warning_score=1.0)
    assert result["cycle_detected"] is False
    assert result["critical_risk_score"] == 0.0
    assert result["risk_triggered"] is False
