"""Deterministic tests for v116.0.0 supervisory automata layer."""

from __future__ import annotations

from qec.analysis.supervisory_automata import (
    classify_supervisory_state,
    map_escalation_action,
    run_supervisory_automata,
    transition_supervisory_state,
)


def test_safe_classification() -> None:
    state = classify_supervisory_state(
        confidence=0.95,
        cycle_period=2.0,
        drift_score=0.1,
        manifold_preserved=True,
    )
    assert state == "safe"


def test_warning_classification() -> None:
    state = classify_supervisory_state(
        confidence=0.70,
        cycle_period=4.0,
        drift_score=0.35,
        manifold_preserved=True,
    )
    assert state == "warning"


def test_critical_classification() -> None:
    state = classify_supervisory_state(
        confidence=0.20,
        cycle_period=2.0,
        drift_score=0.20,
        manifold_preserved=True,
    )
    assert state == "critical"


def test_transition_correctness() -> None:
    assert transition_supervisory_state("safe", "stable_conditions") == "safe"
    assert transition_supervisory_state("safe", "confidence_drop") == "warning"
    assert transition_supervisory_state("warning", "drift_increase") == "critical"
    assert transition_supervisory_state("critical", "stable_conditions") == "warning"


def test_escalation_monotonicity() -> None:
    safe_action = map_escalation_action("safe")
    warning_action = map_escalation_action("warning")
    critical_action = map_escalation_action("critical")

    assert safe_action == "none"
    assert warning_action == "observe_stabilize"
    assert critical_action == "intervene"


def test_manifold_violation_forces_critical() -> None:
    output = run_supervisory_automata(
        {
            "confidence": 0.95,
            "cycle_period": 2.0,
            "drift_score": 0.1,
            "manifold_preserved": False,
        },
        previous_state="safe",
    )
    assert output["supervisory_state"] == "critical"
    assert output["safety_manifold_preserved"] is False
    assert output["intervention_required"] is True


def test_deterministic_identical_inputs() -> None:
    metrics = {
        "confidence": 0.63,
        "cycle_period": 6.3,
        "drift_score": 0.52,
        "manifold_preserved": True,
    }
    out1 = run_supervisory_automata(metrics, previous_state="warning")
    out2 = run_supervisory_automata(metrics, previous_state="warning")
    assert out1 == out2


def test_supervisory_confidence_bounds() -> None:
    high = run_supervisory_automata(
        {
            "confidence": 3.0,
            "cycle_period": 1.0,
            "drift_score": 0.1,
            "manifold_preserved": True,
        }
    )
    low = run_supervisory_automata(
        {
            "confidence": -2.0,
            "cycle_period": 1.0,
            "drift_score": 0.1,
            "manifold_preserved": True,
        }
    )

    assert high["supervisory_confidence"] == 1.0
    assert low["supervisory_confidence"] == 0.0


def test_string_false_parses_to_false() -> None:
    output = run_supervisory_automata(
        {
            "confidence": 0.95,
            "cycle_period": 2.0,
            "drift_score": 0.1,
            "manifold_preserved": "false",
        },
        previous_state="safe",
    )
    assert output["safety_manifold_preserved"] is False
    assert output["supervisory_state"] == "critical"


def test_nan_confidence_uses_default() -> None:
    output = run_supervisory_automata(
        {
            "confidence": float("nan"),
            "cycle_period": 2.0,
            "drift_score": 0.1,
            "manifold_preserved": True,
        },
        previous_state="safe",
    )
    assert output["supervisory_confidence"] == 0.0


def test_nan_drift_uses_default() -> None:
    output = run_supervisory_automata(
        {
            "confidence": 0.9,
            "cycle_period": 2.0,
            "drift_score": float("nan"),
            "manifold_preserved": True,
        },
        previous_state="safe",
    )
    assert output["classified_state"] == "critical"
    assert output["supervisory_state"] == "critical"


def test_supervisory_state_is_post_transition_state() -> None:
    output = run_supervisory_automata(
        {
            "confidence": 0.4,
            "cycle_period": 2.0,
            "drift_score": 0.2,
            "manifold_preserved": True,
        },
        previous_state="safe",
    )
    assert output["classified_state"] == "warning"
    assert output["supervisory_state"] == "warning"
    assert output["transition_event"] == "safe_to_warning"


def test_classified_state_can_differ_from_supervisory_state() -> None:
    output = run_supervisory_automata(
        {
            "confidence": 0.9,
            "cycle_period": 2.0,
            "drift_score": 0.1,
            "manifold_preserved": True,
        },
        previous_state="critical",
    )
    assert output["classified_state"] == "safe"
    assert output["supervisory_state"] == "warning"
    assert output["transition_event"] == "critical_to_warning"


def test_hysteresis_is_preserved() -> None:
    output = run_supervisory_automata(
        {
            "confidence": 0.9,
            "cycle_period": 2.0,
            "drift_score": 0.1,
            "manifold_preserved": True,
        },
        previous_state="warning",
    )
    assert output["classified_state"] == "safe"
    assert output["supervisory_state"] == "safe"
    assert output["transition_event"] == "warning_to_safe"
