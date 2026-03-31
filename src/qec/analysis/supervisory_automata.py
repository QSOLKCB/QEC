"""v116.0.0 — Deterministic supervisory automata layer.

Formal supervisory tuple:
    Γ = (Q, Λ, q0, Q_safe, Q_unsafe, δ)

where
    Q = {"safe", "warning", "critical"}
    q0 = "safe"
    Q_safe = {"safe"}
    Q_unsafe = {"critical"}

All logic in this module is deterministic, explicit, and inspectable.
"""

from __future__ import annotations

from typing import Any, Mapping

SUPERVISORY_STATES = ("safe", "warning", "critical")
INITIAL_SUPERVISORY_STATE = "safe"
SAFE_SUPERVISORY_STATES = ("safe",)
UNSAFE_SUPERVISORY_STATES = ("critical",)

SAFE_CONFIDENCE_THRESHOLD = 0.80
WARNING_CONFIDENCE_THRESHOLD = 0.55
SEVERE_CONFIDENCE_THRESHOLD = 0.35

SAFE_DRIFT_THRESHOLD = 0.25
WARNING_DRIFT_THRESHOLD = 0.50
CRITICAL_DRIFT_THRESHOLD = 0.75

SAFE_CYCLE_PERIOD_THRESHOLD = 3.0
WARNING_CYCLE_PERIOD_THRESHOLD = 6.0
CRITICAL_CYCLE_PERIOD_THRESHOLD = 9.0

EVENT_STABLE = "stable_conditions"
EVENT_CONFIDENCE_DROP = "confidence_drop"
EVENT_CYCLE_INSTABILITY = "cycle_instability"
EVENT_MANIFOLD_VIOLATION = "manifold_violation"
EVENT_ATTRACTOR_SHIFT = "attractor_shift"
EVENT_DRIFT_INCREASE = "drift_increase"

EVENT_ALPHABET = (
    EVENT_STABLE,
    EVENT_CONFIDENCE_DROP,
    EVENT_CYCLE_INSTABILITY,
    EVENT_MANIFOLD_VIOLATION,
    EVENT_ATTRACTOR_SHIFT,
    EVENT_DRIFT_INCREASE,
)

ESCALATION_NONE = "none"
ESCALATION_OBSERVE_STABILIZE = "observe_stabilize"
ESCALATION_INTERVENE = "intervene"


def transition_supervisory_state(current_state: str, event: str) -> str:
    """Deterministic supervisory transition δ(q, λ) -> q'."""
    if current_state not in SUPERVISORY_STATES:
        raise ValueError(f"unknown supervisory state: {current_state}")
    if event not in EVENT_ALPHABET:
        raise ValueError(f"unknown supervisory event: {event}")

    if event == EVENT_MANIFOLD_VIOLATION:
        return "critical"

    if current_state == "safe":
        if event in (EVENT_CONFIDENCE_DROP, EVENT_CYCLE_INSTABILITY, EVENT_ATTRACTOR_SHIFT, EVENT_DRIFT_INCREASE):
            return "warning"
        return "safe"

    if current_state == "warning":
        if event in (EVENT_CONFIDENCE_DROP, EVENT_CYCLE_INSTABILITY, EVENT_DRIFT_INCREASE):
            return "critical"
        if event == EVENT_STABLE:
            return "safe"
        return "warning"

    # current_state == "critical"
    if event == EVENT_STABLE:
        return "warning"
    if event == EVENT_ATTRACTOR_SHIFT:
        return "warning"
    return "critical"


def classify_supervisory_state(
    confidence: float,
    cycle_period: float,
    drift_score: float,
    manifold_preserved: bool,
) -> str:
    """Deterministically classify: safe / warning / critical."""
    confidence_value = _clamp01(confidence)
    drift_value = _clamp01(drift_score)
    cycle_value = max(0.0, float(cycle_period))

    if (not manifold_preserved) or confidence_value <= SEVERE_CONFIDENCE_THRESHOLD:
        return "critical"
    if drift_value >= CRITICAL_DRIFT_THRESHOLD or cycle_value >= CRITICAL_CYCLE_PERIOD_THRESHOLD:
        return "critical"

    if confidence_value >= SAFE_CONFIDENCE_THRESHOLD and drift_value <= SAFE_DRIFT_THRESHOLD and cycle_value <= SAFE_CYCLE_PERIOD_THRESHOLD:
        return "safe"

    return "warning"


def map_escalation_action(state: str) -> str:
    """Monotonic deterministic mapping from state to escalation action."""
    if state == "safe":
        return ESCALATION_NONE
    if state == "warning":
        return ESCALATION_OBSERVE_STABILIZE
    if state == "critical":
        return ESCALATION_INTERVENE
    raise ValueError(f"unknown supervisory state for escalation: {state}")


def run_supervisory_automata(
    engine_metrics: Mapping[str, Any],
    previous_state: str = INITIAL_SUPERVISORY_STATE,
) -> dict[str, Any]:
    """Main supervisory tuple execution layer."""
    previous = _normalize_state(previous_state)

    confidence = _clamp01(float(engine_metrics.get("confidence", 0.0)))
    cycle_period = max(0.0, float(engine_metrics.get("cycle_period", 0.0)))
    drift_score = _clamp01(float(engine_metrics.get("drift_score", 0.0)))
    manifold_preserved = bool(engine_metrics.get("manifold_preserved", True))

    classified_state = classify_supervisory_state(
        confidence=confidence,
        cycle_period=cycle_period,
        drift_score=drift_score,
        manifold_preserved=manifold_preserved,
    )
    transition_event = _derive_transition_event(
        confidence=confidence,
        cycle_period=cycle_period,
        drift_score=drift_score,
        manifold_preserved=manifold_preserved,
        classified_state=classified_state,
    )
    next_state = transition_supervisory_state(previous, transition_event)

    if classified_state == "critical" and next_state != "critical":
        next_state = "critical"

    escalation_action = map_escalation_action(next_state)
    escalation_level = _escalation_level(next_state)

    return {
        "supervisory_state": classified_state,
        "previous_supervisory_state": previous,
        "transition_event": _transition_semantic(previous, next_state),
        "next_state": next_state,
        "safety_manifold_preserved": manifold_preserved,
        "escalation_level": escalation_level,
        "escalation_action": escalation_action,
        "supervisory_confidence": confidence,
        "intervention_required": next_state in UNSAFE_SUPERVISORY_STATES,
    }


def _derive_transition_event(
    *,
    confidence: float,
    cycle_period: float,
    drift_score: float,
    manifold_preserved: bool,
    classified_state: str,
) -> str:
    if not manifold_preserved:
        return EVENT_MANIFOLD_VIOLATION
    if classified_state == "safe":
        return EVENT_STABLE
    if confidence < WARNING_CONFIDENCE_THRESHOLD:
        return EVENT_CONFIDENCE_DROP
    if cycle_period >= WARNING_CYCLE_PERIOD_THRESHOLD:
        return EVENT_CYCLE_INSTABILITY
    if drift_score >= WARNING_DRIFT_THRESHOLD:
        return EVENT_DRIFT_INCREASE
    return EVENT_ATTRACTOR_SHIFT


def _transition_semantic(previous_state: str, next_state: str) -> str:
    if previous_state == "safe" and next_state == "safe":
        return "remain_safe"
    if previous_state == "safe" and next_state == "warning":
        return "safe_to_warning"
    if previous_state == "safe" and next_state == "critical":
        return "safe_to_critical"
    if previous_state == "warning" and next_state == "critical":
        return "warning_to_critical"
    if previous_state == "critical" and next_state == "warning":
        return "critical_to_warning"
    if previous_state == "warning" and next_state == "safe":
        return "warning_to_safe"
    if previous_state == "critical" and next_state == "critical":
        return "remain_critical"
    if previous_state == "critical" and next_state == "safe":
        return "critical_to_safe"
    if previous_state == "warning" and next_state == "warning":
        return "remain_warning"
    raise ValueError(f"unsupported supervisory transition: ({previous_state}, {next_state})")


def _normalize_state(state: str) -> str:
    normalized = str(state)
    if normalized not in SUPERVISORY_STATES:
        raise ValueError(f"unknown supervisory state: {state}")
    return normalized


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _escalation_level(state: str) -> int:
    if state == "safe":
        return 0
    if state == "warning":
        return 1
    if state == "critical":
        return 2
    raise ValueError(f"unknown supervisory state for escalation level: {state}")


__all__ = [
    "CRITICAL_DRIFT_THRESHOLD",
    "SAFE_CONFIDENCE_THRESHOLD",
    "WARNING_CONFIDENCE_THRESHOLD",
    "classify_supervisory_state",
    "map_escalation_action",
    "run_supervisory_automata",
    "transition_supervisory_state",
]
