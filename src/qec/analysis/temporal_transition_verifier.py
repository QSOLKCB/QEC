"""v132.0.0 — Temporal transition verifier."""

from __future__ import annotations

from dataclasses import dataclass


MODE_NORMAL = "normal"
MODE_RECOVERY = "recovery"
MODE_SAFE_MODE = "safe_mode"
MODE_ESCALATION_LOCK = "escalation_lock"


LEGAL_TRANSITIONS = {
    (MODE_NORMAL, MODE_RECOVERY),
    (MODE_RECOVERY, MODE_NORMAL),
    (MODE_RECOVERY, MODE_SAFE_MODE),
    (MODE_RECOVERY, MODE_ESCALATION_LOCK),
    (MODE_SAFE_MODE, MODE_SAFE_MODE),
    (MODE_ESCALATION_LOCK, MODE_ESCALATION_LOCK),
}


@dataclass(frozen=True)
class TransitionHistory:
    """Immutable temporal transition history for deterministic verification."""

    transitions: tuple[str, ...]


def _detect_oscillation(transitions: tuple[str, ...]) -> bool:
    """Detect deterministic two-state alternation for runs of at least 4 transitions."""
    if len(transitions) < 5:
        return False

    for start in range(0, len(transitions) - 4):
        first = transitions[start]
        second = transitions[start + 1]
        if first == second:
            continue

        run_length = 2
        expected = first
        for index in range(start + 2, len(transitions)):
            if transitions[index] != expected:
                break
            run_length += 1
            expected = second if expected == first else first

        if run_length >= 5:
            return True

    return False


def validate_transition_sequence(
    history: TransitionHistory,
) -> dict:
    """Validate transition legality, absorbing modes, and oscillation risk."""
    transitions = history.transitions

    illegal_transition_detected = False
    safe_mode_violation = False

    saw_safe_mode = False
    for state in transitions:
        if saw_safe_mode and state != MODE_SAFE_MODE:
            safe_mode_violation = True
            illegal_transition_detected = True
            break
        if state == MODE_SAFE_MODE:
            saw_safe_mode = True

    if not illegal_transition_detected:
        for index in range(len(transitions) - 1):
            edge = (transitions[index], transitions[index + 1])
            if edge not in LEGAL_TRANSITIONS:
                illegal_transition_detected = True
                break

    oscillation_detected = _detect_oscillation(transitions)
    sequence_valid = not illegal_transition_detected and not safe_mode_violation

    if illegal_transition_detected or safe_mode_violation:
        verification_score = 1.0
        verification_label = "critical"
    elif oscillation_detected:
        verification_score = 0.5
        verification_label = "warning"
    else:
        verification_score = 0.0
        verification_label = "safe"

    return {
        "sequence_valid": sequence_valid,
        "illegal_transition_detected": illegal_transition_detected,
        "safe_mode_violation": safe_mode_violation,
        "oscillation_detected": oscillation_detected,
        "verification_score": verification_score,
        "verification_label": verification_label,
    }


def run_temporal_transition_verifier(
    history: TransitionHistory,
) -> dict:
    """Run deterministic temporal transition verification with stable schema."""
    verification = validate_transition_sequence(history=history)
    return {
        "history": history,
        "verification": verification,
        "temporal_ready": True,
    }


__all__ = [
    "TransitionHistory",
    "validate_transition_sequence",
    "run_temporal_transition_verifier",
]
