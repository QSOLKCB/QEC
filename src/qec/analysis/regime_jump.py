"""Regime jump and parity coherence analysis layer.

Detects control-state transitions and short-term coherence
from adaptive control / control memory outputs.
"""


def compute_regime_jump(previous_state: str, current_state: str) -> dict:
    """Detect a regime jump between two consecutive states."""
    if previous_state == current_state:
        return {"jump_detected": False, "jump_type": "none"}
    return {
        "jump_detected": True,
        "jump_type": f"{previous_state}_to_{current_state}",
    }


def compute_coherence_length(history: list) -> int:
    """Return count of consecutive identical states from end of history."""
    if not history:
        return 0
    last = history[-1]
    count = 0
    for state in reversed(history):
        if state != last:
            break
        count += 1
    return count


def classify_regime_behavior(jump_detected: bool, coherence_length: int) -> str:
    """Classify regime behavior as transition, locked, or oscillatory."""
    if jump_detected:
        return "transition"
    if coherence_length >= 3:
        return "locked"
    return "oscillatory"


def run_regime_jump_analysis(
    previous_state: str, current_state: str, history: list
) -> dict:
    """Compose regime jump detection, coherence, and classification."""
    jump = compute_regime_jump(previous_state, current_state)
    coherence = compute_coherence_length(history)
    behavior = classify_regime_behavior(jump["jump_detected"], coherence)
    return {
        "jump_detected": jump["jump_detected"],
        "jump_type": jump["jump_type"],
        "coherence_length": coherence,
        "regime_behavior": behavior,
    }
