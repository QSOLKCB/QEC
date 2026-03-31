"""v131.0.0 — Adaptive supervisory state machine engine."""

from __future__ import annotations

import math
from dataclasses import dataclass


MODE_NORMAL = "normal"
MODE_RECOVERY = "recovery"
MODE_SAFE_MODE = "safe_mode"
MODE_ESCALATION_LOCK = "escalation_lock"


@dataclass(frozen=True)
class SupervisorState:
    """Immutable supervisory state for deterministic transition evaluation."""

    current_mode: str
    recovery_attempts: int
    safe_mode_latched: bool


def normalize_supervisory_score(value: float) -> float:
    """Normalize supervisory scores into the hardened [0.0, 1.0] interval."""
    if math.isnan(value):
        return 0.0
    if math.isinf(value):
        return 1.0 if value > 0.0 else 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def evaluate_supervisory_transition(
    state: SupervisorState,
    fault_detected: bool,
    recovery_success: bool,
    risk_score: float,
) -> str:
    """Evaluate deterministic mode transition from the current supervisory mode."""
    normalized_risk = normalize_supervisory_score(risk_score)

    if state.safe_mode_latched:
        return MODE_SAFE_MODE

    if state.current_mode == MODE_SAFE_MODE:
        return MODE_SAFE_MODE

    if state.current_mode == MODE_ESCALATION_LOCK:
        return MODE_ESCALATION_LOCK

    if state.current_mode == MODE_NORMAL:
        if fault_detected:
            return MODE_RECOVERY
        return MODE_NORMAL

    if state.current_mode == MODE_RECOVERY:
        if recovery_success:
            return MODE_NORMAL
        if normalized_risk >= 0.95:
            return MODE_SAFE_MODE
        if state.recovery_attempts >= 3:
            return MODE_ESCALATION_LOCK
        return MODE_RECOVERY

    return MODE_NORMAL


def run_adaptive_supervisory_controller(
    state: SupervisorState,
    fault_detected: bool,
    recovery_success: bool,
    risk_score: float,
) -> dict:
    """Run deterministic supervisory transition evaluation with a stable schema."""
    next_mode = evaluate_supervisory_transition(
        state=state,
        fault_detected=fault_detected,
        recovery_success=recovery_success,
        risk_score=risk_score,
    )
    previous_mode = state.current_mode
    return {
        "previous_mode": previous_mode,
        "next_mode": next_mode,
        "mode_transition": f"{previous_mode}->{next_mode}",
        "fail_safe_triggered": next_mode == MODE_SAFE_MODE,
        "supervisory_ready": True,
    }


__all__ = [
    "MODE_NORMAL",
    "MODE_RECOVERY",
    "MODE_SAFE_MODE",
    "MODE_ESCALATION_LOCK",
    "SupervisorState",
    "normalize_supervisory_score",
    "evaluate_supervisory_transition",
    "run_adaptive_supervisory_controller",
]
