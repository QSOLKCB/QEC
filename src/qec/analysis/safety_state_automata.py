"""v124.0.0 — Deterministic safety-state automata for fail-safe transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

SAFETY_STATES = (
    "nominal",
    "alert",
    "recovering",
    "stabilized",
    "emergency_stop",
)

TRANSITION_LABELS = {
    ("nominal", "nominal"): "remain_nominal",
    ("nominal", "alert"): "nominal_to_alert",
    ("alert", "alert"): "remain_alert",
    ("alert", "recovering"): "alert_to_recovering",
    ("alert", "emergency_stop"): "alert_to_emergency_stop",
    ("recovering", "recovering"): "remain_recovering",
    ("recovering", "stabilized"): "recovering_to_stabilized",
    ("recovering", "emergency_stop"): "recovering_to_emergency_stop",
    ("stabilized", "stabilized"): "remain_stabilized",
    ("stabilized", "nominal"): "stabilized_to_nominal",
    ("emergency_stop", "emergency_stop"): "remain_emergency_stop",
}


@dataclass(frozen=True)
class SafetyState:
    current_state: str
    entry_count: int
    invariant_ok: bool
    recovery_active: bool
    emergency_stop: bool


class SafetyStateAutomata:
    """Deterministic explicit FSM for safety transitions."""

    def __init__(self) -> None:
        self._warning_threshold = 0.5

    def validate_transition(self, from_state: str, to_state: str, state_data: dict) -> bool:
        """Return whether transition satisfies explicit deterministic safety constraints."""
        if from_state not in SAFETY_STATES or to_state not in SAFETY_STATES:
            return False

        if from_state == "emergency_stop" and to_state != "emergency_stop":
            return False

        invariant_ok = _parse_bool(state_data.get("invariant_ok", True))
        recovery_active = _parse_bool(state_data.get("recovery_active", False))
        emergency_stop = _parse_bool(state_data.get("emergency_stop", False))

        if to_state == "stabilized" and not invariant_ok:
            return False
        if to_state == "recovering" and not recovery_active:
            return False
        if to_state == "nominal" and emergency_stop:
            return False

        return True

    def process_safety_state(self, current_state: str, state_data: dict) -> str:
        """Determine next state using deterministic guarded transition ordering."""
        if current_state not in SAFETY_STATES:
            raise ValueError(f"unknown safety state: {current_state}")

        invariant_ok = _parse_bool(state_data.get("invariant_ok", True))
        recovery_active = _parse_bool(state_data.get("recovery_active", False))
        emergency_stop = _parse_bool(state_data.get("emergency_stop", False))
        warning_score = _safe_float(state_data.get("warning_score", 0.0), default=0.0)

        if current_state == "emergency_stop":
            return "emergency_stop"

        if emergency_stop:
            return "emergency_stop"

        if current_state == "nominal":
            if (not invariant_ok) or warning_score >= self._warning_threshold:
                return "alert"
            return "nominal"

        if current_state == "alert":
            if recovery_active and not emergency_stop:
                return "recovering"
            return "alert"

        if current_state == "recovering":
            if invariant_ok and not emergency_stop:
                return "stabilized"
            return "recovering"

        if current_state == "stabilized":
            if invariant_ok and not recovery_active:
                return "nominal"
            return "stabilized"

        return current_state

    def step(self, state: SafetyState, state_data: dict) -> dict:
        """Compute one deterministic FSM step and transition metadata."""
        merged_state: dict[str, Any] = {
            "invariant_ok": state.invariant_ok,
            "recovery_active": state.recovery_active,
            "emergency_stop": state.emergency_stop,
            "warning_score": 0.0,
        }
        merged_state.update(state_data)

        previous_state = state.current_state
        next_state = self.process_safety_state(previous_state, merged_state)
        transition_valid = self.validate_transition(previous_state, next_state, merged_state)
        safety_transition = TRANSITION_LABELS.get((previous_state, next_state), f"{previous_state}_to_{next_state}")

        return {
            "previous_state": previous_state,
            "next_state": next_state,
            "transition_valid": transition_valid,
            "safety_transition": safety_transition,
            "fail_safe_triggered": next_state == "emergency_stop",
        }


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true"):
            return True
        if normalized in ("0", "false"):
            return False
    return bool(value)


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
