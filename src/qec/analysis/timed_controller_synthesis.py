"""v119.0.0 — Deterministic timed controller synthesis layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WARNING_PERSISTENCE_THRESHOLD = 3
CRITICAL_PERSISTENCE_THRESHOLD = 2
RECOVERY_PERSISTENCE_THRESHOLD = 2
MAX_TIMER_VALUE = 32

CONTROLLER_STATES = ("observe", "stabilize", "intervene", "recover")
CONTROL_ACTIONS = ("observe", "stabilize", "intervene", "recover")

TIMED_TRANSITIONS: dict[tuple[str, str], tuple[str, str]] = {
    ("observe", "observe"): ("observe", "remain_observe"),
    ("observe", "stabilize"): ("stabilize", "observe_to_stabilize"),
    ("observe", "intervene"): ("intervene", "observe_to_intervene"),
    ("observe", "recover"): ("recover", "observe_to_recover"),
    ("stabilize", "observe"): ("observe", "stabilize_to_observe"),
    ("stabilize", "stabilize"): ("stabilize", "remain_stabilize"),
    ("stabilize", "intervene"): ("intervene", "stabilize_to_intervene"),
    ("stabilize", "recover"): ("recover", "stabilize_to_recover"),
    ("intervene", "observe"): ("observe", "intervene_to_observe"),
    ("intervene", "stabilize"): ("stabilize", "intervene_to_stabilize"),
    ("intervene", "intervene"): ("intervene", "remain_intervene"),
    ("intervene", "recover"): ("recover", "intervene_to_recover"),
    ("recover", "observe"): ("observe", "recover_to_observe"),
    ("recover", "stabilize"): ("stabilize", "recover_to_stabilize"),
    ("recover", "intervene"): ("intervene", "recover_to_intervene"),
    ("recover", "recover"): ("recover", "remain_recover"),
}


@dataclass
class TimedControllerState:
    """Deterministic bounded cycle counters for timed controller synthesis."""

    warning_cycles: int = 0
    critical_cycles: int = 0
    recovery_cycles: int = 0
    fsm_state: str = "observe"
    was_escalated: bool = False

    def reset(self) -> None:
        self.warning_cycles = 0
        self.critical_cycles = 0
        self.recovery_cycles = 0
        self.fsm_state = "observe"
        self.was_escalated = False

    def update(self, observer_state: Any, attractor_state: Any) -> None:
        warning_active = _is_warning(observer_state, attractor_state)
        critical_active = _is_critical(observer_state, attractor_state)
        recovery_active = _is_recovery(observer_state, attractor_state)

        if warning_active or critical_active:
            self.was_escalated = True

        if warning_active and not critical_active:
            self.warning_cycles = _bounded_increment(self.warning_cycles)
        else:
            self.warning_cycles = 0

        if critical_active:
            self.critical_cycles = _bounded_increment(self.critical_cycles)
        else:
            self.critical_cycles = 0

        if self.was_escalated and recovery_active and not warning_active and not critical_active:
            self.recovery_cycles = _bounded_increment(self.recovery_cycles)
        else:
            self.recovery_cycles = 0


def evaluate_timer_guards(state: TimedControllerState) -> dict[str, bool]:
    """Evaluate persistence thresholds deterministically."""
    return {
        "warning_guard": int(state.warning_cycles) >= WARNING_PERSISTENCE_THRESHOLD,
        "critical_guard": int(state.critical_cycles) >= CRITICAL_PERSISTENCE_THRESHOLD,
        "recovery_guard": int(state.recovery_cycles) >= RECOVERY_PERSISTENCE_THRESHOLD,
    }


def synthesize_control_action(
    observer_state: Any,
    attractor_state: Any,
    guards: dict[str, bool],
) -> str:
    """Synthesize deterministic control action: none/observe/stabilize/intervene/recover."""
    _ = (observer_state, attractor_state)
    if bool(guards.get("critical_guard", False)):
        return "intervene"
    if bool(guards.get("warning_guard", False)):
        return "stabilize"
    if bool(guards.get("recovery_guard", False)):
        return "recover"
    return "observe"


def run_timed_controller_synthesis(
    observer_state: Any,
    attractor_state: Any,
    controller_state: TimedControllerState | None = None,
) -> dict[str, Any]:
    """Main deterministic timed controller layer."""
    state = controller_state if controller_state is not None else TimedControllerState()
    if state.fsm_state not in CONTROLLER_STATES:
        raise ValueError(f"unknown controller state: {state.fsm_state}")

    state.update(observer_state, attractor_state)
    guards = evaluate_timer_guards(state)
    control_action = synthesize_control_action(observer_state, attractor_state, guards)

    action_key = control_action
    if action_key not in CONTROL_ACTIONS:
        raise ValueError(f"unknown control action: {control_action}")

    transition_key = (state.fsm_state, action_key)
    if transition_key not in TIMED_TRANSITIONS:
        raise ValueError(f"unsupported timed transition: {transition_key}")

    next_state, transition_event = TIMED_TRANSITIONS[transition_key]
    state.fsm_state = next_state
    if control_action in {"stabilize", "intervene"}:
        state.was_escalated = True
    if control_action == "recover":
        state.was_escalated = False
    if transition_event == "recover_to_observe":
        state.was_escalated = False

    timer_guard_triggered = bool(guards["warning_guard"] or guards["critical_guard"] or guards["recovery_guard"])

    return {
        "controller_state": next_state,
        "elapsed_warning_cycles": int(_bound_counter(state.warning_cycles)),
        "elapsed_critical_cycles": int(_bound_counter(state.critical_cycles)),
        "elapsed_recovery_cycles": int(_bound_counter(state.recovery_cycles)),
        "timer_guard_triggered": timer_guard_triggered,
        "control_action": control_action,
        "timed_transition_event": transition_event,
        "escalation_required": bool(control_action == "intervene"),
    }


def _is_warning(observer_state: Any, attractor_state: Any) -> bool:
    observer = str(observer_state)
    attractor = str(attractor_state)
    return observer == "warning" or attractor == "elevated"


def _is_critical(observer_state: Any, attractor_state: Any) -> bool:
    observer = str(observer_state)
    attractor = str(attractor_state)
    return observer == "critical" or attractor == "critical"


def _is_recovery(observer_state: Any, attractor_state: Any) -> bool:
    observer = str(observer_state)
    attractor = str(attractor_state)
    return observer == "safe" and attractor in {"nominal", "stable", "safe"}


def _bounded_increment(value: int) -> int:
    return _bound_counter(int(value) + 1)


def _bound_counter(value: int) -> int:
    return max(0, min(MAX_TIMER_VALUE, int(value)))


__all__ = [
    "WARNING_PERSISTENCE_THRESHOLD",
    "CRITICAL_PERSISTENCE_THRESHOLD",
    "RECOVERY_PERSISTENCE_THRESHOLD",
    "MAX_TIMER_VALUE",
    "TIMED_TRANSITIONS",
    "TimedControllerState",
    "evaluate_timer_guards",
    "synthesize_control_action",
    "run_timed_controller_synthesis",
]
