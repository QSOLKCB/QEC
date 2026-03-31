"""Deterministic tests for v119.0.0 timed controller synthesis."""

from __future__ import annotations

from qec.analysis.timed_controller_synthesis import (
    CONTROL_ACTIONS,
    CRITICAL_PERSISTENCE_THRESHOLD,
    MAX_TIMER_VALUE,
    RECOVERY_PERSISTENCE_THRESHOLD,
    TIMED_TRANSITIONS,
    WARNING_PERSISTENCE_THRESHOLD,
    TimedControllerState,
    evaluate_timer_guards,
    run_timed_controller_synthesis,
    synthesize_control_action,
)


def test_warning_persistence_threshold() -> None:
    state = TimedControllerState()

    for _ in range(WARNING_PERSISTENCE_THRESHOLD - 1):
        state.update("warning", "elevated")
    guards_before = evaluate_timer_guards(state)
    assert guards_before["warning_guard"] is False

    state.update("warning", "elevated")
    guards_after = evaluate_timer_guards(state)
    assert guards_after["warning_guard"] is True


def test_critical_persistence_threshold() -> None:
    state = TimedControllerState()

    for _ in range(CRITICAL_PERSISTENCE_THRESHOLD - 1):
        state.update("critical", "critical")
    guards_before = evaluate_timer_guards(state)
    assert guards_before["critical_guard"] is False

    state.update("critical", "critical")
    guards_after = evaluate_timer_guards(state)
    assert guards_after["critical_guard"] is True


def test_recovery_persistence_threshold() -> None:
    state = TimedControllerState()
    state.was_escalated = True

    for _ in range(RECOVERY_PERSISTENCE_THRESHOLD - 1):
        state.update("safe", "nominal")
    guards_before = evaluate_timer_guards(state)
    assert guards_before["recovery_guard"] is False

    state.update("safe", "nominal")
    guards_after = evaluate_timer_guards(state)
    assert guards_after["recovery_guard"] is True


def test_timer_cap_behavior() -> None:
    state = TimedControllerState()

    for _ in range(MAX_TIMER_VALUE + 10):
        state.update("critical", "critical")

    assert state.critical_cycles == MAX_TIMER_VALUE


def test_reset_semantics() -> None:
    state = TimedControllerState()
    state.update("warning", "elevated")
    state.update("critical", "critical")
    state.update("safe", "nominal")

    state.reset()

    assert state.warning_cycles == 0
    assert state.critical_cycles == 0
    assert state.recovery_cycles == 0
    assert state.fsm_state == "observe"
    assert state.was_escalated is False


def test_control_action_synthesis() -> None:
    assert (
        synthesize_control_action(
            "safe",
            "nominal",
            {"warning_guard": False, "critical_guard": True, "recovery_guard": False},
        )
        == "intervene"
    )
    assert (
        synthesize_control_action(
            "warning",
            "elevated",
            {"warning_guard": True, "critical_guard": False, "recovery_guard": False},
        )
        == "stabilize"
    )
    assert (
        synthesize_control_action(
            "safe",
            "nominal",
            {"warning_guard": False, "critical_guard": False, "recovery_guard": True},
        )
        == "recover"
    )
    assert (
        synthesize_control_action(
            "safe",
            "nominal",
            {"warning_guard": False, "critical_guard": False, "recovery_guard": False},
        )
        == "observe"
    )


def test_transition_map_correctness() -> None:
    assert TIMED_TRANSITIONS[("observe", "observe")] == ("observe", "remain_observe")
    assert TIMED_TRANSITIONS[("observe", "stabilize")] == ("stabilize", "observe_to_stabilize")
    assert TIMED_TRANSITIONS[("stabilize", "intervene")] == ("intervene", "stabilize_to_intervene")
    assert TIMED_TRANSITIONS[("intervene", "recover")] == ("recover", "intervene_to_recover")
    assert TIMED_TRANSITIONS[("recover", "observe")] == ("observe", "recover_to_observe")


def test_deterministic_repeatability() -> None:
    state_a = TimedControllerState()
    state_b = TimedControllerState()

    sequence = [
        ("warning", "elevated"),
        ("warning", "elevated"),
        ("warning", "elevated"),
        ("critical", "critical"),
        ("critical", "critical"),
        ("safe", "nominal"),
        ("safe", "nominal"),
    ]

    out_a = [run_timed_controller_synthesis(obs, att, controller_state=state_a) for obs, att in sequence]
    out_b = [run_timed_controller_synthesis(obs, att, controller_state=state_b) for obs, att in sequence]

    assert out_a == out_b


def test_bounded_counters() -> None:
    state = TimedControllerState()

    for _ in range(MAX_TIMER_VALUE + 100):
        out = run_timed_controller_synthesis("critical", "critical", controller_state=state)

    assert isinstance(out["elapsed_warning_cycles"], int)
    assert isinstance(out["elapsed_critical_cycles"], int)
    assert isinstance(out["elapsed_recovery_cycles"], int)
    assert 0 <= out["elapsed_warning_cycles"] <= MAX_TIMER_VALUE
    assert 0 <= out["elapsed_critical_cycles"] <= MAX_TIMER_VALUE
    assert 0 <= out["elapsed_recovery_cycles"] <= MAX_TIMER_VALUE


def test_empty_default_state_behavior() -> None:
    out = run_timed_controller_synthesis("safe", "nominal")

    assert out["controller_state"] == "observe"
    assert out["elapsed_warning_cycles"] == 0
    assert out["elapsed_critical_cycles"] == 0
    assert out["elapsed_recovery_cycles"] == 0
    assert out["timer_guard_triggered"] is False
    assert out["control_action"] == "observe"
    assert out["timed_transition_event"] == "remain_observe"
    assert out["escalation_required"] is False


def test_healthy_steady_state_never_enters_recover() -> None:
    state = TimedControllerState()
    outputs = [run_timed_controller_synthesis("safe", "nominal", controller_state=state) for _ in range(8)]

    assert all(out["controller_state"] == "observe" for out in outputs)
    assert all(out["control_action"] == "observe" for out in outputs)
    assert all(out["elapsed_recovery_cycles"] == 0 for out in outputs)


def test_recovery_only_after_prior_escalation_episode() -> None:
    state = TimedControllerState()
    run_timed_controller_synthesis("warning", "elevated", controller_state=state)
    run_timed_controller_synthesis("warning", "elevated", controller_state=state)
    run_timed_controller_synthesis("warning", "elevated", controller_state=state)

    first = run_timed_controller_synthesis("safe", "nominal", controller_state=state)
    second = run_timed_controller_synthesis("safe", "nominal", controller_state=state)

    assert first["control_action"] == "observe"
    assert second["control_action"] == "recover"
    assert second["controller_state"] == "recover"


def test_recovery_timer_reset_after_return_to_observe() -> None:
    state = TimedControllerState()
    run_timed_controller_synthesis("warning", "elevated", controller_state=state)
    run_timed_controller_synthesis("warning", "elevated", controller_state=state)
    run_timed_controller_synthesis("warning", "elevated", controller_state=state)
    run_timed_controller_synthesis("safe", "nominal", controller_state=state)
    recovered = run_timed_controller_synthesis("safe", "nominal", controller_state=state)
    returned = run_timed_controller_synthesis("safe", "nominal", controller_state=state)

    assert recovered["controller_state"] == "recover"
    assert returned["controller_state"] == "observe"
    assert returned["timed_transition_event"] == "recover_to_observe"
    assert state.was_escalated is False
    follow_up = run_timed_controller_synthesis("safe", "nominal", controller_state=state)
    assert follow_up["elapsed_recovery_cycles"] == 0
    assert follow_up["control_action"] == "observe"


def test_none_removed_from_control_actions() -> None:
    assert "none" not in CONTROL_ACTIONS
