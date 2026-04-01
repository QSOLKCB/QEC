"""Deterministic tests for v124.0.0 safety-state automata."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.safety_state_automata import SafetyState, SafetyStateAutomata


def _state(current_state: str, *, invariant_ok: bool = True, recovery_active: bool = False, emergency_stop: bool = False) -> SafetyState:
    return SafetyState(
        current_state=current_state,
        entry_count=1,
        invariant_ok=invariant_ok,
        recovery_active=recovery_active,
        emergency_stop=emergency_stop,
    )


def test_nominal_remains_nominal() -> None:
    automata = SafetyStateAutomata()
    result = automata.step(_state("nominal"), {"warning_score": 0.1})
    assert result["next_state"] == "nominal"


def test_nominal_to_alert_on_invariant_violation() -> None:
    automata = SafetyStateAutomata()
    result = automata.step(_state("nominal", invariant_ok=False), {"warning_score": 0.1})
    assert result["next_state"] == "alert"


def test_nominal_to_alert_on_warning_threshold() -> None:
    automata = SafetyStateAutomata()
    result = automata.step(_state("nominal"), {"warning_score": 0.5})
    assert result["next_state"] == "alert"


def test_alert_to_recovering() -> None:
    automata = SafetyStateAutomata()
    result = automata.step(_state("alert"), {"recovery_active": True})
    assert result["next_state"] == "recovering"


def test_alert_to_emergency_stop() -> None:
    automata = SafetyStateAutomata()
    result = automata.step(_state("alert"), {"emergency_stop": True})
    assert result["next_state"] == "emergency_stop"


def test_recovering_to_stabilized() -> None:
    automata = SafetyStateAutomata()
    result = automata.step(_state("recovering", recovery_active=True), {"invariant_ok": True})
    assert result["next_state"] == "stabilized"


def test_recovering_to_emergency_stop() -> None:
    automata = SafetyStateAutomata()
    result = automata.step(_state("recovering", recovery_active=True), {"emergency_stop": True})
    assert result["next_state"] == "emergency_stop"


def test_stabilized_to_nominal() -> None:
    automata = SafetyStateAutomata()
    result = automata.step(_state("stabilized", invariant_ok=True, recovery_active=False), {})
    assert result["next_state"] == "nominal"


def test_emergency_stop_is_absorbing() -> None:
    automata = SafetyStateAutomata()
    result = automata.step(_state("emergency_stop", emergency_stop=True), {"invariant_ok": True, "recovery_active": True})
    assert result["next_state"] == "emergency_stop"


def test_invalid_transition_rejected() -> None:
    automata = SafetyStateAutomata()
    is_valid = automata.validate_transition(
        "emergency_stop",
        "nominal",
        {"invariant_ok": True, "recovery_active": False, "emergency_stop": False},
    )
    assert is_valid is False


def test_deterministic_repeatability() -> None:
    automata = SafetyStateAutomata()
    start = _state("nominal")
    state_data = {"warning_score": 0.6, "invariant_ok": True, "recovery_active": False, "emergency_stop": False}
    out1 = automata.step(start, state_data)
    out2 = automata.step(start, state_data)
    assert out1 == out2


def test_frozen_dataclass_immutability() -> None:
    state = _state("nominal")
    with pytest.raises(FrozenInstanceError):
        state.current_state = "alert"


def test_exact_transition_label_correctness() -> None:
    automata = SafetyStateAutomata()
    nominal_alert = automata.step(_state("nominal"), {"warning_score": 0.8})
    alert_recovering = automata.step(_state("alert"), {"recovery_active": True})
    alert_emergency = automata.step(_state("alert"), {"emergency_stop": True})
    recovering_stabilized = automata.step(_state("recovering", recovery_active=True), {"invariant_ok": True})
    recovering_emergency = automata.step(_state("recovering", recovery_active=True), {"emergency_stop": True})
    stabilized_nominal = automata.step(_state("stabilized", invariant_ok=True, recovery_active=False), {})
    nominal_remain = automata.step(_state("nominal"), {"warning_score": 0.1})
    emergency_remain = automata.step(_state("emergency_stop", emergency_stop=True), {})

    assert nominal_alert["safety_transition"] == "nominal_to_alert"
    assert alert_recovering["safety_transition"] == "alert_to_recovering"
    assert alert_emergency["safety_transition"] == "alert_to_emergency_stop"
    assert recovering_stabilized["safety_transition"] == "recovering_to_stabilized"
    assert recovering_emergency["safety_transition"] == "recovering_to_emergency_stop"
    assert stabilized_nominal["safety_transition"] == "stabilized_to_nominal"
    assert nominal_remain["safety_transition"] == "remain_nominal"
    assert emergency_remain["safety_transition"] == "remain_emergency_stop"
