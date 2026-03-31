"""Deterministic tests for v125.0.0 threshold hysteresis controller."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.threshold_hysteresis_controller import HysteresisState, ThresholdHysteresisController


def _state(current_band: str, *, warning_counter: int = 0, critical_counter: int = 0, recovery_counter: int = 0) -> HysteresisState:
    return HysteresisState(
        current_band=current_band,
        warning_counter=warning_counter,
        critical_counter=critical_counter,
        recovery_counter=recovery_counter,
    )


def _advance_state(controller: ThresholdHysteresisController, state: HysteresisState, score: float) -> HysteresisState:
    observed = controller.classify_band(score)
    next_band = controller.apply_hysteresis(state, score)

    warning_counter = state.warning_counter + 1 if observed == "warning" else 0
    critical_counter = state.critical_counter + 1 if observed == "critical" else 0

    if next_band == "critical":
        recovery_counter = state.recovery_counter + 1 if score < 0.5 else 0
    elif next_band in ("warning", "recovery"):
        if next_band == "warning" and observed == "nominal":
            recovery_counter = state.recovery_counter + 1
        elif next_band == "recovery" and score < 0.3:
            recovery_counter = state.recovery_counter + 1
        else:
            recovery_counter = 0
    else:
        recovery_counter = 0

    return HysteresisState(
        current_band=next_band,
        warning_counter=warning_counter,
        critical_counter=critical_counter,
        recovery_counter=recovery_counter,
    )


def test_no_single_cycle_escalation() -> None:
    controller = ThresholdHysteresisController()
    result = controller.step(_state("nominal", warning_counter=0), 0.55)
    assert result["next_band"] == "nominal"
    assert result["hysteresis_active"] is True


def test_nominal_to_warning_after_2_cycles() -> None:
    controller = ThresholdHysteresisController()
    first = _state("nominal", warning_counter=0)
    second = _state("nominal", warning_counter=1)
    assert controller.step(first, 0.55)["next_band"] == "nominal"
    assert controller.step(second, 0.55)["next_band"] == "warning"


def test_warning_to_critical_after_2_cycles() -> None:
    controller = ThresholdHysteresisController()
    first = _state("warning", critical_counter=0)
    second = _state("warning", critical_counter=1)
    assert controller.step(first, 0.80)["next_band"] == "warning"
    assert controller.step(second, 0.80)["next_band"] == "critical"


def test_no_single_cycle_de_escalation() -> None:
    controller = ThresholdHysteresisController()
    result = controller.step(_state("warning", recovery_counter=0), 0.20)
    assert result["next_band"] == "warning"
    assert result["hysteresis_active"] is True


def test_critical_to_warning_after_2_cycles() -> None:
    controller = ThresholdHysteresisController()
    first = _state("critical", warning_counter=0, recovery_counter=0)
    second = _state("critical", warning_counter=1, recovery_counter=0)
    assert controller.step(first, 0.60)["next_band"] == "critical"
    assert controller.step(second, 0.60)["next_band"] == "warning"


def test_critical_to_recovery_after_2_cycles_below_threshold() -> None:
    controller = ThresholdHysteresisController()
    first = _state("critical", recovery_counter=0)
    second = _state("critical", recovery_counter=1)
    assert controller.step(first, 0.45)["next_band"] == "critical"
    assert controller.step(second, 0.45)["next_band"] == "recovery"


def test_recovery_to_nominal_after_2_cycles() -> None:
    controller = ThresholdHysteresisController()
    first = _state("recovery", recovery_counter=0)
    second = _state("recovery", recovery_counter=1)
    assert controller.step(first, 0.25)["next_band"] == "recovery"
    assert controller.step(second, 0.25)["next_band"] == "nominal"


def test_anti_flap_repeatability() -> None:
    controller = ThresholdHysteresisController()
    scores = [0.41, 0.39, 0.41, 0.39, 0.72, 0.69, 0.72, 0.69]

    state_a = _state("nominal")
    outputs_a = []
    for score in scores:
        outputs_a.append(controller.step(state_a, score))
        state_a = _advance_state(controller, state_a, score)

    state_b = _state("nominal")
    outputs_b = []
    for score in scores:
        outputs_b.append(controller.step(state_b, score))
        state_b = _advance_state(controller, state_b, score)

    assert outputs_a == outputs_b


def test_exact_transition_label_correctness() -> None:
    controller = ThresholdHysteresisController()

    remain_nominal = controller.step(_state("nominal", warning_counter=0), 0.20)
    nominal_warning = controller.step(_state("nominal", warning_counter=1), 0.50)
    warning_critical = controller.step(_state("warning", critical_counter=1), 0.90)
    critical_warning = controller.step(_state("critical", warning_counter=1, recovery_counter=0), 0.60)
    critical_recovery = controller.step(_state("critical", recovery_counter=1), 0.45)
    recovery_nominal = controller.step(_state("recovery", recovery_counter=1), 0.20)

    assert remain_nominal["band_transition"] == "remain_nominal"
    assert nominal_warning["band_transition"] == "nominal_to_warning"
    assert warning_critical["band_transition"] == "warning_to_critical"
    assert critical_warning["band_transition"] == "critical_to_warning"
    assert critical_recovery["band_transition"] == "critical_to_recovery"
    assert recovery_nominal["band_transition"] == "recovery_to_nominal"


def test_frozen_dataclass_immutability() -> None:
    state = _state("nominal")
    with pytest.raises(FrozenInstanceError):
        state.current_band = "warning"
