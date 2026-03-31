"""Tests for v122.0.0 deterministic hybrid automata layer."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.hybrid_automata import HybridAutomata, HybridState


def test_nominal_remains_nominal() -> None:
    automata = HybridAutomata()
    state = HybridState(
        mode="nominal",
        continuous_value=0.10,
        drift_rate=0.02,
        risk_score=0.20,
    )
    result = automata.step(state)

    assert result["previous_mode"] == "nominal"
    assert result["next_mode"] == "nominal"
    assert result["mode_transition"] == "remain_nominal"
    assert result["hybrid_stable"] is True


def test_nominal_to_warning() -> None:
    automata = HybridAutomata()
    state = HybridState(
        mode="nominal",
        continuous_value=0.25,
        drift_rate=0.10,
        risk_score=0.20,
    )
    result = automata.step(state)

    assert result["next_mode"] == "warning"
    assert result["mode_transition"] == "nominal_to_warning"


def test_warning_to_critical() -> None:
    automata = HybridAutomata()
    state = HybridState(
        mode="warning",
        continuous_value=0.60,
        drift_rate=0.10,
        risk_score=0.40,
    )
    result = automata.step(state)

    assert result["next_mode"] == "critical"
    assert result["mode_transition"] == "warning_to_critical"


def test_critical_to_recovery() -> None:
    automata = HybridAutomata()
    state = HybridState(
        mode="critical",
        continuous_value=0.55,
        drift_rate=-0.20,
        risk_score=0.00,
    )
    result = automata.step(state)

    assert result["continuous_value"] < 0.5
    assert result["next_mode"] == "recovery"
    assert result["mode_transition"] == "critical_to_recovery"


def test_recovery_to_nominal() -> None:
    automata = HybridAutomata()
    state = HybridState(
        mode="recovery",
        continuous_value=0.20,
        drift_rate=0.00,
        risk_score=0.00,
    )
    result = automata.step(state)

    assert result["next_mode"] == "nominal"
    assert result["mode_transition"] == "recovery_to_nominal"


def test_continuous_value_clamps_to_bounds() -> None:
    automata = HybridAutomata()

    high = HybridState(
        mode="warning",
        continuous_value=0.95,
        drift_rate=0.20,
        risk_score=1.00,
    )
    low = HybridState(
        mode="warning",
        continuous_value=0.05,
        drift_rate=-0.30,
        risk_score=0.00,
    )

    assert automata.step(high)["continuous_value"] == 1.0
    assert automata.step(low)["continuous_value"] == 0.0


def test_deterministic_repeatability() -> None:
    automata = HybridAutomata()
    state = HybridState(
        mode="warning",
        continuous_value=0.42,
        drift_rate=0.05,
        risk_score=0.08,
    )

    first = automata.step(state)
    second = automata.step(state)

    assert first == second


def test_hybrid_state_is_frozen() -> None:
    state = HybridState(
        mode="nominal",
        continuous_value=0.10,
        drift_rate=0.00,
        risk_score=0.00,
    )

    with pytest.raises(FrozenInstanceError):
        state.mode = "warning"


def test_transition_labels_are_exact() -> None:
    automata = HybridAutomata()

    nominal_state = HybridState("nominal", 0.29, 0.0, 0.0)
    warning_state = HybridState("warning", 0.69, 0.0, 0.05)
    critical_state = HybridState("critical", 0.51, -0.03, 0.0)
    recovery_state = HybridState("recovery", 0.10, 0.0, 0.0)

    assert automata.step(nominal_state)["mode_transition"] == "remain_nominal"
    assert automata.step(warning_state)["mode_transition"] == "warning_to_critical"
    assert automata.step(critical_state)["mode_transition"] == "critical_to_recovery"
    assert automata.step(recovery_state)["mode_transition"] == "recovery_to_nominal"
