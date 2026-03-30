"""Deterministic tests for v113.0.0 ternary lattice controller."""

from __future__ import annotations

import pytest

from qec.analysis.ternary_lattice_controller import (
    LATTICE_CLASS_INTERVENTION,
    run_ternary_lattice_controller,
)


REQUIRED_KEYS = {
    "chain_length",
    "controller_result",
    "ternary_lattice_state",
    "lattice_transition_count",
    "lattice_stability_score",
    "lattice_control_class",
}


def _controller_stub(*, controller_state: str, chain_length: int = 5) -> dict[str, object]:
    return {
        "chain_length": chain_length,
        "field_result": {},
        "controller_state": controller_state,
        "state_transition_count": 0,
        "controller_stability_score": 1.0,
        "control_loop_class": "stable_loop",
    }


def test_exact_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.ternary_lattice_controller.run_finite_state_controller",
        lambda **_: _controller_stub(controller_state="rebalance_state"),
    )

    out1 = run_ternary_lattice_controller(controller_cycles=3, lattice_cycles=3)
    out2 = run_ternary_lattice_controller(controller_cycles=3, lattice_cycles=3)

    assert out1 == out2


def test_state_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {
        "idle_state": (0, 0, 0, 0, 0),
        "stabilization_state": (0, 1, 1, 1, 0),
        "rebalance_state": (-1, -1, 0, 1, 1),
        "intervention_state": (-1, 0, 0, 0, 1),
    }

    for controller_state, expected_lattice in expected.items():
        monkeypatch.setattr(
            "qec.analysis.ternary_lattice_controller.run_finite_state_controller",
            lambda **_: _controller_stub(controller_state=controller_state),
        )
        out = run_ternary_lattice_controller(lattice_cycles=0)
        assert out["ternary_lattice_state"] == expected_lattice


def test_ternary_value_invariant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.ternary_lattice_controller.run_finite_state_controller",
        lambda **_: _controller_stub(controller_state="intervention_state", chain_length=7),
    )

    out = run_ternary_lattice_controller(lattice_cycles=4)

    assert set(out["ternary_lattice_state"]).issubset({-1, 0, 1})


def test_transition_counting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.ternary_lattice_controller.run_finite_state_controller",
        lambda **_: _controller_stub(controller_state="intervention_state", chain_length=5),
    )

    out = run_ternary_lattice_controller(lattice_cycles=2, return_lattice_trace=True)

    assert out["lattice_trace"] == [
        (-1, 0, 0, 0, 1),
        (-1, -1, 0, 1, 1),
        (-1, -1, 0, 1, 1),
    ]
    assert out["lattice_transition_count"] == 2


def test_stability_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.ternary_lattice_controller.run_finite_state_controller",
        lambda **_: _controller_stub(controller_state="intervention_state", chain_length=5),
    )

    out = run_ternary_lattice_controller(lattice_cycles=2)

    assert 0.0 <= out["lattice_stability_score"] <= 1.0
    assert out["lattice_stability_score"] == pytest.approx(0.8)


def test_trace_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.ternary_lattice_controller.run_finite_state_controller",
        lambda **_: _controller_stub(controller_state="idle_state", chain_length=3),
    )

    out = run_ternary_lattice_controller(lattice_cycles=3, return_lattice_trace=True)

    assert set(out.keys()) == REQUIRED_KEYS | {"lattice_trace"}
    assert out["lattice_trace"] == [
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
    ]


def test_short_chain_edge_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.ternary_lattice_controller.run_finite_state_controller",
        lambda **_: _controller_stub(controller_state="intervention_state", chain_length=1),
    )

    out = run_ternary_lattice_controller(chain_length=1, chain_state=[0.7], lattice_cycles=1)

    assert out["chain_length"] == 1
    assert out["ternary_lattice_state"] == (1,)
    assert out["lattice_control_class"] == LATTICE_CLASS_INTERVENTION
    assert 0.0 <= out["lattice_stability_score"] <= 1.0
