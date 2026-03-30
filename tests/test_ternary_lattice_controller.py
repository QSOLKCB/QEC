"""Deterministic tests for v113.0.0 ternary lattice controller."""

from __future__ import annotations

import pytest

from qec.analysis.ternary_lattice_controller import (
    ALLOWED_BOUNDARY_MODES,
    BOUNDARY_MODE_FIXED,
    BOUNDARY_MODE_PERIODIC,
    BOUNDARY_MODE_REFLECTIVE,
    LATTICE_CLASS_INTERVENTION,
    _count_transitions,
    _lattice_stability_score,
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


def test_boundary_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.ternary_lattice_controller.run_finite_state_controller",
        lambda **_: _controller_stub(controller_state="intervention_state", chain_length=3),
    )

    fixed = run_ternary_lattice_controller(
        lattice_cycles=1,
        lattice_boundary_mode=BOUNDARY_MODE_FIXED,
        return_lattice_trace=True,
    )
    reflective = run_ternary_lattice_controller(
        lattice_cycles=1,
        lattice_boundary_mode=BOUNDARY_MODE_REFLECTIVE,
        return_lattice_trace=True,
    )
    periodic = run_ternary_lattice_controller(
        lattice_cycles=1,
        lattice_boundary_mode=BOUNDARY_MODE_PERIODIC,
        return_lattice_trace=True,
    )

    assert fixed["lattice_trace"][-1] == (-1, 0, 1)
    assert reflective["lattice_trace"][-1] == (-1, 0, 1)
    assert periodic["lattice_trace"][-1] == (0, 0, 0)


def test_invalid_boundary_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.ternary_lattice_controller.run_finite_state_controller",
        lambda **_: _controller_stub(controller_state="idle_state", chain_length=3),
    )

    with pytest.raises(ValueError, match="invalid lattice_boundary_mode: bad-mode"):
        run_ternary_lattice_controller(lattice_cycles=1, lattice_boundary_mode="bad-mode")


def test_invalid_boundary_mode_raises_with_zero_lattice_cycles() -> None:
    invalid_mode = "bad-mode"
    assert invalid_mode not in ALLOWED_BOUNDARY_MODES
    with pytest.raises(ValueError, match="invalid lattice_boundary_mode: bad-mode"):
        run_ternary_lattice_controller(lattice_cycles=0, lattice_boundary_mode=invalid_mode)


def test_zero_controller_cycles_returns_deterministic_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.ternary_lattice_controller.run_finite_state_controller",
        lambda **_: pytest.fail("run_finite_state_controller should not be called for non-positive cycles"),
    )

    out = run_ternary_lattice_controller(controller_cycles=0, lattice_cycles=1, return_lattice_trace=True)

    assert out["controller_result"]["controller_state"] == "idle_state"
    assert out["controller_result"]["controller_stability_score"] == 1.0
    assert out["lattice_trace"] == [(0, 0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0, 0)]


def test_zero_lattice_cycles_stability_is_one() -> None:
    assert _lattice_stability_score(transition_count=999, chain_length=9, lattice_cycles=0) == 1.0


def test_transition_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="lattice states must have equal length"):
        _count_transitions((0, 1), (0,))
