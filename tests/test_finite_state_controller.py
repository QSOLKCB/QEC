"""Deterministic tests for v112.1.0 finite-state correction controller."""

from __future__ import annotations

import pytest

from qec.analysis.finite_state_controller import (
    CONTROL_LOOP_ADAPTIVE,
    CONTROL_LOOP_INTERVENTION,
    CONTROL_LOOP_STABLE,
    run_finite_state_controller,
)


REQUIRED_KEYS = {
    "chain_length",
    "field_result",
    "controller_state",
    "state_transition_count",
    "controller_stability_score",
    "control_loop_class",
}


def _field_stub(*, correction_action: str, chain_length: int = 3) -> dict[str, object]:
    return {
        "chain_length": chain_length,
        "dispatch_result": {"correction_action": correction_action},
        "initial_field": tuple(0.5 for _ in range(chain_length)),
        "corrected_field": tuple(0.5 for _ in range(chain_length)),
        "field_drift_score": 0.0,
        "local_stability_score": 1.0,
        "correction_field_class": "stable_field",
        "automata_trace": [tuple(0.5 for _ in range(chain_length))],
    }


def test_runtime_none_check() -> None:
    with pytest.raises(RuntimeError, match="latest_field_result must not be None after controller execution"):
        run_finite_state_controller(controller_cycles=-1)


def test_exact_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.finite_state_controller.run_cellular_correction_field",
        lambda **_: _field_stub(correction_action="local_stabilize"),
    )

    out1 = run_finite_state_controller(controller_cycles=3)
    out2 = run_finite_state_controller(controller_cycles=3)

    assert out1 == out2


def test_state_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    action_to_state = {
        "hold_state": "idle_state",
        "local_stabilize": "stabilization_state",
        "spectral_rebalance": "rebalance_state",
        "boundary_intervene": "intervention_state",
    }

    for action, expected_state in action_to_state.items():
        monkeypatch.setattr(
            "qec.analysis.finite_state_controller.run_cellular_correction_field",
            lambda **kwargs: _field_stub(correction_action=action, chain_length=kwargs.get("chain_length", 3)),
        )
        out = run_finite_state_controller(controller_cycles=1)
        assert out["controller_state"] == expected_state


def test_transition_counting(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = iter(["hold_state", "local_stabilize", "local_stabilize", "spectral_rebalance"])

    monkeypatch.setattr(
        "qec.analysis.finite_state_controller.run_cellular_correction_field",
        lambda **_: _field_stub(correction_action=next(actions)),
    )

    out = run_finite_state_controller(controller_cycles=4, return_controller_trace=True)

    assert out["controller_trace"] == [
        "idle_state",
        "stabilization_state",
        "stabilization_state",
        "rebalance_state",
    ]
    assert out["state_transition_count"] == 2


def test_stability_score_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    actions = iter(["hold_state", "local_stabilize", "spectral_rebalance", "boundary_intervene"])

    monkeypatch.setattr(
        "qec.analysis.finite_state_controller.run_cellular_correction_field",
        lambda **_: _field_stub(correction_action=next(actions)),
    )

    out = run_finite_state_controller(controller_cycles=4)

    assert 0.0 <= out["controller_stability_score"] <= 1.0
    assert out["controller_stability_score"] == pytest.approx(0.0)


def test_multi_cycle_transition_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            {
                **_field_stub(correction_action="local_stabilize"),
                "field_drift_score": 0.2,
                "local_stability_score": 0.8,
            },
            {
                **_field_stub(correction_action="hold_state"),
                "field_drift_score": 0.0,
                "local_stability_score": 1.0,
            },
        ]
    )

    monkeypatch.setattr(
        "qec.analysis.finite_state_controller.run_cellular_correction_field",
        lambda **_: next(responses),
    )

    out = run_finite_state_controller(controller_cycles=2, return_controller_trace=True)

    assert out["controller_trace"] == ["stabilization_state", "idle_state"]
    assert out["state_transition_count"] == 1
    assert out["controller_stability_score"] == 0.0


def test_stability_changes_across_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.finite_state_controller.run_cellular_correction_field",
        lambda **_: {
            **_field_stub(correction_action="local_stabilize"),
            "field_drift_score": 0.2,
            "local_stability_score": 0.8,
        },
    )
    stable = run_finite_state_controller(controller_cycles=1)

    responses = iter(
        [
            {
                **_field_stub(correction_action="local_stabilize"),
                "field_drift_score": 0.2,
                "local_stability_score": 0.8,
            },
            {
                **_field_stub(correction_action="hold_state"),
                "field_drift_score": 0.0,
                "local_stability_score": 1.0,
            },
        ]
    )
    monkeypatch.setattr(
        "qec.analysis.finite_state_controller.run_cellular_correction_field",
        lambda **_: next(responses),
    )
    adaptive = run_finite_state_controller(controller_cycles=2)

    assert stable["controller_stability_score"] == 1.0
    assert adaptive["controller_stability_score"] == 0.0


def test_trace_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.finite_state_controller.run_cellular_correction_field",
        lambda **_: _field_stub(correction_action="hold_state"),
    )

    out = run_finite_state_controller(controller_cycles=3, return_controller_trace=True)

    assert set(out.keys()) == REQUIRED_KEYS | {"controller_trace"}
    assert out["controller_trace"] == ["idle_state", "idle_state", "idle_state"]


def test_intervention_loop_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.finite_state_controller.run_cellular_correction_field",
        lambda **_: _field_stub(correction_action="boundary_intervene"),
    )

    out = run_finite_state_controller(controller_cycles=2)

    assert out["controller_state"] == "intervention_state"
    assert out["control_loop_class"] == CONTROL_LOOP_INTERVENTION


def test_short_chain_edge_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.finite_state_controller.run_cellular_correction_field",
        lambda **kwargs: _field_stub(correction_action="hold_state", chain_length=kwargs.get("chain_length", 1)),
    )

    out = run_finite_state_controller(chain_length=1, chain_state=[0.7], controller_cycles=1)

    assert out["chain_length"] == 1
    assert out["controller_state"] == "idle_state"
    assert out["state_transition_count"] == 0
    assert out["controller_stability_score"] == 1.0
    assert out["control_loop_class"] == CONTROL_LOOP_STABLE


def test_loop_class_constants_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.finite_state_controller.run_cellular_correction_field",
        lambda **_: _field_stub(correction_action="spectral_rebalance"),
    )

    out = run_finite_state_controller(controller_cycles=2)
    assert out["control_loop_class"] == CONTROL_LOOP_ADAPTIVE
