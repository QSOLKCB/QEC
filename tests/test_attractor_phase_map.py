"""Deterministic tests for v114.0.0 attractor phase-map engine."""

from __future__ import annotations

import pytest

from qec.analysis.attractor_phase_map import run_attractor_phase_map


REQUIRED_KEYS = {
    "chain_length",
    "lattice_result",
    "attractor_state",
    "attractor_cycle_length",
    "phase_stability_score",
    "phase_class",
}


def _lattice_result_stub(*, chain_length: int, controller_state: str, lattice_trace: list[tuple[int, ...]]) -> dict[str, object]:
    return {
        "chain_length": chain_length,
        "controller_result": {
            "controller_state": controller_state,
        },
        "lattice_trace": lattice_trace,
    }


def test_fixed_point_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=5,
            controller_state="idle_state",
            lattice_trace=[
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0),
            ],
        ),
    )

    out = run_attractor_phase_map()

    assert out["attractor_state"] == "fixed_point"
    assert out["attractor_cycle_length"] == 1
    assert out["phase_class"] == "stable_phase"


def test_period_two_oscillation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=4,
            controller_state="rebalance_state",
            lattice_trace=[
                (1, 1, -1, -1),
                (-1, -1, 1, 1),
                (1, 1, -1, -1),
                (-1, -1, 1, 1),
            ],
        ),
    )

    out = run_attractor_phase_map()

    assert out["attractor_state"] == "period_two"
    assert out["attractor_cycle_length"] == 2
    assert out["phase_class"] == "oscillatory_phase"


def test_drifting_regime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=4,
            controller_state="rebalance_state",
            lattice_trace=[
                (0, 0, 0, 0),
                (1, 0, 0, 0),
                (1, 1, 0, 0),
                (1, 1, 1, 0),
            ],
        ),
    )

    out = run_attractor_phase_map()

    assert out["attractor_state"] == "drifting_phase"
    assert out["attractor_cycle_length"] == 0
    assert out["phase_class"] == "drifting_phase"


def test_exact_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=3,
            controller_state="idle_state",
            lattice_trace=[
                (0, 0, 0),
                (0, 0, 0),
            ],
        ),
    )

    out1 = run_attractor_phase_map()
    out2 = run_attractor_phase_map()

    assert out1 == out2


def test_stability_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=4,
            controller_state="rebalance_state",
            lattice_trace=[
                (0, 0, 0, 0),
                (1, 1, 1, 1),
                (0, 0, 0, 0),
                (1, 1, 1, 1),
            ],
        ),
    )

    out = run_attractor_phase_map()

    assert 0.0 <= out["phase_stability_score"] <= 1.0
    assert out["phase_stability_score"] == pytest.approx(0.0)


def test_trace_output(monkeypatch: pytest.MonkeyPatch) -> None:
    lattice_trace = [
        (0, 0, 0),
        (0, 0, 0),
    ]
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=3,
            controller_state="idle_state",
            lattice_trace=lattice_trace,
        ),
    )

    out = run_attractor_phase_map(return_phase_trace=True)

    assert set(out.keys()) == REQUIRED_KEYS | {"phase_trace"}
    assert out["phase_trace"] == lattice_trace


def test_short_chain_edge_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=1,
            controller_state="intervention_state",
            lattice_trace=[
                (1,),
                (-1,),
            ],
        ),
    )

    out = run_attractor_phase_map(chain_length=1, chain_state=[0.7])

    assert out["chain_length"] == 1
    assert out["attractor_state"] == "intervention_phase"
    assert out["attractor_cycle_length"] == 0
    assert out["phase_class"] == "critical_phase"
    assert 0.0 <= out["phase_stability_score"] <= 1.0
