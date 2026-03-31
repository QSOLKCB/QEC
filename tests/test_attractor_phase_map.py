"""Deterministic tests for v114.0.0 attractor phase-map engine."""

from __future__ import annotations

import pytest

import qec.analysis.api as analysis_api
from qec.analysis.attractor_phase_map import run_attractor_phase_map
from qec.analysis.spectral_phase_boundary import PHASE_CLASS_STABLE as BOUNDARY_PHASE_CLASS_STABLE


REQUIRED_KEYS = {
    "chain_length",
    "lattice_result",
    "attractor_state",
    "attractor_cycle_length",
    "phase_stability_score",
    "phase_class",
    "phase_transition_index",
    "attractor_entry_cycle",
    "transition_sharpness_score",
    "attractor_confidence_score",
    "detected_cycle_period",
    "cycle_spectrum_class",
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
    assert out["phase_transition_index"] == 1
    assert out["attractor_entry_cycle"] == 0
    assert out["transition_sharpness_score"] == pytest.approx(1.0)


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
    assert out["phase_transition_index"] == 2
    assert out["attractor_entry_cycle"] == 0
    assert 0.0 <= out["transition_sharpness_score"] <= 1.0


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
    assert out["phase_transition_index"] == -1
    assert out["attractor_entry_cycle"] == -1
    assert out["transition_sharpness_score"] == pytest.approx(0.0)


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
    assert 0.0 <= out["transition_sharpness_score"] <= 1.0


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
    assert out["attractor_entry_cycle"] == -1
    assert 0.0 <= out["transition_sharpness_score"] <= 1.0


def test_insufficient_trace_not_fixed_point(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=3,
            controller_state="idle_state",
            lattice_trace=[(0, 0, 0)],
        ),
    )

    out = run_attractor_phase_map()

    assert out["attractor_state"] == "drifting_phase"
    assert out["attractor_cycle_length"] == 0
    assert out["phase_transition_index"] == -1
    assert out["attractor_entry_cycle"] == -1


def test_period_two_entry_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=4,
            controller_state="rebalance_state",
            lattice_trace=[
                (0, 0, 0, 0),
                (1, 1, 1, 1),
                (1, -1, 1, -1),
                (-1, 1, -1, 1),
                (1, -1, 1, -1),
                (-1, 1, -1, 1),
            ],
        ),
    )

    out = run_attractor_phase_map()
    assert out["attractor_state"] == "period_two"
    assert out["phase_transition_index"] == 4
    assert out["attractor_entry_cycle"] == 2


def test_fixed_point_entry_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=3,
            controller_state="idle_state",
            lattice_trace=[
                (0, 0, 0),
                (0, 1, 0),
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
            ],
        ),
    )

    out = run_attractor_phase_map()
    assert out["attractor_state"] == "fixed_point"
    assert out["phase_transition_index"] == 3
    assert out["attractor_entry_cycle"] == 2


def test_period_two_entry_opposite_phase_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=2,
            controller_state="rebalance_state",
            lattice_trace=[
                (0, 0),
                (3, 3),
                (2, 2),
                (1, 1),
                (2, 2),
                (1, 1),
                (2, 2),
            ],
        ),
    )

    out = run_attractor_phase_map()
    assert out["attractor_state"] == "period_two"
    assert out["attractor_entry_cycle"] == 2


def test_fixed_point_entry_cycle_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=2,
            controller_state="idle_state",
            lattice_trace=[
                (0, 0),
                (1, 0),
                (1, 1),
                (1, 1),
                (1, 1),
                (1, 1),
            ],
        ),
    )

    out = run_attractor_phase_map()
    assert out["attractor_state"] == "fixed_point"
    assert out["attractor_entry_cycle"] == 2


def test_sharpness_score_deterministic_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=3,
            controller_state="idle_state",
            lattice_trace=[
                (0, 0, 0),
                (1, 0, 0),
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
            ],
        ),
    )

    out1 = run_attractor_phase_map()
    out2 = run_attractor_phase_map()
    assert out1["transition_sharpness_score"] == pytest.approx(0.0)
    assert out1["transition_sharpness_score"] == out2["transition_sharpness_score"]


def test_mismatched_lattice_lengths_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=3,
            controller_state="rebalance_state",
            lattice_trace=[(0, 0, 0), (0, 0)],
        ),
    )

    with pytest.raises(ValueError, match="lattice trace states must have equal length"):
        run_attractor_phase_map()


def test_api_phase_class_symbol_collision_prevented() -> None:
    assert analysis_api.PHASE_CLASS_STABLE == BOUNDARY_PHASE_CLASS_STABLE


def test_confidence_fixed_point_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=3,
            controller_state="idle_state",
            lattice_trace=[
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
            ],
        ),
    )

    out = run_attractor_phase_map()
    assert out["detected_cycle_period"] == 1
    assert out["attractor_confidence_score"] == pytest.approx(1.0)
    assert out["cycle_spectrum_class"] == "fixed_cycle"


def test_confidence_period_two_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=2,
            controller_state="idle_state",
            lattice_trace=[
                (0, 1),
                (1, 0),
                (0, 1),
                (1, 0),
                (0, 1),
                (1, 0),
            ],
        ),
    )

    out = run_attractor_phase_map()
    assert out["detected_cycle_period"] == 2
    assert out["attractor_confidence_score"] == pytest.approx(1.0)
    assert out["cycle_spectrum_class"] == "low_period_cycle"


def test_detected_cycle_period_three(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=1,
            controller_state="idle_state",
            lattice_trace=[(0,), (1,), (1,), (0,), (1,), (1,)],
        ),
    )

    out = run_attractor_phase_map()
    assert out["detected_cycle_period"] == 3
    assert out["cycle_spectrum_class"] == "medium_period_cycle"


def test_detected_cycle_period_four(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=1,
            controller_state="idle_state",
            lattice_trace=[(0,), (1,), (1,), (0,), (0,), (1,), (1,), (0,)],
        ),
    )

    out = run_attractor_phase_map()
    assert out["detected_cycle_period"] == 4
    assert out["cycle_spectrum_class"] == "medium_period_cycle"


def test_fixed_and_period_two_detection_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=2,
            controller_state="idle_state",
            lattice_trace=[(1, 1), (1, 1), (1, 1), (1, 1)],
        ),
    )
    out_fixed = run_attractor_phase_map()
    assert out_fixed["detected_cycle_period"] == 1

    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=2,
            controller_state="idle_state",
            lattice_trace=[(0, 1), (1, 0), (0, 1), (1, 0)],
        ),
    )
    out_period_two = run_attractor_phase_map()
    assert out_period_two["detected_cycle_period"] == 2


def test_drifting_detected_cycle_period_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=1,
            controller_state="idle_state",
            lattice_trace=[(0,), (1,), (2,), (3,), (4,), (5,)],
        ),
    )

    out = run_attractor_phase_map()
    assert out["detected_cycle_period"] == 0
    assert out["attractor_confidence_score"] == pytest.approx(0.0)
    assert out["cycle_spectrum_class"] == "drifting_cycle"


def test_cycle_spectrum_outputs_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=2,
            controller_state="idle_state",
            lattice_trace=[(0, 1), (1, 0), (0, 1), (1, 0)],
        ),
    )

    out1 = run_attractor_phase_map()
    out2 = run_attractor_phase_map()
    assert out1["detected_cycle_period"] == out2["detected_cycle_period"]
    assert out1["attractor_confidence_score"] == out2["attractor_confidence_score"]
    assert out1["cycle_spectrum_class"] == out2["cycle_spectrum_class"]


def test_confidence_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qec.analysis.attractor_phase_map.run_ternary_lattice_controller",
        lambda **_: _lattice_result_stub(
            chain_length=1,
            controller_state="idle_state",
            lattice_trace=[(0,), (1,), (0,), (1,), (2,)],
        ),
    )

    out = run_attractor_phase_map()
    assert 0.0 <= out["attractor_confidence_score"] <= 1.0
