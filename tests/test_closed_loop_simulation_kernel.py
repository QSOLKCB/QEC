from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.closed_loop_simulation_kernel import (
    MAX_SIMULATION_CYCLES,
    ClosedLoopSimulationReceipt,
    SimulationConfig,
    SimulationCycleRecord,
    SimulationSummary,
    run_closed_loop_simulation,
)
from qec.analysis.deterministic_stress_lattice import StressAxis, generate_stress_lattice
from qec.analysis.state_conditioned_filter_mesh import FilterMeshState, FilterOrdering


def _axes() -> tuple[StressAxis, ...]:
    return (
        StressAxis("thermal_pressure", 0.0, 1.0),
        StressAxis("latency_drift", 0.0, 1.0),
        StressAxis("timing_skew", 0.0, 1.0),
        StressAxis("power_pressure", 0.0, 1.0),
        StressAxis("consensus_instability", 0.0, 1.0),
    )


def _orderings() -> tuple[FilterOrdering, ...]:
    return (
        FilterOrdering.build(("thermal_stabilize", "parity_gate"), ("boundary_control",)),
        FilterOrdering.build(("latency_buffer", "spectral_phase"), ("surface_sync",)),
        FilterOrdering.build(("power_budget", "consensus_stabilize"), ("timing_sync",)),
    )


def _config(*, cycle_count: int = 8, recurrence_window: int = 6, point_count: int = 4) -> SimulationConfig:
    axes = _axes()
    candidate_orderings = _orderings()
    payload = {
        "axes": tuple(axis.to_dict() for axis in axes),
        "point_count": point_count,
        "stress_method": "lattice",
        "cycle_count": cycle_count,
        "candidate_orderings": tuple(item.to_dict() for item in candidate_orderings),
        "recurrence_window": recurrence_window,
    }
    return SimulationConfig(
        axes=axes,
        point_count=point_count,
        stress_method="lattice",
        cycle_count=cycle_count,
        candidate_orderings=candidate_orderings,
        recurrence_window=recurrence_window,
        stable_hash=sha256_hex(payload),
    )


def test_deterministic_replay() -> None:
    config = _config()
    first = run_closed_loop_simulation(config)
    second = run_closed_loop_simulation(config)
    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_bounded_cycle_count() -> None:
    config = _config(cycle_count=MAX_SIMULATION_CYCLES)
    receipt = run_closed_loop_simulation(config)
    assert len(receipt.cycle_records) == config.cycle_count
    assert len(receipt.cycle_records) <= MAX_SIMULATION_CYCLES


def test_stress_point_cycling_by_modulo() -> None:
    config = _config(cycle_count=7, point_count=3)
    receipt = run_closed_loop_simulation(config)
    stress = generate_stress_lattice(list(config.axes), config.point_count, config.stress_method)
    expected = tuple(stress.points[i % config.point_count].stable_hash for i in range(config.cycle_count))
    observed = tuple(item.stress_point_hash for item in receipt.cycle_records)
    assert observed == expected


def test_state_derivation_boundedness() -> None:
    receipt = run_closed_loop_simulation(_config())
    for cycle in receipt.cycle_records:
        state = cycle.state
        for value in (
            state.thermal_pressure,
            state.latency_drift,
            state.timing_skew,
            state.power_pressure,
            state.consensus_instability,
        ):
            assert 0.0 <= value <= 1.0


def test_end_to_end_receipt_linkage() -> None:
    receipt = run_closed_loop_simulation(_config())
    for cycle in receipt.cycle_records:
        assert len(cycle.stress_point_hash) == 64
        assert len(cycle.filter_mesh_receipt_hash) == 64
        assert len(cycle.transition_policy_receipt_hash) == 64
        assert len(cycle.refinement_receipt_hash) == 64


def test_summary_count_consistency() -> None:
    receipt = run_closed_loop_simulation(_config())
    summary = receipt.summary
    assert summary.stable_transition_count + summary.uncertain_transition_count == summary.cycle_count
    assert summary.converged_count + summary.bounded_count + summary.no_improvement_count == summary.cycle_count


def test_recurrence_not_evaluated_path() -> None:
    config = _config(cycle_count=3, recurrence_window=8)
    receipt = run_closed_loop_simulation(config)
    assert receipt.summary.recurrence_classification == "not_evaluated"
    assert receipt.summary.dominant_recurrence_period is None


def test_recurrence_evaluated_path() -> None:
    config = _config(cycle_count=12, recurrence_window=2, point_count=2)
    receipt = run_closed_loop_simulation(config)
    assert receipt.summary.recurrence_classification in {"aperiodic", "weak_periodic", "strong_periodic"}


def test_tampered_config_rejection() -> None:
    config = _config()
    object.__setattr__(config, "stable_hash", "0" * 64)
    with pytest.raises(ValueError, match="stable_hash"):
        run_closed_loop_simulation(config)


@pytest.mark.parametrize("bad_cycle_count", [True, 0, MAX_SIMULATION_CYCLES + 1, 1.5])
def test_invalid_cycle_count_rejection(bad_cycle_count: int) -> None:
    payload = {
        "axes": tuple(axis.to_dict() for axis in _axes()),
        "point_count": 4,
        "stress_method": "lattice",
        "cycle_count": bad_cycle_count,
        "candidate_orderings": tuple(item.to_dict() for item in _orderings()),
        "recurrence_window": 3,
    }
    with pytest.raises(ValueError, match="cycle_count"):
        SimulationConfig(
            axes=_axes(),
            point_count=4,
            stress_method="lattice",
            cycle_count=bad_cycle_count,  # type: ignore[arg-type]
            candidate_orderings=_orderings(),
            recurrence_window=3,
            stable_hash=sha256_hex(payload),
        )


def test_duplicate_ordering_rejection() -> None:
    order = FilterOrdering.build(("thermal",), ("ctrl",))
    with pytest.raises(ValueError, match="duplicate ordering signatures"):
        SimulationConfig(
            axes=_axes(),
            point_count=4,
            stress_method="halton",
            cycle_count=3,
            candidate_orderings=(order, order),
            recurrence_window=2,
            stable_hash="f" * 64,
        )


def test_immutable_artifacts() -> None:
    receipt = run_closed_loop_simulation(_config())
    with pytest.raises(FrozenInstanceError):
        receipt.summary.cycle_count = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        receipt.cycle_records[0].decision_type = "x"  # type: ignore[misc]


def test_canonical_hash_stability_reconstruct() -> None:
    receipt = run_closed_loop_simulation(_config())
    payload = json.loads(receipt.to_canonical_json())
    config_payload = payload["config"]
    reconstructed = ClosedLoopSimulationReceipt(
        config=SimulationConfig(
            axes=tuple(StressAxis(**axis) for axis in config_payload["axes"]),
            point_count=config_payload["point_count"],
            stress_method=config_payload["stress_method"],
            cycle_count=config_payload["cycle_count"],
            candidate_orderings=tuple(
                FilterOrdering(
                    input_filters=tuple(item["input_filters"]),
                    control_filters=tuple(item["control_filters"]),
                    ordering_signature=item["ordering_signature"],
                    stable_hash=item["stable_hash"],
                )
                for item in config_payload["candidate_orderings"]
            ),
            recurrence_window=config_payload["recurrence_window"],
            stable_hash=config_payload["stable_hash"],
        ),
        stress_receipt_hash=payload["stress_receipt_hash"],
        cycle_records=tuple(
            SimulationCycleRecord(
                state=FilterMeshState(**item["state"]),
                **{k: v for k, v in item.items() if k != "state"},
            )
            for item in payload["cycle_records"]
        ),
        summary=SimulationSummary(**payload["summary"]),
        stable_hash=payload["stable_hash"],
    )
    assert reconstructed.to_canonical_json() == receipt.to_canonical_json()


def test_no_external_state_dependence() -> None:
    config = _config(cycle_count=5)
    first = run_closed_loop_simulation(config)
    second = run_closed_loop_simulation(config)
    assert first.to_canonical_bytes() == second.to_canonical_bytes()
