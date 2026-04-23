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
from qec.analysis.deterministic_transition_policy import select_deterministic_transition
from qec.analysis.deterministic_stress_lattice import StressAxis, generate_stress_lattice
from qec.analysis.periodicity_structure_kernel import PeriodicityReceipt
from qec.analysis.state_conditioned_filter_mesh import FilterMeshState, FilterOrdering, score_filter_mesh


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
    axes = tuple(sorted(_axes(), key=lambda axis: axis.name))
    candidate_orderings = tuple(sorted(_orderings(), key=lambda o: (o.ordering_signature, o.stable_hash)))
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
    expected_stable = sum(
        1 for record in receipt.cycle_records if record.transition_classification == "stable_transition"
    )
    assert summary.stable_transition_count + summary.uncertain_transition_count == summary.cycle_count
    assert summary.stable_transition_count == expected_stable
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


def test_canonical_config_hashing_for_different_input_ordering() -> None:
    axes = _axes()
    orderings = _orderings()
    reversed_axes = tuple(reversed(axes))
    reordered_orderings = (orderings[2], orderings[0], orderings[1])
    config_a = SimulationConfig(
        axes=axes,
        point_count=4,
        stress_method="lattice",
        cycle_count=8,
        candidate_orderings=orderings,
        recurrence_window=6,
        stable_hash=sha256_hex(
            {
                "axes": tuple(axis.to_dict() for axis in sorted(axes, key=lambda axis: axis.name)),
                "point_count": 4,
                "stress_method": "lattice",
                "cycle_count": 8,
                "candidate_orderings": tuple(
                    item.to_dict() for item in sorted(orderings, key=lambda o: (o.ordering_signature, o.stable_hash))
                ),
                "recurrence_window": 6,
            }
        ),
    )
    config_b = SimulationConfig(
        axes=reversed_axes,
        point_count=4,
        stress_method="lattice",
        cycle_count=8,
        candidate_orderings=reordered_orderings,
        recurrence_window=6,
        stable_hash=config_a.stable_hash,
    )
    assert config_a.stable_hash == config_b.stable_hash
    assert config_a.to_canonical_json() == config_b.to_canonical_json()


def test_periodic_recurrence_maps_to_oscillatory_for_mesh_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_detect_periodicity(trace: list[str]) -> PeriodicityReceipt:
        payload = {
            "trace_length": len(trace),
            "candidates": (),
            "dominant_period": 2,
            "dominant_confidence": 0.25,
            "classification": "weak_periodic",
        }
        return PeriodicityReceipt(stable_hash=sha256_hex(payload), **payload)

    monkeypatch.setattr(
        "qec.analysis.closed_loop_simulation_kernel.detect_periodicity",
        _fake_detect_periodicity,
    )

    config = _config(cycle_count=12, recurrence_window=2, point_count=2)
    receipt = run_closed_loop_simulation(config)
    assert receipt.summary.recurrence_classification == "weak_periodic"
    assert any(record.state.recurrence_class == "oscillatory" for record in receipt.cycle_records)

    oscillatory_records = tuple(record for record in receipt.cycle_records if record.state.recurrence_class == "oscillatory")
    assert oscillatory_records

    sample_record = oscillatory_records[0]
    point_hash_to_coords = {
        point.stable_hash: point.coordinates
        for point in generate_stress_lattice(list(config.axes), config.point_count, config.stress_method).points
    }
    coords = point_hash_to_coords[sample_record.stress_point_hash]
    periodic_state = FilterMeshState(
        invariant_class=sample_record.state.invariant_class,
        geometry_class=sample_record.state.geometry_class,
        spectral_regime=sample_record.state.spectral_regime,
        hardware_class=sample_record.state.hardware_class,
        recurrence_class="oscillatory",
        thermal_pressure=coords.get("thermal_pressure", 0.0),
        latency_drift=coords.get("latency_drift", 0.0),
        timing_skew=coords.get("timing_skew", 0.0),
        power_pressure=coords.get("power_pressure", 0.0),
        consensus_instability=sample_record.state.consensus_instability,
    )
    aperiodic_state = FilterMeshState(
        invariant_class=sample_record.state.invariant_class,
        geometry_class=sample_record.state.geometry_class,
        spectral_regime=sample_record.state.spectral_regime,
        hardware_class=sample_record.state.hardware_class,
        recurrence_class="aperiodic",
        thermal_pressure=periodic_state.thermal_pressure,
        latency_drift=periodic_state.latency_drift,
        timing_skew=periodic_state.timing_skew,
        power_pressure=periodic_state.power_pressure,
        consensus_instability=periodic_state.consensus_instability,
    )
    periodic_mesh = score_filter_mesh(periodic_state, config.candidate_orderings)
    aperiodic_mesh = score_filter_mesh(aperiodic_state, config.candidate_orderings)
    periodic_transition = select_deterministic_transition(periodic_mesh)
    aperiodic_transition = select_deterministic_transition(aperiodic_mesh)
    assert periodic_transition.stable_hash != aperiodic_transition.stable_hash


@pytest.mark.parametrize(
    ("field_name", "field_value", "error_pattern"),
    (
        ("decision_type", "not_valid", "invalid decision_type"),
        ("refinement_classification", "not_valid", "invalid refinement_classification"),
        ("transition_classification", "not_valid", "invalid transition_classification"),
    ),
)
def test_simulation_cycle_record_rejects_invalid_enum_fields(
    field_name: str, field_value: str, error_pattern: str
) -> None:
    base_payload = {
        "cycle_index": 0,
        "stress_point_hash": "0" * 64,
        "state": FilterMeshState(
            invariant_class="thermal_dominant",
            geometry_class="thermal_balanced",
            spectral_regime="mixed_band",
            hardware_class="stable",
            recurrence_class="aperiodic",
            thermal_pressure=0.2,
            latency_drift=0.2,
            timing_skew=0.2,
            power_pressure=0.2,
            consensus_instability=0.2,
        ),
        "filter_mesh_receipt_hash": "1" * 64,
        "transition_policy_receipt_hash": "2" * 64,
        "refinement_receipt_hash": "3" * 64,
        "dominant_ordering_signature": "sig",
        "decision_type": "tie_break",
        "transition_classification": "uncertain_transition",
        "refinement_classification": "bounded",
        "convergence_metric": 0.5,
    }
    payload = {**base_payload, field_name: field_value}
    payload["stable_hash"] = sha256_hex({k: v.to_dict() if k == "state" else v for k, v in payload.items()})
    with pytest.raises(ValueError, match=error_pattern):
        SimulationCycleRecord(**payload)
