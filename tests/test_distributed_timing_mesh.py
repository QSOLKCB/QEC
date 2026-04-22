from __future__ import annotations

import math

import pytest

from qec.analysis.adaptive_thermal_control_kernel import (
    ThermalNodeSignal,
    ThermalPolicy,
    evaluate_adaptive_thermal_control,
)
from qec.analysis.distributed_timing_mesh import (
    MAX_DRIFT_MS,
    MAX_OFFSET_MS,
    DISTRIBUTED_TIMING_MESH_VERSION,
    NodeTimingState,
    TimingMeshInputs,
    evaluate_distributed_timing_mesh,
)
from qec.analysis.latency_stabilization_loop import (
    LatencyNodeSignal,
    LatencyPolicy,
    run_latency_stabilization_loop,
)


def _node_state(
    node_id: str,
    *,
    clock_offset_ms: float = 2.0,
    clock_drift_ms: float = 0.2,
    last_sync_error_ms: float = 0.1,
) -> NodeTimingState:
    return NodeTimingState(
        node_id=node_id,
        clock_offset_ms=clock_offset_ms,
        clock_drift_ms=clock_drift_ms,
        last_sync_error_ms=last_sync_error_ms,
    )


def _latency_receipt(*, pressure_multiplier: float = 1.0):
    signals = (
        LatencyNodeSignal(
            node_id="n1",
            latency_ms=10.0 * pressure_multiplier,
            target_latency_ms=8.0,
            max_acceptable_latency_ms=20.0,
            latency_delta_ms=2.0 * pressure_multiplier,
            jitter_ms=1.0 * pressure_multiplier,
            utilization=0.7,
        ),
    )
    policy = LatencyPolicy(
        jitter_weight=0.5,
        drift_weight=0.3,
        utilization_weight=0.2,
        max_correction_strength=1.0,
    )
    return run_latency_stabilization_loop(signals, policy)


def _thermal_receipt(*, pressure_multiplier: float = 1.0):
    signals = (
        ThermalNodeSignal(
            node_id="n1",
            temperature_c=65.0 + 20.0 * pressure_multiplier,
            target_temperature_c=60.0,
            max_safe_temperature_c=90.0,
            temperature_delta_c=2.0 * pressure_multiplier,
            utilization=0.8,
            power_draw_w=200.0,
            throttle_active=False,
        ),
    )
    policy = ThermalPolicy(
        critical_margin_c=10.0,
        max_cooling_delta=0.9,
        max_workload_derate=0.8,
        hotspot_weight=0.5,
        drift_weight=0.3,
        utilization_weight=0.2,
    )
    return evaluate_adaptive_thermal_control(signals, policy)


def _inputs(
    node_states: tuple[NodeTimingState, ...],
    *,
    latency_mult: float = 1.0,
    thermal_mult: float = 1.0,
) -> TimingMeshInputs:
    return TimingMeshInputs(
        node_states=node_states,
        latency_receipt=_latency_receipt(pressure_multiplier=latency_mult),
        thermal_receipt=_thermal_receipt(pressure_multiplier=thermal_mult),
    )


def test_deterministic_replay_identical_json_and_hash() -> None:
    inputs = _inputs((_node_state("n1"), _node_state("n2", clock_offset_ms=4.0)))

    first = evaluate_distributed_timing_mesh(inputs)
    second = evaluate_distributed_timing_mesh(inputs)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_ordering_shuffled_node_states_identical_output() -> None:
    ordered = _inputs((_node_state("a"), _node_state("b"), _node_state("c")))
    shuffled = _inputs((_node_state("c"), _node_state("a"), _node_state("b")))

    r1 = evaluate_distributed_timing_mesh(ordered)
    r2 = evaluate_distributed_timing_mesh(shuffled)

    assert r1.to_canonical_json() == r2.to_canonical_json()
    assert tuple(decision.node_id for decision in r1.node_decisions) == ("a", "b", "c")




def test_inputs_hash_permutation_invariant() -> None:
    a = _node_state("a", clock_offset_ms=1.2, clock_drift_ms=0.3, last_sync_error_ms=0.05)
    b = _node_state("b", clock_offset_ms=-2.4, clock_drift_ms=0.6, last_sync_error_ms=0.10)
    c = _node_state("c", clock_offset_ms=3.6, clock_drift_ms=0.9, last_sync_error_ms=0.15)

    inputs_a = _inputs((a, b, c))
    inputs_b = _inputs((c, a, b))

    assert inputs_a.to_canonical_json() == inputs_b.to_canonical_json()
    assert inputs_a.stable_hash() == inputs_b.stable_hash()

def test_bounded_outputs_in_unit_interval() -> None:
    receipt = evaluate_distributed_timing_mesh(
        _inputs(
            (
                _node_state("n1", clock_offset_ms=1000.0, clock_drift_ms=1000.0, last_sync_error_ms=1000.0),
                _node_state("n2", clock_offset_ms=-1000.0, clock_drift_ms=-1000.0, last_sync_error_ms=-1000.0),
            )
        )
    )

    for node in receipt.node_decisions:
        assert 0.0 <= node.timing_drift <= 1.0
        assert 0.0 <= node.alignment_error <= 1.0

    assert 0.0 <= receipt.mesh_timing_drift <= 1.0
    assert 0.0 <= receipt.mesh_alignment_error <= 1.0
    assert 0.0 <= receipt.synchronization_confidence <= 1.0
    assert 0.0 <= receipt.mesh_stability <= 1.0


def test_classification_low_stable_and_high_resync() -> None:
    low = evaluate_distributed_timing_mesh(
        _inputs((_node_state("n-low", clock_offset_ms=0.0, clock_drift_ms=0.0, last_sync_error_ms=0.0),), latency_mult=0.0, thermal_mult=0.0)
    )
    high = evaluate_distributed_timing_mesh(
        _inputs(
            (_node_state("n-high", clock_offset_ms=MAX_OFFSET_MS * 10.0, clock_drift_ms=MAX_DRIFT_MS * 10.0, last_sync_error_ms=1.0e6),),
            latency_mult=10.0,
            thermal_mult=10.0,
        )
    )

    assert low.node_decisions[0].action_label == "stable"
    assert high.node_decisions[0].action_label == "resync"


def test_cross_signal_influence_increases_timing_drift() -> None:
    node = _node_state("n1", clock_offset_ms=2.0, clock_drift_ms=0.5, last_sync_error_ms=0.3)
    low_pressure = evaluate_distributed_timing_mesh(_inputs((node,), latency_mult=0.0, thermal_mult=0.0))
    high_pressure = evaluate_distributed_timing_mesh(_inputs((node,), latency_mult=4.0, thermal_mult=4.0))

    assert high_pressure.node_decisions[0].timing_drift > low_pressure.node_decisions[0].timing_drift


def test_duplicate_node_id_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate node_id"):
        TimingMeshInputs(
            node_states=(_node_state("dup"), _node_state("dup")),
            latency_receipt=_latency_receipt(),
            thermal_receipt=_thermal_receipt(),
        )


def test_validation_nan_inf_and_invalid_types_raise_value_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        _node_state("bad", clock_offset_ms=math.nan)
    with pytest.raises(ValueError, match="finite"):
        _node_state("bad", clock_drift_ms=math.inf)
    with pytest.raises(ValueError, match="tuple"):
        TimingMeshInputs(  # type: ignore[arg-type]
            node_states=[_node_state("n1")],
            latency_receipt=_latency_receipt(),
            thermal_receipt=_thermal_receipt(),
        )
    with pytest.raises(ValueError, match="LatencyControlReceipt"):
        TimingMeshInputs(  # type: ignore[arg-type]
            node_states=(_node_state("n1"),),
            latency_receipt=object(),
            thermal_receipt=_thermal_receipt(),
        )


def test_extreme_offset_correction_scales_with_timing_drift() -> None:
    node = _node_state("n1", clock_offset_ms=MAX_OFFSET_MS * 20.0, clock_drift_ms=MAX_DRIFT_MS * 20.0, last_sync_error_ms=1.0e6)
    receipt = evaluate_distributed_timing_mesh(_inputs((node,), latency_mult=10.0, thermal_mult=10.0))

    decision = receipt.node_decisions[0]
    assert decision.timing_drift > 0.99
    assert decision.correction_offset_ms == pytest.approx(-node.clock_offset_ms * decision.timing_drift)


def test_hash_stability_repeated_runs_identical() -> None:
    inputs = _inputs((_node_state("n1"), _node_state("n2", clock_offset_ms=-3.5)))

    hashes = [evaluate_distributed_timing_mesh(inputs).stable_hash for _ in range(5)]
    assert len(set(hashes)) == 1


def test_constants_and_mode_are_fixed() -> None:
    receipt = evaluate_distributed_timing_mesh(_inputs((_node_state("n1"),)))

    assert receipt.version == DISTRIBUTED_TIMING_MESH_VERSION
    assert receipt.control_mode == "timing_mesh_advisory"
    assert receipt.observatory_only is True
