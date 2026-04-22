from __future__ import annotations

import math

import pytest

from qec.analysis.adaptive_thermal_control_kernel import (
    ThermalNodeSignal,
    ThermalPolicy,
    evaluate_adaptive_thermal_control,
)
from qec.analysis.distributed_timing_mesh import (
    NodeTimingState,
    TimingMeshInputs,
    evaluate_distributed_timing_mesh,
)
from qec.analysis.latency_stabilization_loop import (
    LatencyNodeSignal,
    LatencyPolicy,
    run_latency_stabilization_loop,
)
from qec.analysis.power_aware_control_modulation import (
    POWER_AWARE_CONTROL_MODULATION_VERSION,
    PowerControlInputs,
    PowerNodeSignal,
    evaluate_power_aware_control_modulation,
)


def _latency_receipt(*, pressure_multiplier: float = 1.0):
    signals = (
        LatencyNodeSignal(
            node_id="n1",
            latency_ms=10.0 * pressure_multiplier,
            target_latency_ms=8.0,
            max_acceptable_latency_ms=20.0,
            latency_delta_ms=2.0 * pressure_multiplier,
            jitter_ms=20.0 * pressure_multiplier,
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
            utilization=1.0,
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


def _timing_receipt(*, latency_mult: float = 1.0, thermal_mult: float = 1.0):
    inputs = TimingMeshInputs(
        node_states=(
            NodeTimingState(
                node_id="n1",
                clock_offset_ms=50.0 * latency_mult,
                clock_drift_ms=10.0 * thermal_mult,
                last_sync_error_ms=20.0 * max(latency_mult, thermal_mult),
            ),
        ),
        latency_receipt=_latency_receipt(pressure_multiplier=latency_mult),
        thermal_receipt=_thermal_receipt(pressure_multiplier=thermal_mult),
    )
    return evaluate_distributed_timing_mesh(inputs)


def _inputs(
    node_power: tuple[PowerNodeSignal, ...],
    *,
    latency_mult: float = 1.0,
    thermal_mult: float = 1.0,
):
    return PowerControlInputs(
        node_power=node_power,
        thermal_receipt=_thermal_receipt(pressure_multiplier=thermal_mult),
        latency_receipt=_latency_receipt(pressure_multiplier=latency_mult),
        timing_receipt=_timing_receipt(latency_mult=latency_mult, thermal_mult=thermal_mult),
    )


def _node(
    node_id: str,
    *,
    power_draw_w: float = 80.0,
    max_power_capacity_w: float = 100.0,
    power_delta_w: float = 5.0,
    utilization: float = 0.5,
) -> PowerNodeSignal:
    return PowerNodeSignal(
        node_id=node_id,
        power_draw_w=power_draw_w,
        max_power_capacity_w=max_power_capacity_w,
        power_delta_w=power_delta_w,
        utilization=utilization,
    )


def test_deterministic_replay_identical_json_and_hash() -> None:
    inputs = _inputs((_node("n1"), _node("n2", power_draw_w=90.0, utilization=0.7)))

    first = evaluate_power_aware_control_modulation(inputs)
    second = evaluate_power_aware_control_modulation(inputs)

    assert first.to_canonical_json() == second.to_canonical_json()
    assert first.stable_hash == second.stable_hash


def test_ordering_shuffled_node_power_identical_output() -> None:
    ordered = _inputs((_node("a"), _node("b"), _node("c")))
    shuffled = _inputs((_node("c"), _node("a"), _node("b")))

    r1 = evaluate_power_aware_control_modulation(ordered)
    r2 = evaluate_power_aware_control_modulation(shuffled)

    assert r1.to_canonical_json() == r2.to_canonical_json()
    assert tuple(decision.node_id for decision in r1.node_decisions) == ("a", "b", "c")


def test_bounded_outputs_in_unit_interval() -> None:
    receipt = evaluate_power_aware_control_modulation(
        _inputs(
            (
                _node("n1", power_draw_w=10000.0, power_delta_w=1.0e6, utilization=1.0),
                _node("n2", power_draw_w=0.0, power_delta_w=-1.0e6, utilization=0.0),
            )
        )
    )

    for node in receipt.node_decisions:
        assert 0.0 <= node.power_pressure <= 1.0
        assert 0.0 <= node.load_balance_score <= 1.0
        assert 0.0 <= node.modulation_strength <= 1.0
        assert 0.0 <= node.efficiency_score <= 1.0

    assert 0.0 <= receipt.mesh_power_pressure <= 1.0
    assert 0.0 <= receipt.mesh_efficiency_score <= 1.0


def test_classification_low_stable_and_high_critical() -> None:
    low = evaluate_power_aware_control_modulation(
        _inputs((_node("n-low", power_draw_w=0.0, power_delta_w=0.0, utilization=0.0),), latency_mult=0.0, thermal_mult=0.0)
    )
    high = evaluate_power_aware_control_modulation(
        _inputs((_node("n-high", power_draw_w=1.0e6, power_delta_w=1.0e6, utilization=1.0),), latency_mult=10.0, thermal_mult=10.0)
    )

    assert low.node_decisions[0].action_label == "stable"
    assert high.node_decisions[0].action_label == "critical"


def test_cross_signal_influence_increases_power_pressure() -> None:
    node = _node("n1", power_draw_w=70.0, power_delta_w=4.0, utilization=0.4)
    low_pressure = evaluate_power_aware_control_modulation(_inputs((node,), latency_mult=0.0, thermal_mult=0.0))
    high_pressure = evaluate_power_aware_control_modulation(_inputs((node,), latency_mult=5.0, thermal_mult=5.0))

    assert high_pressure.node_decisions[0].power_pressure > low_pressure.node_decisions[0].power_pressure


def test_power_excess_increases_pressure() -> None:
    mild_overload = evaluate_power_aware_control_modulation(
        _inputs((_node("n1", power_draw_w=105.0, max_power_capacity_w=100.0, power_delta_w=0.0, utilization=0.3),))
    )
    high_overload = evaluate_power_aware_control_modulation(
        _inputs((_node("n1", power_draw_w=200.0, max_power_capacity_w=100.0, power_delta_w=0.0, utilization=0.3),))
    )

    assert high_overload.node_decisions[0].power_pressure > mild_overload.node_decisions[0].power_pressure


def test_negative_delta_does_not_increase_pressure() -> None:
    rising = evaluate_power_aware_control_modulation(
        _inputs((_node("n1", power_draw_w=80.0, power_delta_w=50.0, utilization=0.4),))
    )
    falling = evaluate_power_aware_control_modulation(
        _inputs((_node("n1", power_draw_w=80.0, power_delta_w=-50.0, utilization=0.4),))
    )

    assert rising.node_decisions[0].power_pressure > falling.node_decisions[0].power_pressure


def test_duplicate_node_id_rejection() -> None:
    with pytest.raises(ValueError, match="duplicate node_id"):
        PowerControlInputs(
            node_power=(_node("dup"), _node("dup", utilization=0.6)),
            thermal_receipt=_thermal_receipt(),
            latency_receipt=_latency_receipt(),
            timing_receipt=_timing_receipt(),
        )


def test_validation_nan_inf_and_invalid_bounds_raise_value_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        _node("n1", power_draw_w=math.nan)
    with pytest.raises(ValueError, match="finite"):
        _node("n1", power_draw_w=math.inf)
    with pytest.raises(ValueError, match=r"in \[0,1\]"):
        _node("n1", utilization=1.1)
    with pytest.raises(ValueError, match="must be > 0"):
        _node("n1", max_power_capacity_w=0.0)


def test_extreme_overload_clamps_and_full_modulation() -> None:
    receipt = evaluate_power_aware_control_modulation(
        _inputs((_node("n1", power_draw_w=1.0e9, power_delta_w=1.0e9, utilization=1.0),), latency_mult=10.0, thermal_mult=10.0)
    )
    decision = receipt.node_decisions[0]

    assert decision.power_pressure == 1.0
    assert decision.modulation_strength == 1.0
    assert decision.action_label == "critical"


def test_efficiency_behavior_high_utilization_and_high_power_is_low() -> None:
    low_efficiency = evaluate_power_aware_control_modulation(
        _inputs((_node("n1", power_draw_w=100.0, power_delta_w=100.0, utilization=1.0),))
    )
    decision = low_efficiency.node_decisions[0]

    assert decision.efficiency_score < 0.2


def test_hash_stability_repeated_runs_identical() -> None:
    inputs = _inputs((_node("n1"), _node("n2", power_draw_w=92.0, utilization=0.83)))

    hashes = [evaluate_power_aware_control_modulation(inputs).stable_hash for _ in range(5)]
    assert len(set(hashes)) == 1


def test_constants_and_mode_are_fixed() -> None:
    receipt = evaluate_power_aware_control_modulation(_inputs((_node("n1"),)))

    assert receipt.version == POWER_AWARE_CONTROL_MODULATION_VERSION
    assert receipt.control_mode == "power_advisory"
    assert receipt.observatory_only is True
