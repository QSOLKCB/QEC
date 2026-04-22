from __future__ import annotations

import math

import pytest

from qec.analysis.adaptive_thermal_control_kernel import (
    ThermalNodeSignal,
    ThermalPolicy,
    evaluate_adaptive_thermal_control,
)


def _policy() -> ThermalPolicy:
    return ThermalPolicy(
        warning_margin_c=3.0,
        critical_margin_c=10.0,
        max_cooling_delta=0.9,
        max_workload_derate=0.8,
        hotspot_weight=0.5,
        drift_weight=0.3,
        utilization_weight=0.2,
    )


def _signal(node_id: str, temp: float, delta: float, util: float) -> ThermalNodeSignal:
    return ThermalNodeSignal(
        node_id=node_id,
        temperature_c=temp,
        target_temperature_c=60.0,
        max_safe_temperature_c=90.0,
        temperature_delta_c=delta,
        utilization=util,
        power_draw_w=200.0,
        throttle_active=False,
    )


def test_deterministic_replay_identical_payload_and_hash() -> None:
    policy = _policy()
    signals = (
        _signal("n2", 70.0, 4.0, 0.5),
        _signal("n1", 62.0, 0.5, 0.2),
    )

    r1 = evaluate_adaptive_thermal_control(signals, policy)
    r2 = evaluate_adaptive_thermal_control(signals, policy)

    assert r1.to_canonical_json() == r2.to_canonical_json()
    assert r1.stable_hash == r2.stable_hash


def test_ordering_shuffled_input_yields_identical_sorted_output() -> None:
    policy = _policy()
    forward = (
        _signal("n1", 62.0, 1.0, 0.2),
        _signal("n2", 74.0, 6.0, 0.8),
    )
    reverse = tuple(reversed(forward))

    r1 = evaluate_adaptive_thermal_control(forward, policy)
    r2 = evaluate_adaptive_thermal_control(reverse, policy)

    assert tuple(d.node_id for d in r1.node_decisions) == ("n1", "n2")
    assert r1.to_canonical_json() == r2.to_canonical_json()
    assert r1.stable_hash == r2.stable_hash


def test_all_outputs_bounded_to_unit_interval() -> None:
    policy = _policy()
    receipt = evaluate_adaptive_thermal_control(
        (
            _signal("cold", 45.0, -2.0, 0.0),
            _signal("hot", 120.0, 30.0, 1.0),
        ),
        policy,
    )

    assert 0.0 <= receipt.mesh_thermal_pressure <= 1.0
    assert 0.0 <= receipt.mesh_stability_score <= 1.0
    for decision in receipt.node_decisions:
        assert 0.0 <= decision.thermal_pressure <= 1.0
        assert 0.0 <= decision.cooling_bias <= 1.0
        assert 0.0 <= decision.workload_derate <= 1.0
        assert 0.0 <= decision.stability_score <= 1.0


def test_classification_low_is_hold_high_is_critical() -> None:
    policy = _policy()
    receipt = evaluate_adaptive_thermal_control(
        (
            _signal("low", 60.0, 0.0, 0.0),
            _signal("high", 90.0, 10.0, 1.0),
        ),
        policy,
    )
    labels = {d.node_id: d.action_label for d in receipt.node_decisions}

    assert labels["low"] == "hold"
    assert labels["high"] == "critical"


def test_validation_nan_inf_and_invalid_bounds_raise_value_error() -> None:
    with pytest.raises(ValueError):
        ThermalNodeSignal(
            node_id="bad",
            temperature_c=math.nan,
            target_temperature_c=60.0,
            max_safe_temperature_c=90.0,
            temperature_delta_c=0.0,
            utilization=0.5,
            power_draw_w=100.0,
            throttle_active=False,
        )

    with pytest.raises(ValueError):
        ThermalNodeSignal(
            node_id="bad2",
            temperature_c=70.0,
            target_temperature_c=70.0,
            max_safe_temperature_c=70.0,
            temperature_delta_c=0.0,
            utilization=0.5,
            power_draw_w=100.0,
            throttle_active=False,
        )

    with pytest.raises(ValueError):
        ThermalPolicy(
            warning_margin_c=2.0,
            critical_margin_c=float("inf"),
            max_cooling_delta=0.5,
            max_workload_derate=0.5,
            hotspot_weight=0.5,
            drift_weight=0.4,
            utilization_weight=0.1,
        )

    with pytest.raises(ValueError):
        ThermalPolicy(
            warning_margin_c=2.0,
            critical_margin_c=10.0,
            max_cooling_delta=1.2,
            max_workload_derate=0.5,
            hotspot_weight=0.5,
            drift_weight=0.4,
            utilization_weight=0.1,
        )


def test_stability_score_is_one_minus_thermal_pressure() -> None:
    policy = _policy()
    receipt = evaluate_adaptive_thermal_control((
        _signal("n1", 75.0, 4.0, 0.7),
    ), policy)

    decision = receipt.node_decisions[0]
    assert decision.stability_score == pytest.approx(1.0 - decision.thermal_pressure)


def test_hash_stability_repeated_runs_identical() -> None:
    policy = _policy()
    signals = (
        _signal("a", 64.0, 1.0, 0.3),
        _signal("b", 71.0, 4.0, 0.6),
    )

    hashes = [evaluate_adaptive_thermal_control(signals, policy).stable_hash for _ in range(5)]
    assert len(set(hashes)) == 1
