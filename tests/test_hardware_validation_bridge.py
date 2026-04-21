from __future__ import annotations

import pytest

from qec.analysis import hardware_validation_bridge as bridge


def _stub_simulation_single_value(scenarios):
    scenario = scenarios[0]
    simulated_by_id = {
        "a": 2.0,
        "b": 4.0,
        "s1": 3.0,
    }
    latency = simulated_by_id[scenario["id"]]
    return {
        "mean_latency_baseline": latency * 2.0,
        "mean_latency_accelerated": latency,
        "mean_latency_improvement": latency,
        "mean_normalized_speedup": 2.0,
    }


def test_deterministic_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bridge, "run_neural_acceleration_simulation", _stub_simulation_single_value
    )
    scenarios = [
        {"id": "b", "nodes": ["n0", "n1"], "edges": [("n0", "n1")]},
        {"id": "a", "nodes": ["m0", "m1"], "edges": [("m0", "m1")]},
    ]
    hardware = {
        "a": {"latency": 2.1},
        "b": {"latency": 3.9},
    }

    first = bridge.run_hardware_validation_bridge(scenarios, hardware)
    second = bridge.run_hardware_validation_bridge(scenarios, hardware)

    assert first == second


def test_perfect_match_has_full_agreement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bridge, "run_neural_acceleration_simulation", _stub_simulation_single_value
    )

    result = bridge.run_hardware_validation_bridge(
        [{"id": "s1", "nodes": ["x"], "edges": []}],
        {"s1": {"latency": 3.0}},
    )

    assert result["mean_absolute_error"] == 0.0
    assert result["mean_relative_error"] == 0.0
    assert result["mean_agreement_score"] == 1.0


def test_mismatch_has_less_than_full_agreement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bridge, "run_neural_acceleration_simulation", _stub_simulation_single_value
    )

    result = bridge.run_hardware_validation_bridge(
        [{"id": "s1", "nodes": ["x"], "edges": []}],
        {"s1": {"latency": 6.0}},
    )

    assert result["mean_absolute_error"] == 3.0
    assert result["mean_agreement_score"] < 1.0


def test_missing_hardware_entry_raises_value_error() -> None:
    with pytest.raises(ValueError, match="hardware_measurements missing scenario id"):
        bridge.run_hardware_validation_bridge(
            [{"id": "s1", "nodes": ["x"], "edges": []}],
            {},
        )


def test_negative_hardware_latency_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bridge, "run_neural_acceleration_simulation", _stub_simulation_single_value
    )

    with pytest.raises(ValueError, match="hardware latency must be positive float"):
        bridge.run_hardware_validation_bridge(
            [{"id": "s1", "nodes": ["x"], "edges": []}],
            {"s1": {"latency": -1.0}},
        )
