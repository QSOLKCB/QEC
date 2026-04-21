"""Tests for v138.8.4 noise-tailoring hardware bridge."""

from __future__ import annotations

import math

import pytest

from qec.analysis.noise_tailoring_hardware_bridge import (
    compare_tailored_to_hardware,
    run_noise_tailoring_hardware_bridge,
)
from qec.analysis.noise_tailoring_pulse_runtime import derive_noise_control_signal


def _bridge_config() -> dict[str, float]:
    return {
        "node_mitigation_strength": 0.5,
        "edge_mitigation_strength": 0.25,
        "regime_bias_strength": 1.0,
        "max_mean_node_error": 1.0,
        "max_mean_edge_error": 1.0,
    }


def _projected_result(scenario_id: str = "s1") -> dict[str, object]:
    return {
        "id": scenario_id,
        "regime": "S2",
        "nodes": ["a", "b"],
        "edges": [("a", "b")],
        "node_weights": {"a": 0.8, "b": 0.4},
        "edge_weights": {"a->b": 0.6},
    }


def test_deterministic_replay_is_stable():
    projected = [_projected_result()]
    tailored = derive_noise_control_signal(projected[0], _bridge_config())
    hardware = {
        "s1": {
            "mean_node_weight_after": sum(tailored["node_weights"].values()) / 2.0,
            "mean_edge_weight_after": tailored["edge_weights"]["a->b"],
        }
    }

    out1 = run_noise_tailoring_hardware_bridge(projected, hardware, _bridge_config())
    out2 = run_noise_tailoring_hardware_bridge(projected, hardware, _bridge_config())

    assert out1 == out2


def test_perfect_match_has_zero_error_and_full_agreement():
    projected = _projected_result()
    tailored = derive_noise_control_signal(projected, _bridge_config())
    hardware = {
        "mean_node_weight_after": sum(tailored["node_weights"].values()) / 2.0,
        "mean_edge_weight_after": tailored["edge_weights"]["a->b"],
    }

    compared = compare_tailored_to_hardware(tailored, hardware)

    assert compared["node_error"] == 0.0
    assert compared["edge_error"] == 0.0
    assert compared["node_agreement"] == 1.0
    assert compared["edge_agreement"] == 1.0


def test_mismatch_case_has_positive_error_and_subunit_agreement():
    projected = _projected_result()
    tailored = derive_noise_control_signal(projected, _bridge_config())
    hardware = {
        "mean_node_weight_after": 0.0,
        "mean_edge_weight_after": 0.0,
    }

    compared = compare_tailored_to_hardware(tailored, hardware)

    assert compared["node_error"] > 0.0
    assert compared["edge_error"] > 0.0
    assert compared["node_agreement"] < 1.0
    assert compared["edge_agreement"] < 1.0


def test_missing_hardware_entry_raises_value_error():
    with pytest.raises(ValueError, match="missing scenario id"):
        run_noise_tailoring_hardware_bridge([_projected_result()], {}, _bridge_config())


def test_non_finite_hardware_value_raises_value_error():
    hardware = {
        "s1": {
            "mean_node_weight_after": math.inf,
            "mean_edge_weight_after": 0.1,
        }
    }
    with pytest.raises(ValueError, match="finite numeric"):
        run_noise_tailoring_hardware_bridge([_projected_result()], hardware, _bridge_config())


def test_negative_hardware_value_raises_value_error():
    hardware = {
        "s1": {
            "mean_node_weight_after": -0.1,
            "mean_edge_weight_after": 0.1,
        }
    }
    with pytest.raises(ValueError, match=">= 0"):
        run_noise_tailoring_hardware_bridge([_projected_result()], hardware, _bridge_config())


def test_duplicate_projected_result_ids_raise_value_error():
    projected = [_projected_result("dup"), _projected_result("dup")]
    hardware = {
        "dup": {
            "mean_node_weight_after": 0.1,
            "mean_edge_weight_after": 0.1,
        }
    }
    with pytest.raises(ValueError, match="duplicate scenario ids"):
        run_noise_tailoring_hardware_bridge(projected, hardware, _bridge_config())


def test_aggregate_correctness_fixed_scenarios():
    config = {
        "node_mitigation_strength": 0.0,
        "edge_mitigation_strength": 0.0,
        "regime_bias_strength": 1.0,
        "max_mean_node_error": 10.0,
        "max_mean_edge_error": 10.0,
    }
    projected = [
        {
            "id": "b",
            "regime": "S2",
            "nodes": ["y"],
            "edges": [],
            "node_weights": {"y": 0.6},
            "edge_weights": {},
        },
        {
            "id": "a",
            "regime": "S1",
            "nodes": ["x1", "x2"],
            "edges": [("x1", "x2")],
            "node_weights": {"x1": 0.2, "x2": 0.4},
            "edge_weights": {"x1->x2": 0.5},
        },
    ]
    hardware = {
        "a": {
            "mean_node_weight_after": 0.4,
            "mean_edge_weight_after": 0.4,
        },
        "b": {
            "mean_node_weight_after": 0.2,
            "mean_edge_weight_after": 0.3,
        },
    }

    out = run_noise_tailoring_hardware_bridge(projected, hardware, config)

    assert out == {
        "mean_node_error": 0.25,
        "mean_edge_error": 0.2,
        "mean_node_agreement": 0.811688311688,
        "mean_edge_agreement": 0.839160839161,
    }
