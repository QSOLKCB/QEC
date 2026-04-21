"""Tests for v138.8.3 noise-tailoring pulse runtime."""

from __future__ import annotations

import math

import pytest

from qec.analysis.noise_tailoring_pulse_runtime import (
    derive_noise_control_signal,
    run_noise_tailoring_runtime,
)


def _base_config() -> dict[str, float]:
    return {
        "node_mitigation_strength": 0.5,
        "edge_mitigation_strength": 0.25,
        "regime_bias_strength": 1.0,
    }


def _base_projected() -> dict[str, object]:
    return {
        "id": "p1",
        "regime": "S2",
        "nodes": ["a", "b"],
        "edges": [("a", "b")],
        "node_weights": {"a": 0.8, "b": 0.4},
        "edge_weights": {"a->b": 0.6},
    }


def test_deterministic_replay_runtime_output_is_stable():
    scenarios = [
        {
            "id": "b",
            "regime": "S2",
            "nodes": ["n2", "n1"],
            "edges": [("n1", "n2")],
            "node_weights": {"n1": 0.7, "n2": 0.3},
            "edge_weights": {"n1->n2": 0.5},
        },
        {
            "id": "a",
            "regime": "S1",
            "nodes": ["x"],
            "edges": [],
            "node_weights": {"x": 0.2},
            "edge_weights": {},
        },
    ]

    out1 = run_noise_tailoring_runtime(scenarios, _base_config())
    out2 = run_noise_tailoring_runtime(scenarios, _base_config())

    assert out1 == out2


def test_mitigation_reduces_weights():
    projected = _base_projected()
    out = derive_noise_control_signal(projected, _base_config())

    assert out["node_weights"]["a"] < projected["node_weights"]["a"]
    assert out["node_weights"]["b"] < projected["node_weights"]["b"]
    assert out["edge_weights"]["a->b"] < projected["edge_weights"]["a->b"]


def test_s3_produces_stronger_reduction_than_s1():
    config = _base_config()
    s1 = _base_projected()
    s1["id"] = "s1"
    s1["regime"] = "S1"

    s3 = _base_projected()
    s3["id"] = "s3"
    s3["regime"] = "S3"

    out_s1 = derive_noise_control_signal(s1, config)
    out_s3 = derive_noise_control_signal(s3, config)

    s1_reduction = s1["node_weights"]["a"] - out_s1["node_weights"]["a"]
    s3_reduction = s3["node_weights"]["a"] - out_s3["node_weights"]["a"]
    assert s3_reduction > s1_reduction


def test_clamp_to_zero_behavior():
    projected = _base_projected()
    projected["node_weights"] = {"a": 5.0, "b": 4.0}
    projected["edge_weights"] = {"a->b": 5.0}

    config = {
        "node_mitigation_strength": 1.0,
        "edge_mitigation_strength": 1.0,
        "regime_bias_strength": 1.0,
    }
    projected["regime"] = "S3"

    out = derive_noise_control_signal(projected, config)
    assert out["node_weights"]["a"] == 0.0
    assert out["node_weights"]["b"] == 0.0
    assert out["edge_weights"]["a->b"] == 0.0


def test_outputs_are_finite():
    scenarios = [_base_projected()]
    aggregate = run_noise_tailoring_runtime(scenarios, _base_config())
    single = derive_noise_control_signal(_base_projected(), _base_config())

    assert all(math.isfinite(value) for value in aggregate.values())
    assert all(math.isfinite(value) for value in single["node_weights"].values())
    assert all(math.isfinite(value) for value in single["edge_weights"].values())


def test_malformed_input_raises_value_error():
    malformed = {
        "id": "bad",
        "regime": "SX",
        "nodes": ["a"],
        "edges": [],
        "node_weights": {"a": 1.0},
        "edge_weights": {},
    }

    with pytest.raises(ValueError, match="regime"):
        derive_noise_control_signal(malformed, _base_config())


def test_invalid_config_raises_value_error():
    bad_config = {
        "node_mitigation_strength": 1.1,
        "edge_mitigation_strength": 0.2,
        "regime_bias_strength": 1.0,
    }

    with pytest.raises(ValueError, match="<= 1.0"):
        derive_noise_control_signal(_base_projected(), bad_config)
