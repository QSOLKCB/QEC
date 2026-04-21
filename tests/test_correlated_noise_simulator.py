"""Tests for v138.8.0 correlated noise simulator (SU(3-native bootstrap))."""

from __future__ import annotations

import math

import pytest

from qec.analysis.correlated_noise_simulator import (
    apply_su3_correlated_noise,
    classify_noise_regime,
    _validate_config,
    _validate_scenario,
    run_correlated_noise_simulation,
)


def _base_config() -> dict[str, float]:
    return {
        "local_contrast_noise_scale": 0.2,
        "global_drift_noise_scale": 0.1,
        "stable_threshold": 2.0,
        "relaxation_threshold": 6.0,
    }


def test_deterministic_replay_identical_outputs():
    cfg = _base_config()
    scenarios = [
        {"id": "b", "nodes": ["n2", "n1"], "edges": [("n1", "n2")]},
        {"id": "a", "nodes": ["x"], "edges": []},
    ]
    out1 = run_correlated_noise_simulation(scenarios, cfg)
    out2 = run_correlated_noise_simulation(scenarios, cfg)
    assert out1 == out2


def test_regime_classification_s1_s2_s3():
    cfg = _base_config()
    s1 = {"id": "s1", "nodes": ["a"], "edges": []}  # load=1 <= 2
    s2 = {"id": "s2", "nodes": ["a", "b"], "edges": [("a", "b")] }  # load=3
    s3 = {
        "id": "s3",
        "nodes": ["a", "b", "c", "d"],
        "edges": [("a", "b"), ("b", "c"), ("c", "d")],
    }  # load=7 >= 6
    parsed = _validate_config(cfg)
    assert classify_noise_regime(_validate_scenario(s1), parsed) == "S1"
    assert classify_noise_regime(_validate_scenario(s2), parsed) == "S2"
    assert classify_noise_regime(_validate_scenario(s3), parsed) == "S3"


def test_simple_noise_application_exact_expected_values():
    cfg = _base_config()
    scenario = {
        "id": "mid",
        "nodes": ["n1", "n2"],
        "edges": [("n1", "n2")],
        "node_weights": {"n1": 2.0, "n2": 1.5},
        "edge_weights": {"n1->n2": 4.0},
    }
    out = apply_su3_correlated_noise(scenario, cfg)
    # S2 factors: (1-0.2)*(1-0.1)=0.72
    assert out["regime"] == "S2"
    assert out["node_weights"]["n1"] == 1.44
    assert out["node_weights"]["n2"] == 1.08
    assert out["edge_weights"]["n1->n2"] == 2.88


def test_zero_noise_keeps_weights_unchanged():
    cfg = {
        "local_contrast_noise_scale": 0.0,
        "global_drift_noise_scale": 0.0,
        "stable_threshold": 1,
        "relaxation_threshold": 3,
    }
    scenario = {
        "id": "z",
        "nodes": ["n1", "n2"],
        "edges": [("n1", "n2")],
        "node_weights": {"n1": 3.0, "n2": 5.0},
        "edge_weights": {"n1->n2": 7.0},
    }
    out = apply_su3_correlated_noise(scenario, cfg)
    assert out["node_weights"]["n1"] == 3.0
    assert out["node_weights"]["n2"] == 5.0
    assert out["edge_weights"]["n1->n2"] == 7.0


def test_duplicate_scenario_ids_raise_value_error():
    cfg = _base_config()
    scenarios = [
        {"id": "dup", "nodes": ["a"], "edges": []},
        {"id": "dup", "nodes": ["b"], "edges": []},
    ]
    with pytest.raises(ValueError, match="duplicate scenario ids"):
        run_correlated_noise_simulation(scenarios, cfg)


def test_malformed_scenario_raises_value_error():
    cfg = _base_config()
    bad = {"id": "bad", "nodes": ["a"], "edges": [["a", "b"]]}  # edge must be tuple
    with pytest.raises(ValueError, match="edges"):
        apply_su3_correlated_noise(bad, cfg)


def test_invalid_config_raises_value_error():
    bad_cfg = {
        "local_contrast_noise_scale": -1.0,
        "global_drift_noise_scale": 0.0,
        "stable_threshold": 3.0,
        "relaxation_threshold": 2.0,
    }
    scenario = {"id": "s", "nodes": ["n"], "edges": []}
    with pytest.raises(ValueError, match="must be >= 0"):
        apply_su3_correlated_noise(scenario, bad_cfg)


@pytest.mark.parametrize(
    ("scenario", "error_match"),
    [
        (
            {"id": "dup_nodes", "nodes": ["a", "a"], "edges": []},
            "must contain unique node identifiers",
        ),
        (
            {"id": "bad_edge_endpoint", "nodes": ["a"], "edges": [("a", "b")]},
            "all edge endpoints must be present in 'nodes'",
        ),
        (
            {
                "id": "bad_node_weight_key",
                "nodes": ["a"],
                "edges": [],
                "node_weights": {"b": 1.0},
            },
            "keys must match declared nodes",
        ),
        (
            {"id": "dup_edges", "nodes": ["a", "b"], "edges": [("a", "b"), ("a", "b")]},
            "must not contain duplicates",
        ),
    ],
)
def test_scenario_validation_rejects_invalid_duplicates_and_keys(
    scenario: dict[str, object], error_match: str
):
    with pytest.raises(ValueError, match=error_match):
        _validate_scenario(scenario)


def test_aggregate_metrics_expected_values():
    cfg = {
        "local_contrast_noise_scale": 0.2,
        "global_drift_noise_scale": 0.1,
        "stable_threshold": 2.0,
        "relaxation_threshold": 4.0,
    }
    scenarios = [
        {"id": "b", "nodes": ["n1"], "edges": []},  # S1 -> factor 0.8775
        {"id": "a", "nodes": ["x", "y"], "edges": [("x", "y")]},  # S2 -> factor 0.72
    ]
    out = run_correlated_noise_simulation(scenarios, cfg)

    assert out == {
        "mean_node_weight": 0.7725,
        "mean_edge_weight": 0.72,
        "mean_node_noise_delta": 0.2275,
        "mean_edge_noise_delta": 0.28,
    }


def test_non_negative_and_finite_outputs():
    cfg = {
        "local_contrast_noise_scale": 4.0,
        "global_drift_noise_scale": 3.0,
        "stable_threshold": 0.0,
        "relaxation_threshold": 1.0,
    }
    scenario = {
        "id": "nn",
        "nodes": ["a", "b"],
        "edges": [("a", "b")],
        "node_weights": {"a": 2.0, "b": 1.0},
        "edge_weights": {"a->b": 5.0},
    }
    out = apply_su3_correlated_noise(scenario, cfg)

    for value in out["node_weights"].values():
        assert value >= 0.0
        assert math.isfinite(value)
    for value in out["edge_weights"].values():
        assert value >= 0.0
        assert math.isfinite(value)
