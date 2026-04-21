"""Tests for v138.8.1 cluster expansion noise kernel."""

from __future__ import annotations

import math

import pytest

from qec.analysis.cluster_expansion_noise_kernel import (
    apply_cluster_expansion_noise,
    run_cluster_expansion_simulation,
)
from qec.analysis.correlated_noise_simulator import apply_su3_correlated_noise


def _base_config(pairwise: float = 0.2) -> dict[str, float]:
    return {
        "local_contrast_noise_scale": 0.2,
        "global_drift_noise_scale": 0.1,
        "pairwise_correlation_scale": pairwise,
        "stable_threshold": 2.0,
        "relaxation_threshold": 6.0,
    }


def test_deterministic_replay_identical_outputs():
    cfg = _base_config(pairwise=0.3)
    scenarios = [
        {"id": "b", "nodes": ["n2", "n1"], "edges": [("n1", "n2")]},
        {"id": "a", "nodes": ["x", "z", "y"], "edges": [("x", "y"), ("x", "z")]},
    ]

    out1 = run_cluster_expansion_simulation(scenarios, cfg)
    out2 = run_cluster_expansion_simulation(scenarios, cfg)
    assert out1 == out2


def test_pairwise_zero_matches_v138_8_0_base_model():
    cfg = _base_config(pairwise=0.0)
    scenario = {
        "id": "eq",
        "nodes": ["n2", "n1"],
        "edges": [("n1", "n2")],
        "node_weights": {"n1": 2.0, "n2": 3.0},
        "edge_weights": {"n1->n2": 4.0},
    }

    base = apply_su3_correlated_noise(scenario, {k: v for k, v in cfg.items() if k != "pairwise_correlation_scale"})
    ext = apply_cluster_expansion_noise(scenario, cfg)

    assert ext["regime"] == base["regime"]
    assert ext["node_weights"] == base["node_weights"]
    assert ext["edge_weights"] == base["edge_weights"]


def test_degree_based_node_variation_is_applied():
    cfg = _base_config(pairwise=0.25)
    scenario = {
        "id": "deg",
        "nodes": ["a", "b", "c", "d"],
        "edges": [("a", "b"), ("a", "c"), ("a", "d")],
    }

    out = apply_cluster_expansion_noise(scenario, cfg)
    # S3 base factor: (1-0.2*0.25)*(1-0.1*0.5)=0.9025
    # degree(a)=3/3 => pair=0.75 ; degree(b,c,d)=1/3 => pair=11/12
    assert out["node_weights"]["a"] == 0.676875
    assert out["node_weights"]["b"] == 0.827291666667
    assert out["node_weights"]["c"] == 0.827291666667
    assert out["node_weights"]["d"] == 0.827291666667


def test_edge_correlation_applied_uniformly():
    cfg = _base_config(pairwise=0.25)
    scenario = {
        "id": "edge",
        "nodes": ["a", "b", "c", "d"],
        "edges": [("a", "b"), ("a", "c"), ("a", "d")],
        "edge_weights": {"a->b": 2.0, "a->c": 3.0, "a->d": 4.0},
    }

    out = apply_cluster_expansion_noise(scenario, cfg)
    # S3 base factor=0.9025, edge pairwise factor=0.75 => multiplier=0.676875
    assert out["edge_weights"]["a->b"] == 1.35375
    assert out["edge_weights"]["a->c"] == 2.030625
    assert out["edge_weights"]["a->d"] == 2.7075


def test_non_negative_and_finite_outputs():
    cfg = {
        "local_contrast_noise_scale": 4.0,
        "global_drift_noise_scale": 3.0,
        "pairwise_correlation_scale": 5.0,
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

    out = apply_cluster_expansion_noise(scenario, cfg)
    for value in out["node_weights"].values():
        assert value >= 0.0
        assert math.isfinite(value)
    for value in out["edge_weights"].values():
        assert value >= 0.0
        assert math.isfinite(value)


def test_clamps_negative_weights_to_zero():
    cfg = {
        "local_contrast_noise_scale": 0.0,
        "global_drift_noise_scale": 0.0,
        "pairwise_correlation_scale": 0.0,
        "stable_threshold": 0.0,
        "relaxation_threshold": 1.0,
    }
    scenario = {
        "id": "clamp",
        "nodes": ["a", "b"],
        "edges": [("a", "b")],
        "node_weights": {"a": -1.0, "b": -2.5},
        "edge_weights": {"a->b": -3.75},
    }

    out = apply_cluster_expansion_noise(scenario, cfg)
    assert out["node_weights"]["a"] == 0.0
    assert out["node_weights"]["b"] == 0.0
    assert out["edge_weights"]["a->b"] == 0.0


def test_rounds_weights_to_12_decimal_places():
    cfg = {
        "local_contrast_noise_scale": 0.0,
        "global_drift_noise_scale": 0.0,
        "pairwise_correlation_scale": 0.0,
        "stable_threshold": 0.0,
        "relaxation_threshold": 1.0,
    }
    scenario = {
        "id": "rounding",
        "nodes": ["a", "b"],
        "edges": [("a", "b")],
        "node_weights": {"a": 1.0 / 3.0, "b": 2.0 / 3.0},
        "edge_weights": {"a->b": 2.0 / 9.0},
    }

    out = apply_cluster_expansion_noise(scenario, cfg)
    assert out["node_weights"]["a"] == round(1.0 / 3.0, 12)
    assert out["node_weights"]["b"] == round(2.0 / 3.0, 12)
    assert out["edge_weights"]["a->b"] == round(2.0 / 9.0, 12)


def test_edge_tuple_keys_prevent_simulation_collision_corruption():
    cfg = _base_config(pairwise=0.0)
    scenario = {
        "id": "collision",
        "nodes": ["a->b", "a", "b->c", "c"],
        "edges": [("a->b", "c"), ("a", "b->c")],
        "edge_weights": {"a->b->c": 5.0},
    }

    out = run_cluster_expansion_simulation([scenario], cfg)
    # Under S3 at pairwise=0, multiplier=(1-0.2*0.25)*(1-0.1*0.5)=0.9025:
    # default edge weight=1.0 and provided edge weight=5.0 are both included.
    assert out["mean_edge_weight"] == 2.7075


@pytest.mark.parametrize(
    ("scenario", "config", "error_match"),
    [
        (
            {"id": "dup_nodes", "nodes": ["a", "a"], "edges": []},
            _base_config(),
            "must contain unique node identifiers",
        ),
        (
            {"id": "bad_endpoint", "nodes": ["a"], "edges": [("a", "b")]},
            _base_config(),
            "all edge endpoints",
        ),
        (
            {"id": "s", "nodes": ["a"], "edges": []},
            {
                "local_contrast_noise_scale": True,
                "global_drift_noise_scale": 0.1,
                "pairwise_correlation_scale": 0.2,
                "stable_threshold": 1.0,
                "relaxation_threshold": 2.0,
            },
            "not bool",
        ),
    ],
)
def test_malformed_input_raises_value_error(scenario, config, error_match):
    with pytest.raises(ValueError, match=error_match):
        apply_cluster_expansion_noise(scenario, config)


def test_aggregation_correctness_expected_values():
    cfg = {
        "local_contrast_noise_scale": 0.2,
        "global_drift_noise_scale": 0.1,
        "pairwise_correlation_scale": 0.25,
        "stable_threshold": 2.0,
        "relaxation_threshold": 4.0,
    }
    scenarios = [
        {"id": "b", "nodes": ["n1"], "edges": []},  # S1, no pairwise node reduction
        {"id": "a", "nodes": ["x", "y"], "edges": [("x", "y")]},  # S2
    ]

    out = run_cluster_expansion_simulation(scenarios, cfg)

    assert out == {
        "mean_node_weight": 0.6525,
        "mean_edge_weight": 0.54,
        "mean_node_noise_delta": 0.3475,
        "mean_edge_noise_delta": 0.46,
    }
