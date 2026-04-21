"""Tests for v138.8.2 honest noise approximation pack."""

from __future__ import annotations

import pytest

from qec.analysis.honest_noise_approximation import (
    _project_to_honest_noise_validated,
    _validate_config,
    _validate_noisy_result,
    project_to_honest_noise,
    run_honest_noise_projection,
)


def _base_config() -> dict[str, float]:
    return {
        "max_node_weight": 1.0,
        "max_edge_weight": 0.8,
        "min_weight": 0.1,
        "contraction_factor": 0.5,
    }


def _base_noisy_result() -> dict[str, object]:
    return {
        "id": "s1",
        "regime": "S2",
        "nodes": ["a", "b"],
        "edges": [("a", "b")],
        "node_weights": {"a": 0.8, "b": 0.6},
        "edge_weights": {"a->b": 0.7},
    }


def test_clamp_behavior_values_below_min_are_raised():
    config = {
        "max_node_weight": 2.0,
        "max_edge_weight": 2.0,
        "min_weight": 0.25,
        "contraction_factor": 1.0,
    }
    noisy = _base_noisy_result()
    noisy["node_weights"] = {"a": -3.0, "b": 0.1}
    noisy["edge_weights"] = {"a->b": -2.0}

    out = project_to_honest_noise(noisy, config)
    assert out["node_weights"]["a"] == 0.25
    assert out["node_weights"]["b"] == 0.25
    assert out["edge_weights"]["a->b"] == 0.25


def test_cap_behavior_values_above_max_are_reduced():
    config = {
        "max_node_weight": 1.1,
        "max_edge_weight": 0.9,
        "min_weight": 0.0,
        "contraction_factor": 1.0,
    }
    noisy = _base_noisy_result()
    noisy["node_weights"] = {"a": 4.0, "b": 2.0}
    noisy["edge_weights"] = {"a->b": 7.0}

    out = project_to_honest_noise(noisy, config)
    assert out["node_weights"]["a"] == 1.1
    assert out["node_weights"]["b"] == 1.1
    assert out["edge_weights"]["a->b"] == 0.9


def test_contraction_behavior_scales_weights_correctly():
    config = {
        "max_node_weight": 3.0,
        "max_edge_weight": 3.0,
        "min_weight": 0.0,
        "contraction_factor": 0.25,
    }
    noisy = _base_noisy_result()
    noisy["node_weights"] = {"a": 2.0, "b": 1.0}
    noisy["edge_weights"] = {"a->b": 0.8}

    out = project_to_honest_noise(noisy, config)
    assert out["node_weights"]["a"] == 0.5
    assert out["node_weights"]["b"] == 0.25
    assert out["edge_weights"]["a->b"] == 0.2


def test_projection_no_normalization_path_needed():
    config = {
        "max_node_weight": 1.0,
        "max_edge_weight": 1.0,
        "min_weight": 0.0,
        "contraction_factor": 1.0,
    }
    noisy = {
        "id": "norm",
        "regime": "S3",
        "nodes": ["a", "b"],
        "edges": [("a", "b"), ("b", "a")],
        "node_weights": {"a": 2.0, "b": 2.0},
        "edge_weights": {"a->b": 4.0, "b->a": 2.0},
    }

    out = project_to_honest_noise(noisy, config)
    assert out["node_weights"] == {"a": 1.0, "b": 1.0}
    assert out["edge_weights"] == {"a->b": 1.0, "b->a": 1.0}


def test_edge_key_collision_inputs_do_not_corrupt_edge_outputs():
    config = _base_config()
    noisy = {
        "id": "collision",
        "regime": "S1",
        "nodes": ["a", "a->b", "b->c", "c"],
        "edges": [("a", "b->c"), ("a->b", "c")],
        "node_weights": {"a": 1.0, "a->b": 1.0, "b->c": 1.0, "c": 1.0},
        "edge_weights": {"a->b->c": 0.7},
    }

    with pytest.raises(ValueError, match="edge_weights"):
        project_to_honest_noise(noisy, config)


def test_validated_runner_helper_matches_public_api_results():
    config = _base_config()
    scenario = _base_noisy_result()
    normalized = _validate_noisy_result(scenario)
    parsed = _validate_config(config)

    via_api = project_to_honest_noise(scenario, config)
    via_validated = _project_to_honest_noise_validated(normalized, parsed)

    assert via_validated == via_api


def test_deterministic_replay_and_aggregate_metrics():
    scenarios = [
        {
            "id": "b",
            "regime": "S2",
            "nodes": ["n2", "n1"],
            "edges": [("n1", "n2")],
            "node_weights": {"n1": 2.5, "n2": -0.3},
            "edge_weights": {"n1->n2": 1.6},
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

    out1 = run_honest_noise_projection(scenarios, _base_config())
    out2 = run_honest_noise_projection(scenarios, _base_config())

    assert out1 == out2
    assert out1 == {
        "mean_node_weight": 0.233333333333,
        "mean_edge_weight": 0.4,
        "mean_node_adjustment": -0.566666666667,
        "mean_edge_adjustment": -1.2,
    }


def test_malformed_input_raises_value_error():
    config = _base_config()
    malformed = {
        "id": "bad",
        "regime": "S2",
        "nodes": ["a"],
        "edges": [["a", "a"]],  # must be tuple
        "node_weights": {"a": 1.0},
        "edge_weights": {},
    }

    with pytest.raises(ValueError, match="edges"):
        project_to_honest_noise(malformed, config)


def test_non_finite_input_raises_value_error():
    config = _base_config()
    noisy = _base_noisy_result()
    noisy["node_weights"] = {"a": float("inf"), "b": 0.2}

    with pytest.raises(ValueError, match="finite"):
        project_to_honest_noise(noisy, config)


def test_invalid_config_raises_value_error():
    noisy = _base_noisy_result()
    bad_config = {
        "max_node_weight": 1.0,
        "max_edge_weight": 1.0,
        "min_weight": 0.0,
        "contraction_factor": 0.0,
    }

    with pytest.raises(ValueError, match="0 < contraction_factor <= 1"):
        project_to_honest_noise(noisy, bad_config)
