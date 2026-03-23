"""Tests for the deterministic metrics & topology probe (v98.6)."""

from __future__ import annotations

import copy

from qec.experiments.metrics_probe import (
    analyze_topology,
    classify_state,
    evaluate_metrics,
    generate_mock_strategies,
    generate_test_inputs,
    run_experiments,
)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_generate_test_inputs_deterministic():
    """Inputs must be identical across calls."""
    a = generate_test_inputs()
    b = generate_test_inputs()
    assert a == b


def test_evaluate_metrics_deterministic():
    """Metrics must be identical across calls for the same input."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    a = evaluate_metrics(values)
    b = evaluate_metrics(values)
    assert a == b


def test_run_experiments_deterministic():
    """Full experiment output must be identical across runs."""
    a = run_experiments()
    b = run_experiments()
    assert a == b


# ---------------------------------------------------------------------------
# Metrics presence
# ---------------------------------------------------------------------------


def test_evaluate_metrics_has_field_keys():
    """Field metrics must contain expected keys."""
    values = [1.0, 0.0, -1.0, 0.5, -0.5, 0.0]
    m = evaluate_metrics(values)
    assert "field" in m
    for key in ("phi_alignment", "symmetry_score", "curvature", "complexity"):
        assert key in m["field"], f"missing field key: {key}"


def test_evaluate_metrics_has_multiscale_keys():
    """Multiscale metrics must contain expected keys."""
    values = [1.0, 0.0, -1.0, 0.5, -0.5, 0.0]
    m = evaluate_metrics(values)
    assert "multiscale" in m
    assert "scale_consistency" in m["multiscale"]
    assert "scale_divergence" in m["multiscale"]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classify_state_stable():
    """High phi + high consistency → stable."""
    metrics = {
        "field": {
            "phi_alignment": 0.9,
            "curvature": {"abs_curvature": 0.1, "curvature_variation": 0.0},
            "nonlinear_response": 0.1,
        },
        "multiscale": {"scale_consistency": 0.9, "scale_divergence": 0.1},
    }
    assert classify_state(metrics) == "stable"


def test_classify_state_transitional():
    """High divergence → transitional."""
    metrics = {
        "field": {
            "phi_alignment": 0.3,
            "curvature": {"abs_curvature": 0.1, "curvature_variation": 0.0},
            "nonlinear_response": 0.1,
        },
        "multiscale": {"scale_consistency": 0.3, "scale_divergence": 0.7},
    }
    assert classify_state(metrics) == "transitional"


def test_classify_state_unstable():
    """High curvature → unstable."""
    metrics = {
        "field": {
            "phi_alignment": 0.3,
            "curvature": {"abs_curvature": 2.0, "curvature_variation": 0.0},
            "nonlinear_response": 0.1,
        },
        "multiscale": {"scale_consistency": 0.3, "scale_divergence": 0.3},
    }
    assert classify_state(metrics) == "unstable"


def test_classify_state_mixed():
    """Low everything → mixed."""
    metrics = {
        "field": {
            "phi_alignment": 0.5,
            "curvature": {"abs_curvature": 0.5, "curvature_variation": 0.0},
            "nonlinear_response": 0.5,
        },
        "multiscale": {"scale_consistency": 0.5, "scale_divergence": 0.3},
    }
    assert classify_state(metrics) == "mixed"


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def test_topology_has_clusters():
    """Topology result must contain clusters."""
    strategies = generate_mock_strategies()
    result = analyze_topology(strategies)
    assert "clusters" in result
    assert isinstance(result["clusters"], list)


def test_topology_has_dominant():
    """Topology result must identify a dominant strategy."""
    strategies = generate_mock_strategies()
    result = analyze_topology(strategies)
    assert "dominant" in result
    assert result["dominant"] in strategies


def test_topology_average_distance():
    """Average distance must be a non-negative float."""
    strategies = generate_mock_strategies()
    result = analyze_topology(strategies)
    assert result["average_distance"] >= 0.0


# ---------------------------------------------------------------------------
# No mutation
# ---------------------------------------------------------------------------


def test_evaluate_metrics_no_mutation():
    """evaluate_metrics must not mutate its input list."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    original = copy.deepcopy(values)
    evaluate_metrics(values)
    assert values == original


def test_run_experiments_returns_all_inputs():
    """run_experiments must include results for every test input."""
    results = run_experiments()
    inputs = generate_test_inputs()
    assert len(results["inputs"]) == len(inputs)
    for entry, case in zip(results["inputs"], inputs):
        assert entry["name"] == case["name"]


# ---------------------------------------------------------------------------
# Strategies deterministic
# ---------------------------------------------------------------------------


def test_mock_strategies_deterministic():
    """Mock strategies must be identical across calls."""
    a = generate_mock_strategies()
    b = generate_mock_strategies()
    assert set(a.keys()) == set(b.keys())
    for k in a:
        assert a[k].action_type == b[k].action_type
        assert a[k].params == b[k].params
