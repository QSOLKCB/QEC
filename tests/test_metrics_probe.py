"""Tests for the deterministic metrics & topology probe (v98.7)."""

from __future__ import annotations

import copy
import importlib
import sys

from qec.experiments.metrics_probe import (
    analyze_topology,
    classify_state,
    evaluate_metrics,
    generate_metric_sequences,
    generate_mock_strategies,
    generate_test_inputs,
    print_experiment_report,
    run_experiments,
    run_trajectory_experiments,
    summarize_experiment_patterns,
)


# ---------------------------------------------------------------------------
# Import safety
# ---------------------------------------------------------------------------


def test_import_metrics_probe_does_not_raise():
    """Importing metrics_probe must succeed without optional dependencies."""
    mod = importlib.import_module("qec.experiments.metrics_probe")
    assert hasattr(mod, "run_experiments")


def test_import_does_not_pull_cffi():
    """Importing metrics_probe must not directly import cffi from QEC code."""
    before = set(sys.modules.keys())
    importlib.import_module("qec.experiments.metrics_probe")
    after = set(sys.modules.keys())
    new_modules = after - before
    assert not any(
        m.startswith("_cffi_backend") or m.startswith("cffi")
        for m in new_modules
        if "qec" in repr(sys.modules.get(m, ""))
    )


def test_run_experiments_executes_successfully():
    """run_experiments must complete without error and return expected shape."""
    result = run_experiments()
    assert "inputs" in result
    assert "topology" in result
    assert len(result["inputs"]) > 0


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


# ---------------------------------------------------------------------------
# Expanded inputs (v98.7)
# ---------------------------------------------------------------------------


def test_expanded_inputs_count():
    """generate_test_inputs must return at least 16 test cases."""
    inputs = generate_test_inputs()
    assert len(inputs) >= 16


def test_expanded_inputs_names_unique():
    """All test input names must be unique."""
    inputs = generate_test_inputs()
    names = [inp["name"] for inp in inputs]
    assert len(names) == len(set(names))


def test_expanded_inputs_all_have_values():
    """Every test input must have a 'values' list of floats."""
    inputs = generate_test_inputs()
    for inp in inputs:
        assert "name" in inp
        assert "values" in inp
        assert isinstance(inp["values"], list)
        assert len(inp["values"]) > 0
        for v in inp["values"]:
            assert isinstance(v, float)


def test_expanded_inputs_deterministic():
    """Expanded inputs must be deterministic across calls."""
    a = generate_test_inputs()
    b = generate_test_inputs()
    assert a == b


# ---------------------------------------------------------------------------
# Calibration summary (v98.7)
# ---------------------------------------------------------------------------


def test_summarize_experiment_patterns_structure():
    """Summary must return a dict keyed by regime with correct fields."""
    results = run_experiments()
    summary = summarize_experiment_patterns(results)
    assert isinstance(summary, dict)
    assert len(summary) > 0
    for regime, info in summary.items():
        assert isinstance(regime, str)
        assert "count" in info
        assert "avg_basin_score" in info
        assert "avg_summary_score" in info
        assert "avg_phi" in info
        assert "avg_divergence" in info
        assert "avg_curvature" in info
        assert "avg_resonance" in info
        assert info["count"] > 0


def test_summarize_experiment_patterns_counts():
    """Total count across regimes must equal number of inputs."""
    results = run_experiments()
    summary = summarize_experiment_patterns(results)
    total = sum(info["count"] for info in summary.values())
    assert total == len(results["inputs"])


def test_summarize_experiment_patterns_deterministic():
    """Summary must be deterministic."""
    r1 = run_experiments()
    r2 = run_experiments()
    s1 = summarize_experiment_patterns(r1)
    s2 = summarize_experiment_patterns(r2)
    assert s1 == s2


# ---------------------------------------------------------------------------
# Trajectory experiments (v98.7)
# ---------------------------------------------------------------------------


def test_generate_metric_sequences_count():
    """Must return at least 3 metric sequences."""
    seqs = generate_metric_sequences()
    assert len(seqs) >= 3


def test_generate_metric_sequences_structure():
    """Each sequence must have name and sequence keys."""
    seqs = generate_metric_sequences()
    for seq in seqs:
        assert "name" in seq
        assert "sequence" in seq
        assert isinstance(seq["sequence"], list)
        assert len(seq["sequence"]) > 0


def test_generate_metric_sequences_deterministic():
    """Metric sequences must be deterministic."""
    a = generate_metric_sequences()
    b = generate_metric_sequences()
    assert a == b


def test_run_trajectory_experiments_returns_results():
    """Trajectory experiments must return results for each sequence."""
    results = run_trajectory_experiments()
    seqs = generate_metric_sequences()
    assert len(results) == len(seqs)
    for r in results:
        assert "name" in r
        assert "trajectory" in r
        traj = r["trajectory"]
        assert "regimes" in traj
        assert "transitions" in traj
        assert "stable_segments" in traj
        assert "oscillation_flags" in traj


def test_run_trajectory_experiments_deterministic():
    """Trajectory experiments must be deterministic."""
    a = run_trajectory_experiments()
    b = run_trajectory_experiments()
    assert a == b


def test_trajectory_stable_convergence():
    """Stable convergence sequence must produce all-stable regimes."""
    results = run_trajectory_experiments()
    stable_result = [r for r in results if r["name"] == "stable_convergence"][0]
    regimes = stable_result["trajectory"]["regimes"]
    assert all(r == "stable" for r in regimes)


def test_trajectory_regime_transition():
    """Regime transition sequence must show at least one transition."""
    results = run_trajectory_experiments()
    trans_result = [r for r in results if r["name"] == "regime_transition"][0]
    transitions = trans_result["trajectory"]["transitions"]
    assert any(t["transition"] for t in transitions)


# ---------------------------------------------------------------------------
# Report output (v98.7)
# ---------------------------------------------------------------------------


def test_print_experiment_report_runs(capsys):
    """Report printer must complete without error and produce output."""
    results = run_experiments()
    print_experiment_report(results)
    captured = capsys.readouterr()
    assert "Experiment Report" in captured.out
    assert "Calibration Summary by Regime" in captured.out
    # Check compact summary line format
    assert "basin=" in captured.out
    assert "phi=" in captured.out
    assert "consistency=" in captured.out


# ---------------------------------------------------------------------------
# Confidence preservation (v98.8.1)
# ---------------------------------------------------------------------------


def test_strategy_dicts_preserve_confidence():
    """strategy_dicts in run_experiments must include confidence key."""
    strategies = generate_mock_strategies()
    strategy_dicts = {
        sid: {
            "action_type": s.action_type,
            "params": dict(s.params),
            "confidence": getattr(s, "confidence", 0.0),
        }
        for sid, s in strategies.items()
    }
    for sid, d in strategy_dicts.items():
        assert "confidence" in d
        assert isinstance(d["confidence"], float)


def test_mock_strategy_has_confidence_attr():
    """_MockStrategy must expose a confidence attribute."""
    strategies = generate_mock_strategies()
    for s in strategies.values():
        assert hasattr(s, "confidence")
        assert isinstance(s.confidence, float)
