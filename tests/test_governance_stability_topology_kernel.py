from __future__ import annotations

import copy
import json

import pytest

from qec.orchestration.governance_stability_topology_kernel import (
    GovernanceStabilityTopologyKernel,
    StabilityTopologyMetric,
    StabilityTopologyReceipt,
    StabilityTopologyScenario,
    build_stability_topology_receipt,
    build_stability_topology_scenario,
    compare_stability_topology_replay,
    run_governance_stability_topology,
    summarize_stability_topology,
    validate_stability_topology,
)


def _sample_benchmark_series():
    return [
        {
            "benchmark_id": "b0",
            "decision_basin": "allow",
            "boundary_failures": 0,
            "continuity_ok": True,
            "replay_identity": "r0",
        },
        {
            "benchmark_id": "b1",
            "decision_basin": "allow",
            "boundary_failures": 1,
            "continuity_ok": True,
            "replay_identity": "r0",
        },
        {
            "benchmark_id": "b2",
            "decision_basin": "deny",
            "boundary_failures": 2,
            "continuity_ok": False,
            "replay_identity": "r1",
        },
    ]


def _sample_drift_series():
    return [
        {"drift_id": "d0", "from_basin": "allow", "to_basin": "deny", "transition_count": 2},
        {"drift_id": "d1", "from_basin": "deny", "to_basin": "allow", "transition_count": 1},
    ]


def test_deterministic_repeated_analysis():
    scenario = build_stability_topology_scenario(
        scenario_id="s0",
        benchmark_series=_sample_benchmark_series(),
        drift_series=_sample_drift_series(),
    )
    run_a = run_governance_stability_topology(scenario)
    run_b = run_governance_stability_topology(scenario)
    assert run_a.to_canonical_json() == run_b.to_canonical_json()
    assert run_a.stable_hash() == run_b.stable_hash()


def test_stable_hash_reproducibility():
    scenario_a = build_stability_topology_scenario(
        scenario_id="s0",
        benchmark_series=_sample_benchmark_series(),
        drift_series=_sample_drift_series(),
    )
    scenario_b = build_stability_topology_scenario(
        scenario_id="s0",
        benchmark_series=_sample_benchmark_series(),
        drift_series=_sample_drift_series(),
    )
    assert scenario_a.stable_hash() == scenario_b.stable_hash()


def test_identical_series_stable_topology():
    data = _sample_benchmark_series()
    scenario = build_stability_topology_scenario(
        scenario_id="stable",
        benchmark_series=data,
        drift_series=[{"drift_id": "d0", "from_basin": "allow", "to_basin": "allow", "transition_count": 1}],
    )
    analysis = run_governance_stability_topology(scenario)
    metric_map = {m.metric_name: m.metric_value for m in analysis.metrics}
    assert metric_map["replay_basin_stability"] >= 0.5
    assert metric_map["topology_severity_score"] <= 1.0


def test_drift_rich_series_nontrivial_topology():
    scenario = build_stability_topology_scenario(
        scenario_id="rich",
        benchmark_series=[
            {"benchmark_id": "b0", "decision_basin": "a", "boundary_failures": 5, "continuity_ok": False, "replay_identity": "r0"},
            {"benchmark_id": "b1", "decision_basin": "b", "boundary_failures": 4, "continuity_ok": True, "replay_identity": "r1"},
            {"benchmark_id": "b2", "decision_basin": "c", "boundary_failures": 3, "continuity_ok": False, "replay_identity": "r2"},
        ],
        drift_series=[
            {"drift_id": "d0", "from_basin": "a", "to_basin": "b", "transition_count": 9},
            {"drift_id": "d1", "from_basin": "b", "to_basin": "c", "transition_count": 10},
            {"drift_id": "d2", "from_basin": "c", "to_basin": "a", "transition_count": 8},
        ],
    )
    analysis = run_governance_stability_topology(scenario)
    metric_map = {m.metric_name: m.metric_value for m in analysis.metrics}
    assert metric_map["stability_cluster_count"] >= 3.0
    assert metric_map["topology_severity_score"] > 0.2


def test_malformed_input_handling_deterministic_violations():
    analysis = run_governance_stability_topology({"not": "scenario"})
    assert "malformed_scenario_input" in analysis.violations
    assert "empty_benchmark_series" in analysis.violations
    assert "empty_drift_series" in analysis.violations


def test_validator_never_raises_on_malformed_object():
    violations = validate_stability_topology(object())
    assert violations == ("malformed_governance_stability_topology",)


def test_replay_comparison_stability():
    scenario = build_stability_topology_scenario(
        scenario_id="s1",
        benchmark_series=_sample_benchmark_series(),
        drift_series=_sample_drift_series(),
    )
    run_a = run_governance_stability_topology(scenario)
    run_b = run_governance_stability_topology(scenario)
    comparison = compare_stability_topology_replay(run_a, run_b)
    assert comparison["match"] is True
    assert comparison["mismatch_fields"] == ()


def test_canonical_json_round_trip():
    scenario = build_stability_topology_scenario(
        scenario_id="s2",
        benchmark_series=_sample_benchmark_series(),
        drift_series=_sample_drift_series(),
    )
    analysis = run_governance_stability_topology(scenario)
    parsed = json.loads(analysis.to_canonical_json())
    assert parsed["topology_hash"] == analysis.topology_hash
    assert parsed["receipt"]["receipt_hash"] == analysis.receipt.receipt_hash


def test_deterministic_metric_ordering():
    scenario = build_stability_topology_scenario(
        scenario_id="s3",
        benchmark_series=_sample_benchmark_series(),
        drift_series=_sample_drift_series(),
    )
    analysis = run_governance_stability_topology(scenario)
    names = tuple(metric.metric_name for metric in analysis.metrics)
    assert names == (
        "stability_cluster_count",
        "dominant_decision_basin_share",
        "drift_transition_density",
        "replay_basin_stability",
        "continuity_surface_entropy",
        "boundary_failure_topology_score",
        "topology_severity_score",
    )


def test_severity_score_bounds():
    scenario = build_stability_topology_scenario(
        scenario_id="s4",
        benchmark_series=_sample_benchmark_series(),
        drift_series=_sample_drift_series(),
    )
    analysis = run_governance_stability_topology(scenario)
    severity = analysis.receipt.topology_severity_score
    assert 0.0 <= severity <= 1.0


def test_no_input_mutation():
    bench = _sample_benchmark_series()
    drift = _sample_drift_series()
    bench_before = copy.deepcopy(bench)
    drift_before = copy.deepcopy(drift)
    scenario = build_stability_topology_scenario(
        scenario_id="mut",
        benchmark_series=bench,
        drift_series=drift,
    )
    _ = run_governance_stability_topology(scenario)
    assert bench == bench_before
    assert drift == drift_before


def test_receipt_builder_deterministic():
    receipt_a = build_stability_topology_receipt(
        scenario_hash="a",
        metrics_hash="b",
        topology_hash="c",
        topology_severity_score=0.25,
    )
    receipt_b = build_stability_topology_receipt(
        scenario_hash="a",
        metrics_hash="b",
        topology_hash="c",
        topology_severity_score=0.25,
    )
    assert receipt_a.receipt_hash == receipt_b.receipt_hash


def test_summary_and_dataclass_contracts():
    scenario = build_stability_topology_scenario(
        scenario_id="s5",
        benchmark_series=_sample_benchmark_series(),
        drift_series=_sample_drift_series(),
    )
    analysis = run_governance_stability_topology(scenario)
    summary = summarize_stability_topology(analysis)
    assert isinstance(analysis, GovernanceStabilityTopologyKernel)
    assert isinstance(analysis.scenario, StabilityTopologyScenario)
    assert isinstance(analysis.metrics[0], StabilityTopologyMetric)
    assert isinstance(analysis.receipt, StabilityTopologyReceipt)
    assert summary["scenario_id"] == "s5"
    assert "topology_severity_score" in summary
    assert "dominant_decision_basin_share" in summary["metric_values"]
    assert "dominant_decision_basin" not in summary["metric_values"]


def test_zero_transition_count_preserved():
    """Regression: explicit transition_count=0 must not be coerced to 1."""
    scenario = build_stability_topology_scenario(
        scenario_id="zero_tc",
        benchmark_series=_sample_benchmark_series(),
        drift_series=[
            {"drift_id": "d0", "from_basin": "allow", "to_basin": "deny", "transition_count": 0},
            {"drift_id": "d1", "from_basin": "deny", "to_basin": "allow", "transition_count": 2},
        ],
    )
    zero_node = next(n for n in scenario.drift_series if n.drift_id == "d0")
    assert zero_node.transition_count == 0, "explicit zero transition_count must be preserved"
    analysis = run_governance_stability_topology(scenario)
    metric_map = {m.metric_name: m.metric_value for m in analysis.metrics}
    # With one zero and one 2-transition drift, mean is 1.0 → density = 0.1
    # If 0 were coerced to 1, mean would be 1.5 → density = 0.15; values differ.
    assert metric_map["drift_transition_density"] == pytest.approx(0.1, abs=1e-9)


def test_validate_topology_hash_integrity():
    """Regression: receipt with tampered topology_hash must trigger topology_hash_mismatch."""
    import dataclasses

    scenario = build_stability_topology_scenario(
        scenario_id="tamper",
        benchmark_series=_sample_benchmark_series(),
        drift_series=_sample_drift_series(),
    )
    analysis = run_governance_stability_topology(scenario)
    tampered_receipt = dataclasses.replace(analysis.receipt, topology_hash="deadbeef" * 8)
    tampered_analysis = dataclasses.replace(analysis, receipt=tampered_receipt)
    violations = validate_stability_topology(tampered_analysis)
    assert "topology_hash_mismatch" in violations
