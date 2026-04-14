import json

from qec.orchestration.governance_topology_evolution_kernel import (
    GovernanceTopologyEvolutionKernel,
    TopologyEvolutionMetric,
    TopologyEvolutionReceipt,
    TopologyEvolutionScenario,
    build_topology_evolution_receipt,
    build_topology_evolution_scenario,
    compare_topology_evolution_replay,
    run_governance_topology_evolution,
    summarize_topology_evolution,
    validate_topology_evolution,
)


def _stable_series():
    topology_series = (
        {"topology_id": "t0", "basin_id": "A", "continuity_ok": True, "severity": 0.1, "replay_identity": "r0"},
        {"topology_id": "t1", "basin_id": "A", "continuity_ok": True, "severity": 0.1, "replay_identity": "r0"},
        {"topology_id": "t2", "basin_id": "A", "continuity_ok": True, "severity": 0.1, "replay_identity": "r0"},
    )
    drift_series = (
        {"drift_id": "d0", "from_basin": "A", "to_basin": "A", "transition_count": 0, "drift_magnitude": 0.0},
    )
    return topology_series, drift_series


def _evolving_series():
    topology_series = (
        {"topology_id": "t0", "basin_id": "A", "continuity_ok": True, "severity": 0.1, "replay_identity": "r0"},
        {"topology_id": "t1", "basin_id": "B", "continuity_ok": False, "severity": 0.7, "replay_identity": "r1"},
        {"topology_id": "t2", "basin_id": "C", "continuity_ok": False, "severity": 0.9, "replay_identity": "r2"},
    )
    drift_series = (
        {"drift_id": "d0", "from_basin": "A", "to_basin": "B", "transition_count": 2, "drift_magnitude": 0.4},
        {"drift_id": "d1", "from_basin": "B", "to_basin": "C", "transition_count": 3, "drift_magnitude": 0.7},
    )
    return topology_series, drift_series


def _metric_map(kernel):
    return {m.metric_name: m.metric_value for m in kernel.metrics}


def test_deterministic_repeated_analysis() -> None:
    topology_series, drift_series = _evolving_series()
    scenario = build_topology_evolution_scenario(
        scenario_id="repeatable",
        topology_series=topology_series,
        drift_series=drift_series,
    )
    a = run_governance_topology_evolution(scenario=scenario)
    b = run_governance_topology_evolution(scenario=scenario)
    assert a.to_canonical_json() == b.to_canonical_json()


def test_stable_hash_reproducibility() -> None:
    topology_series, drift_series = _evolving_series()
    scenario_a = build_topology_evolution_scenario(
        scenario_id="hash-repro",
        topology_series=topology_series,
        drift_series=drift_series,
    )
    scenario_b = build_topology_evolution_scenario(
        scenario_id="hash-repro",
        topology_series=topology_series,
        drift_series=drift_series,
    )
    assert scenario_a.stable_hash() == scenario_b.stable_hash()


def test_stable_identical_topology_series_metrics() -> None:
    topology_series, drift_series = _stable_series()
    scenario = build_topology_evolution_scenario(
        scenario_id="stable-series",
        topology_series=topology_series,
        drift_series=drift_series,
    )
    kernel = run_governance_topology_evolution(scenario=scenario)
    metrics = _metric_map(kernel)
    assert metrics["topology_change_rate"] == 0.0
    assert metrics["continuity_decay_score"] == 0.0
    assert metrics["replay_evolution_stability"] == 1.0


def test_evolving_topology_series_non_trivial_metrics() -> None:
    topology_series, drift_series = _evolving_series()
    kernel = run_governance_topology_evolution(
        scenario=build_topology_evolution_scenario(
            scenario_id="evolving",
            topology_series=topology_series,
            drift_series=drift_series,
        )
    )
    metrics = _metric_map(kernel)
    assert metrics["topology_change_rate"] > 0.0
    assert metrics["continuity_decay_score"] > 0.0
    assert metrics["severity_evolution_score"] > 0.0


def test_malformed_input_handling_is_deterministic() -> None:
    scenario = build_topology_evolution_scenario(
        scenario_id="",
        topology_series={"bad": "type"},
        drift_series=None,
    )
    kernel = run_governance_topology_evolution(scenario=scenario)
    assert "empty_scenario_id" in kernel.violations
    assert "empty_topology_series" in kernel.violations


def test_validator_never_raises_for_malformed_kernel() -> None:
    bad_kernel = object()
    violations = validate_topology_evolution(bad_kernel)
    assert isinstance(violations, tuple)
    assert "missing_scenario" in violations


def test_replay_comparison_stability() -> None:
    topology_series, drift_series = _stable_series()
    scenario = build_topology_evolution_scenario(
        scenario_id="replay",
        topology_series=topology_series,
        drift_series=drift_series,
    )
    a = run_governance_topology_evolution(scenario=scenario)
    b = run_governance_topology_evolution(scenario=scenario)
    report = compare_topology_evolution_replay(a, b)
    assert report["is_stable_replay"] is True
    assert report["mismatches"] == ()


def test_canonical_json_round_trip() -> None:
    topology_series, drift_series = _evolving_series()
    kernel = run_governance_topology_evolution(
        scenario=build_topology_evolution_scenario(
            scenario_id="roundtrip",
            topology_series=topology_series,
            drift_series=drift_series,
        )
    )
    encoded = kernel.to_canonical_json()
    parsed = json.loads(encoded)
    assert json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) == encoded


def test_deterministic_metric_ordering() -> None:
    topology_series, drift_series = _evolving_series()
    kernel = run_governance_topology_evolution(
        scenario=build_topology_evolution_scenario(
            scenario_id="ordering",
            topology_series=topology_series,
            drift_series=drift_series,
        )
    )
    names = tuple(metric.metric_name for metric in kernel.metrics)
    assert names == (
        "topology_change_rate",
        "basin_persistence_score",
        "evolution_transition_density",
        "topology_drift_velocity",
        "continuity_decay_score",
        "severity_evolution_score",
        "replay_evolution_stability",
    )


def test_evolution_score_bounds() -> None:
    topology_series, drift_series = _evolving_series()
    kernel = run_governance_topology_evolution(
        scenario=build_topology_evolution_scenario(
            scenario_id="bounds",
            topology_series=topology_series,
            drift_series=drift_series,
        )
    )
    metrics = _metric_map(kernel)
    assert 0.0 <= metrics["severity_evolution_score"] <= 1.0
    assert 0.0 <= metrics["topology_change_rate"] <= 1.0
    assert 0.0 <= metrics["basin_persistence_score"] <= 1.0
    assert 0.0 <= metrics["continuity_decay_score"] <= 1.0
    assert 0.0 <= metrics["replay_evolution_stability"] <= 1.0


def test_receipt_builder_determinism() -> None:
    topology_series, drift_series = _evolving_series()
    scenario = build_topology_evolution_scenario(
        scenario_id="receipt",
        topology_series=topology_series,
        drift_series=drift_series,
    )
    kernel = run_governance_topology_evolution(scenario=scenario)
    r1 = build_topology_evolution_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        evolution_hash=kernel.evolution_hash,
    )
    r2 = build_topology_evolution_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        evolution_hash=kernel.evolution_hash,
    )
    assert r1.to_canonical_json() == r2.to_canonical_json()


def test_summary_contains_hashes() -> None:
    topology_series, drift_series = _evolving_series()
    kernel = run_governance_topology_evolution(
        scenario=build_topology_evolution_scenario(
            scenario_id="summary",
            topology_series=topology_series,
            drift_series=drift_series,
        )
    )
    text = summarize_topology_evolution(kernel)
    assert kernel.evolution_hash in text
    assert kernel.receipt.receipt_hash in text


def test_dataclass_support_methods() -> None:
    scenario = TopologyEvolutionScenario(
        scenario_id="x",
        topology_series=(),
        drift_series=(),
    )
    metric = TopologyEvolutionMetric(
        metric_name="topology_change_rate",
        metric_order=0,
        metric_value=0.0,
    )
    receipt = TopologyEvolutionReceipt(
        scenario_hash="a" * 64,
        metrics_hash="b" * 64,
        evolution_hash="c" * 64,
        severity_evolution_score=0.0,
        receipt_hash="d" * 64,
    )
    kernel = GovernanceTopologyEvolutionKernel(
        scenario=scenario,
        metrics=(metric,),
        violations=(),
        receipt=receipt,
        evolution_hash="e" * 64,
    )
    assert isinstance(scenario.to_dict(), dict)
    assert isinstance(metric.to_dict(), dict)
    assert isinstance(receipt.to_dict(), dict)
    assert isinstance(kernel.to_dict(), dict)
    assert scenario.stable_hash() == scenario.stable_hash()
    assert metric.stable_hash() == metric.stable_hash()
    assert receipt.stable_hash() == receipt.receipt_hash
    assert kernel.stable_hash() == kernel.evolution_hash
