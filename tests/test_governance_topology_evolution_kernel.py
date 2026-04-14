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


# --------------------------------------------------------------------------- #
# Regression: healthy run must not inherit artificial receipt violations       #
# --------------------------------------------------------------------------- #

def test_healthy_run_no_receipt_violations() -> None:
    """Healthy scenario must produce zero violations (no false receipt violations)."""
    topology_series, drift_series = _stable_series()
    scenario = build_topology_evolution_scenario(
        scenario_id="healthy",
        topology_series=topology_series,
        drift_series=drift_series,
    )
    kernel = run_governance_topology_evolution(scenario=scenario)
    assert "receipt_metrics_hash_mismatch" not in kernel.violations
    assert len(kernel.violations) == 0


def test_healthy_evolving_run_no_receipt_violations() -> None:
    """Evolving scenario with valid data must not inherit artificial receipt violations."""
    topology_series, drift_series = _evolving_series()
    scenario = build_topology_evolution_scenario(
        scenario_id="healthy-evolving",
        topology_series=topology_series,
        drift_series=drift_series,
    )
    kernel = run_governance_topology_evolution(scenario=scenario)
    assert "receipt_metrics_hash_mismatch" not in kernel.violations


# --------------------------------------------------------------------------- #
# Regression: temporal metric ordering follows input, not ID sort order        #
# --------------------------------------------------------------------------- #

def test_temporal_metrics_follow_input_order_not_id_order() -> None:
    """topology_change_rate must reflect the caller-provided sequence, not sorted IDs."""
    # IDs are intentionally non-chronological (z, a, m) so sorting by ID would
    # reorder A→A→B into A→B→A, changing the computed change rate.
    topology_series_temporal = (
        {"topology_id": "z1", "basin_id": "A", "continuity_ok": True, "severity": 0.0, "replay_identity": "r0"},
        {"topology_id": "a1", "basin_id": "A", "continuity_ok": True, "severity": 0.0, "replay_identity": "r0"},
        {"topology_id": "m1", "basin_id": "B", "continuity_ok": True, "severity": 0.0, "replay_identity": "r0"},
    )
    drift_series_temporal = ()
    scenario = build_topology_evolution_scenario(
        scenario_id="temporal-order",
        topology_series=topology_series_temporal,
        drift_series=drift_series_temporal,
    )
    kernel = run_governance_topology_evolution(scenario=scenario)
    metrics = _metric_map(kernel)
    # Temporal sequence: A → A → B → 1 change out of 2 transitions = 0.5
    assert metrics["topology_change_rate"] == 0.5


def test_temporal_replay_stability_follows_input_order() -> None:
    """replay_evolution_stability must be computed over the caller-provided sequence."""
    # Temporal order: same-id (stable), different-id (unstable) — IDs sort opposite.
    topology_series_temporal = (
        {"topology_id": "z2", "basin_id": "A", "continuity_ok": True, "severity": 0.0, "replay_identity": "stable"},
        {"topology_id": "a2", "basin_id": "A", "continuity_ok": True, "severity": 0.0, "replay_identity": "stable"},
        {"topology_id": "m2", "basin_id": "A", "continuity_ok": True, "severity": 0.0, "replay_identity": "different"},
    )
    scenario = build_topology_evolution_scenario(
        scenario_id="replay-order",
        topology_series=topology_series_temporal,
        drift_series=(),
    )
    kernel = run_governance_topology_evolution(scenario=scenario)
    metrics = _metric_map(kernel)
    # Temporal pairs: (z2,a2) same identity → stable; (a2,m2) different → unstable.
    # 1 stable pair out of 2 = 0.5.
    assert metrics["replay_evolution_stability"] == 0.5


def test_temporal_drift_velocity_follows_input_order() -> None:
    """Drift metrics must use caller-provided drift_series order, not sorted order."""
    drift_series_temporal = (
        {"drift_id": "z_d", "from_basin": "A", "to_basin": "B", "transition_count": 1, "drift_magnitude": 0.2},
        {"drift_id": "a_d", "from_basin": "B", "to_basin": "C", "transition_count": 2, "drift_magnitude": 0.8},
    )
    topology_series_temporal = (
        {"topology_id": "t0", "basin_id": "A", "continuity_ok": True, "severity": 0.0, "replay_identity": "r0"},
        {"topology_id": "t1", "basin_id": "B", "continuity_ok": True, "severity": 0.0, "replay_identity": "r0"},
        {"topology_id": "t2", "basin_id": "C", "continuity_ok": True, "severity": 0.0, "replay_identity": "r0"},
    )
    scenario = build_topology_evolution_scenario(
        scenario_id="drift-order",
        topology_series=topology_series_temporal,
        drift_series=drift_series_temporal,
    )
    kernel = run_governance_topology_evolution(scenario=scenario)
    metrics = _metric_map(kernel)
    # Average drift magnitude: (0.2 + 0.8) / 2 = 0.5 regardless of sort order,
    # but total_transitions must reflect the actual drift entries.
    assert metrics["topology_drift_velocity"] == 0.5


def test_build_scenario_preserves_temporal_order() -> None:
    """build_topology_evolution_scenario must preserve the original series order."""
    topology_series = (
        {"topology_id": "z3", "basin_id": "X", "continuity_ok": True, "severity": 0.0, "replay_identity": "r"},
        {"topology_id": "a3", "basin_id": "Y", "continuity_ok": True, "severity": 0.0, "replay_identity": "r"},
    )
    scenario = build_topology_evolution_scenario(
        scenario_id="preserve-order",
        topology_series=topology_series,
        drift_series=(),
    )
    stored_ids = tuple(row["topology_id"] for row in scenario.topology_series)
    # Original order must be preserved: z3 before a3 (not sorted to a3 before z3).
    assert stored_ids == ("z3", "a3")
