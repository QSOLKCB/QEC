import json

from qec.orchestration.governance_stability_forecast_kernel import (
    GovernanceStabilityForecastKernel,
    StabilityForecastMetric,
    StabilityForecastReceipt,
    StabilityForecastScenario,
    build_stability_forecast_receipt,
    build_stability_forecast_scenario,
    compare_stability_forecast_replay,
    run_governance_stability_forecast,
    summarize_stability_forecast,
    validate_stability_forecast,
)


def _stable_series():
    evolution_series = (
        {"evolution_id": "e0", "basin_id": "A", "continuity_ok": True, "severity": 0.05, "replay_identity": "r0"},
        {"evolution_id": "e1", "basin_id": "A", "continuity_ok": True, "severity": 0.05, "replay_identity": "r0"},
        {"evolution_id": "e2", "basin_id": "A", "continuity_ok": True, "severity": 0.05, "replay_identity": "r0"},
    )
    drift_series = (
        {"drift_id": "d0", "from_basin": "A", "to_basin": "A", "transition_count": 0, "drift_magnitude": 0.0},
        {"drift_id": "d1", "from_basin": "A", "to_basin": "A", "transition_count": 0, "drift_magnitude": 0.0},
    )
    return evolution_series, drift_series


def _unstable_series():
    evolution_series = (
        {"evolution_id": "e0", "basin_id": "A", "continuity_ok": True, "severity": 0.2, "replay_identity": "r0"},
        {"evolution_id": "e1", "basin_id": "B", "continuity_ok": False, "severity": 0.8, "replay_identity": "r1"},
        {"evolution_id": "e2", "basin_id": "C", "continuity_ok": False, "severity": 1.0, "replay_identity": "r2"},
    )
    drift_series = (
        {"drift_id": "d0", "from_basin": "A", "to_basin": "B", "transition_count": 1, "drift_magnitude": 0.3},
        {"drift_id": "d1", "from_basin": "B", "to_basin": "C", "transition_count": 3, "drift_magnitude": 0.7},
    )
    return evolution_series, drift_series


def _metric_map(kernel):
    return {m.metric_name: m.metric_value for m in kernel.metrics}


def test_deterministic_repeated_analysis() -> None:
    evolution_series, drift_series = _unstable_series()
    scenario = build_stability_forecast_scenario(
        scenario_id="repeatable",
        evolution_series=evolution_series,
        drift_series=drift_series,
    )
    a = run_governance_stability_forecast(scenario=scenario)
    b = run_governance_stability_forecast(scenario=scenario)
    assert a.to_canonical_json() == b.to_canonical_json()


def test_stable_hash_reproducibility() -> None:
    evolution_series, drift_series = _unstable_series()
    scenario_a = build_stability_forecast_scenario(
        scenario_id="hash-repro",
        evolution_series=evolution_series,
        drift_series=drift_series,
    )
    scenario_b = build_stability_forecast_scenario(
        scenario_id="hash-repro",
        evolution_series=evolution_series,
        drift_series=drift_series,
    )
    assert scenario_a.stable_hash() == scenario_b.stable_hash()


def test_stable_evolution_low_risk_forecast() -> None:
    evolution_series, drift_series = _stable_series()
    kernel = run_governance_stability_forecast(
        scenario=build_stability_forecast_scenario(
            scenario_id="stable",
            evolution_series=evolution_series,
            drift_series=drift_series,
        )
    )
    metrics = _metric_map(kernel)
    assert metrics["projected_instability_risk"] < 0.2
    assert metrics["continuity_failure_risk"] == 0.0
    assert metrics["forecast_confidence_score"] > 0.7


def test_unstable_evolution_elevated_forecast() -> None:
    evolution_series, drift_series = _unstable_series()
    kernel = run_governance_stability_forecast(
        scenario=build_stability_forecast_scenario(
            scenario_id="unstable",
            evolution_series=evolution_series,
            drift_series=drift_series,
        )
    )
    metrics = _metric_map(kernel)
    assert metrics["projected_instability_risk"] > 0.4
    assert metrics["continuity_failure_risk"] > 0.4
    assert metrics["severity_projection_score"] > 0.4


def test_malformed_input_handling() -> None:
    scenario = build_stability_forecast_scenario(
        scenario_id="",
        evolution_series={"bad": "type"},
        drift_series=None,
    )
    kernel = run_governance_stability_forecast(scenario=scenario)
    assert "empty_scenario_id" in kernel.violations
    assert "empty_evolution_series" in kernel.violations


def test_validator_never_raises() -> None:
    violations = validate_stability_forecast(object())
    assert isinstance(violations, tuple)
    assert "missing_scenario" in violations


def test_replay_comparison_stability() -> None:
    evolution_series, drift_series = _stable_series()
    scenario = build_stability_forecast_scenario(
        scenario_id="replay",
        evolution_series=evolution_series,
        drift_series=drift_series,
    )
    baseline = run_governance_stability_forecast(scenario=scenario)
    replay = run_governance_stability_forecast(scenario=scenario)
    report = compare_stability_forecast_replay(baseline, replay)
    assert report["is_stable_replay"] is True
    assert report["mismatches"] == ()


def test_canonical_json_round_trip() -> None:
    evolution_series, drift_series = _unstable_series()
    kernel = run_governance_stability_forecast(
        scenario=build_stability_forecast_scenario(
            scenario_id="roundtrip",
            evolution_series=evolution_series,
            drift_series=drift_series,
        )
    )
    encoded = kernel.to_canonical_json()
    parsed = json.loads(encoded)
    assert json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) == encoded


def test_deterministic_metric_ordering() -> None:
    evolution_series, drift_series = _unstable_series()
    kernel = run_governance_stability_forecast(
        scenario=build_stability_forecast_scenario(
            scenario_id="ordering",
            evolution_series=evolution_series,
            drift_series=drift_series,
        )
    )
    names = tuple(m.metric_name for m in kernel.metrics)
    assert names == (
        "projected_instability_risk",
        "basin_decay_forecast",
        "transition_acceleration_score",
        "replay_stability_forecast",
        "severity_projection_score",
        "continuity_failure_risk",
        "forecast_confidence_score",
    )


def test_forecast_score_bounds() -> None:
    evolution_series, drift_series = _unstable_series()
    kernel = run_governance_stability_forecast(
        scenario=build_stability_forecast_scenario(
            scenario_id="bounds",
            evolution_series=evolution_series,
            drift_series=drift_series,
        )
    )
    metrics = _metric_map(kernel)
    for key in (
        "projected_instability_risk",
        "basin_decay_forecast",
        "transition_acceleration_score",
        "replay_stability_forecast",
        "severity_projection_score",
        "continuity_failure_risk",
        "forecast_confidence_score",
    ):
        assert 0.0 <= metrics[key] <= 1.0


def test_receipt_builder_determinism() -> None:
    evolution_series, drift_series = _unstable_series()
    scenario = build_stability_forecast_scenario(
        scenario_id="receipt",
        evolution_series=evolution_series,
        drift_series=drift_series,
    )
    kernel = run_governance_stability_forecast(scenario=scenario)
    r1 = build_stability_forecast_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        forecast_hash=kernel.forecast_hash,
    )
    r2 = build_stability_forecast_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        forecast_hash=kernel.forecast_hash,
    )
    assert r1.to_canonical_json() == r2.to_canonical_json()


def test_summary_contains_hashes() -> None:
    evolution_series, drift_series = _unstable_series()
    kernel = run_governance_stability_forecast(
        scenario=build_stability_forecast_scenario(
            scenario_id="summary",
            evolution_series=evolution_series,
            drift_series=drift_series,
        )
    )
    text = summarize_stability_forecast(kernel)
    assert kernel.forecast_hash in text
    assert kernel.receipt.receipt_hash in text


def test_dataclass_support_methods() -> None:
    scenario = StabilityForecastScenario(
        scenario_id="x",
        evolution_series=(),
        drift_series=(),
    )
    metric = StabilityForecastMetric(
        metric_name="projected_instability_risk",
        metric_order=0,
        metric_value=0.0,
    )
    receipt = StabilityForecastReceipt(
        scenario_hash="a" * 64,
        metrics_hash="b" * 64,
        forecast_hash="c" * 64,
        severity_projection_score=0.0,
        receipt_hash="d" * 64,
    )
    kernel = GovernanceStabilityForecastKernel(
        scenario=scenario,
        metrics=(metric,),
        violations=(),
        receipt=receipt,
        forecast_hash="e" * 64,
    )
    assert isinstance(scenario.to_dict(), dict)
    assert isinstance(metric.to_dict(), dict)
    assert isinstance(receipt.to_dict(), dict)
    assert isinstance(kernel.to_dict(), dict)
    assert scenario.stable_hash() == scenario.stable_hash()
    assert metric.stable_hash() == metric.stable_hash()
    assert receipt.stable_hash() == receipt.receipt_hash
    assert kernel.stable_hash() == kernel.forecast_hash
