import json

from qec.orchestration.governance_forecast_drift_reconciliation_kernel import (
    ForecastReconciliationMetric,
    ForecastReconciliationReceipt,
    ForecastReconciliationScenario,
    GovernanceForecastDriftReconciliationKernel,
    build_forecast_reconciliation_receipt,
    build_forecast_reconciliation_scenario,
    compare_forecast_reconciliation_replay,
    run_governance_forecast_reconciliation,
    summarize_forecast_reconciliation,
    validate_forecast_reconciliation,
)


def _accurate_inputs():
    forecast_series = (
        {
            "evolution_id": "e0",
            "basin_id": "A",
            "projected_drift": 0.10,
            "projected_stability": 0.90,
            "projected_severity": 0.10,
            "continuity_expected": True,
            "forecast_confidence": 0.90,
            "replay_identity": "r0",
        },
        {
            "evolution_id": "e1",
            "basin_id": "A",
            "projected_drift": 0.08,
            "projected_stability": 0.92,
            "projected_severity": 0.08,
            "continuity_expected": True,
            "forecast_confidence": 0.92,
            "replay_identity": "r0",
        },
    )
    realized_evolution_series = (
        {
            "evolution_id": "e0",
            "basin_id": "A",
            "continuity_ok": True,
            "severity": 0.10,
            "stability_realized": 0.90,
            "replay_identity": "r0",
        },
        {
            "evolution_id": "e1",
            "basin_id": "A",
            "continuity_ok": True,
            "severity": 0.09,
            "stability_realized": 0.91,
            "replay_identity": "r0",
        },
    )
    realized_drift_series = (
        {
            "drift_id": "d0",
            "from_basin": "A",
            "to_basin": "A",
            "transition_count": 0,
            "drift_magnitude": 0.09,
        },
        {
            "drift_id": "d1",
            "from_basin": "A",
            "to_basin": "A",
            "transition_count": 0,
            "drift_magnitude": 0.11,
        },
    )
    return forecast_series, realized_evolution_series, realized_drift_series


def _poor_inputs():
    forecast_series = (
        {
            "evolution_id": "e0",
            "basin_id": "A",
            "projected_drift": 0.00,
            "projected_stability": 1.00,
            "projected_severity": 0.00,
            "continuity_expected": True,
            "forecast_confidence": 1.00,
            "replay_identity": "r-good",
        },
        {
            "evolution_id": "e1",
            "basin_id": "A",
            "projected_drift": 0.00,
            "projected_stability": 1.00,
            "projected_severity": 0.00,
            "continuity_expected": True,
            "forecast_confidence": 1.00,
            "replay_identity": "r-good",
        },
    )
    realized_evolution_series = (
        {
            "evolution_id": "e0",
            "basin_id": "B",
            "continuity_ok": False,
            "severity": 1.00,
            "stability_realized": 0.00,
            "replay_identity": "r-bad",
        },
        {
            "evolution_id": "e1",
            "basin_id": "C",
            "continuity_ok": False,
            "severity": 0.90,
            "stability_realized": 0.00,
            "replay_identity": "r-bad",
        },
    )
    realized_drift_series = (
        {
            "drift_id": "d0",
            "from_basin": "A",
            "to_basin": "B",
            "transition_count": 3,
            "drift_magnitude": 1.0,
        },
        {
            "drift_id": "d1",
            "from_basin": "B",
            "to_basin": "C",
            "transition_count": 4,
            "drift_magnitude": 0.9,
        },
    )
    return forecast_series, realized_evolution_series, realized_drift_series


def _metric_map(kernel):
    return {m.metric_name: m.metric_value for m in kernel.metrics}


def test_deterministic_repeated_analysis() -> None:
    f, e, d = _accurate_inputs()
    scenario = build_forecast_reconciliation_scenario(
        scenario_id="repeatable",
        forecast_series=f,
        realized_evolution_series=e,
        realized_drift_series=d,
    )
    a = run_governance_forecast_reconciliation(scenario=scenario)
    b = run_governance_forecast_reconciliation(scenario=scenario)
    assert a.to_canonical_json() == b.to_canonical_json()


def test_stable_hash_reproducibility() -> None:
    f, e, d = _accurate_inputs()
    scenario_a = build_forecast_reconciliation_scenario(
        scenario_id="hash-repro",
        forecast_series=f,
        realized_evolution_series=e,
        realized_drift_series=d,
    )
    scenario_b = build_forecast_reconciliation_scenario(
        scenario_id="hash-repro",
        forecast_series=f,
        realized_evolution_series=e,
        realized_drift_series=d,
    )
    assert scenario_a.stable_hash() == scenario_b.stable_hash()


def test_accurate_forecast_low_residual() -> None:
    f, e, d = _accurate_inputs()
    kernel = run_governance_forecast_reconciliation(
        scenario=build_forecast_reconciliation_scenario(
            scenario_id="accurate",
            forecast_series=f,
            realized_evolution_series=e,
            realized_drift_series=d,
        )
    )
    metrics = _metric_map(kernel)
    assert metrics["forecast_error_rate"] < 0.1
    assert metrics["severity_forecast_residual"] < 0.1
    assert metrics["forecast_confidence_calibration"] > 0.8


def test_poor_forecast_elevated_residual() -> None:
    f, e, d = _poor_inputs()
    kernel = run_governance_forecast_reconciliation(
        scenario=build_forecast_reconciliation_scenario(
            scenario_id="poor",
            forecast_series=f,
            realized_evolution_series=e,
            realized_drift_series=d,
        )
    )
    metrics = _metric_map(kernel)
    assert metrics["forecast_error_rate"] > 0.8
    assert metrics["severity_forecast_residual"] > 0.8
    assert metrics["continuity_forecast_error"] > 0.8


def test_topology_id_lineage_preservation_distinct_hashes() -> None:
    forecast_series = ({"evolution_id": "topo-A", "projected_stability": 0.5},)
    realized_base = (
        {
            "topology_id": "topo-A",
            "basin_id": "A",
            "continuity_ok": True,
            "severity": 0.3,
            "stability_realized": 0.7,
        },
    )
    drift_series = ({"from_basin": "A", "to_basin": "A", "drift_magnitude": 0.2},)

    scenario_a = build_forecast_reconciliation_scenario(
        scenario_id="lineage",
        forecast_series=forecast_series,
        realized_evolution_series=realized_base,
        realized_drift_series=drift_series,
    )
    kernel_a = run_governance_forecast_reconciliation(scenario=scenario_a)

    realized_b = (
        {
            "topology_id": "topo-B",
            "basin_id": "A",
            "continuity_ok": True,
            "severity": 0.3,
            "stability_realized": 0.7,
        },
    )
    scenario_b = build_forecast_reconciliation_scenario(
        scenario_id="lineage",
        forecast_series=forecast_series,
        realized_evolution_series=realized_b,
        realized_drift_series=drift_series,
    )
    kernel_b = run_governance_forecast_reconciliation(scenario=scenario_b)

    assert scenario_a.realized_evolution_series[0]["evolution_id"] == "topo-A"
    assert scenario_b.realized_evolution_series[0]["evolution_id"] == "topo-B"
    assert scenario_a.stable_hash() != scenario_b.stable_hash()
    assert kernel_a.reconciliation_hash != kernel_b.reconciliation_hash
    assert kernel_a.receipt.receipt_hash != kernel_b.receipt.receipt_hash


def test_malformed_input_handling() -> None:
    scenario = build_forecast_reconciliation_scenario(
        scenario_id="",
        forecast_series={"bad": "type"},
        realized_evolution_series=None,
        realized_drift_series=None,
    )
    kernel = run_governance_forecast_reconciliation(scenario=scenario)
    assert "empty_scenario_id" in kernel.violations
    assert "empty_forecast_series" in kernel.violations
    assert "empty_realized_evolution_series" in kernel.violations


def test_validator_never_raises() -> None:
    violations = validate_forecast_reconciliation(object())
    assert isinstance(violations, tuple)
    assert "missing_scenario" in violations


def test_replay_comparison_stability() -> None:
    f, e, d = _accurate_inputs()
    scenario = build_forecast_reconciliation_scenario(
        scenario_id="replay",
        forecast_series=f,
        realized_evolution_series=e,
        realized_drift_series=d,
    )
    baseline = run_governance_forecast_reconciliation(scenario=scenario)
    replay = run_governance_forecast_reconciliation(scenario=scenario)
    report = compare_forecast_reconciliation_replay(baseline, replay)
    assert report["is_stable_replay"] is True
    assert report["mismatches"] == ()


def test_canonical_json_round_trip() -> None:
    f, e, d = _poor_inputs()
    kernel = run_governance_forecast_reconciliation(
        scenario=build_forecast_reconciliation_scenario(
            scenario_id="roundtrip",
            forecast_series=f,
            realized_evolution_series=e,
            realized_drift_series=d,
        )
    )
    encoded = kernel.to_canonical_json()
    parsed = json.loads(encoded)
    assert (
        json.dumps(
            parsed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        == encoded
    )


def test_deterministic_metric_ordering() -> None:
    f, e, d = _poor_inputs()
    kernel = run_governance_forecast_reconciliation(
        scenario=build_forecast_reconciliation_scenario(
            scenario_id="ordering",
            forecast_series=f,
            realized_evolution_series=e,
            realized_drift_series=d,
        )
    )
    names = tuple(m.metric_name for m in kernel.metrics)
    assert names == (
        "forecast_error_rate",
        "realized_vs_projected_drift_delta",
        "stability_prediction_accuracy",
        "continuity_forecast_error",
        "severity_forecast_residual",
        "replay_reconciliation_score",
        "forecast_confidence_calibration",
    )


def test_reconciliation_score_bounds() -> None:
    f, e, d = _poor_inputs()
    kernel = run_governance_forecast_reconciliation(
        scenario=build_forecast_reconciliation_scenario(
            scenario_id="bounds",
            forecast_series=f,
            realized_evolution_series=e,
            realized_drift_series=d,
        )
    )
    metrics = _metric_map(kernel)
    for key in (
        "forecast_error_rate",
        "realized_vs_projected_drift_delta",
        "stability_prediction_accuracy",
        "continuity_forecast_error",
        "severity_forecast_residual",
        "replay_reconciliation_score",
        "forecast_confidence_calibration",
    ):
        assert 0.0 <= metrics[key] <= 1.0


def test_receipt_builder_determinism() -> None:
    f, e, d = _accurate_inputs()
    scenario = build_forecast_reconciliation_scenario(
        scenario_id="receipt",
        forecast_series=f,
        realized_evolution_series=e,
        realized_drift_series=d,
    )
    kernel = run_governance_forecast_reconciliation(scenario=scenario)
    r1 = build_forecast_reconciliation_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        reconciliation_hash=kernel.reconciliation_hash,
    )
    r2 = build_forecast_reconciliation_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        reconciliation_hash=kernel.reconciliation_hash,
    )
    assert r1.to_canonical_json() == r2.to_canonical_json()


def test_summary_contains_hashes() -> None:
    f, e, d = _accurate_inputs()
    kernel = run_governance_forecast_reconciliation(
        scenario=build_forecast_reconciliation_scenario(
            scenario_id="summary",
            forecast_series=f,
            realized_evolution_series=e,
            realized_drift_series=d,
        )
    )
    text = summarize_forecast_reconciliation(kernel)
    assert kernel.reconciliation_hash in text
    assert kernel.receipt.receipt_hash in text


def test_carry_forward_evolution_id_priority_chain() -> None:
    scenario = build_forecast_reconciliation_scenario(
        scenario_id="priority",
        forecast_series=({"evolution_id": "e0"}, {"evolution_id": "e1"}, {"evolution_id": "e2"}),
        realized_evolution_series=(
            {"evolution_id": "keep-me", "topology_id": "topo-ignore"},
            {"topology_id": "topo-fallback"},
            {},
        ),
        realized_drift_series=(),
    )
    ids = tuple(row["evolution_id"] for row in scenario.realized_evolution_series)
    assert ids == ("keep-me", "topo-fallback", "evolution_2")


def test_dataclass_support_methods() -> None:
    scenario = ForecastReconciliationScenario(
        scenario_id="x",
        forecast_series=(),
        realized_evolution_series=(),
        realized_drift_series=(),
    )
    metric = ForecastReconciliationMetric(
        metric_name="forecast_error_rate",
        metric_order=0,
        metric_value=0.0,
    )
    receipt = ForecastReconciliationReceipt(
        scenario_hash="a" * 64,
        metrics_hash="b" * 64,
        reconciliation_hash="c" * 64,
        replay_reconciliation_score=0.0,
        receipt_hash="d" * 64,
    )
    kernel = GovernanceForecastDriftReconciliationKernel(
        scenario=scenario,
        metrics=(metric,),
        violations=(),
        receipt=receipt,
        reconciliation_hash="e" * 64,
    )
    assert isinstance(scenario.to_dict(), dict)
    assert isinstance(metric.to_dict(), dict)
    assert isinstance(receipt.to_dict(), dict)
    assert isinstance(kernel.to_dict(), dict)
    assert scenario.stable_hash() == scenario.stable_hash()
    assert metric.stable_hash() == metric.stable_hash()
    assert receipt.stable_hash() == receipt.receipt_hash
    assert kernel.stable_hash() == kernel.reconciliation_hash


def test_metric_pairing_is_stable_for_reordered_realized_rows() -> None:
    forecast_series = (
        {
            "evolution_id": "e0",
            "projected_stability": 0.9,
            "projected_severity": 0.1,
            "continuity_expected": True,
            "replay_identity": "r0",
        },
        {
            "evolution_id": "e1",
            "projected_stability": 0.2,
            "projected_severity": 0.8,
            "continuity_expected": False,
            "replay_identity": "r1",
        },
    )
    realized_base = (
        {
            "evolution_id": "e0",
            "stability_realized": 0.9,
            "severity": 0.1,
            "continuity_ok": True,
            "replay_identity": "r0",
        },
        {
            "evolution_id": "e1",
            "stability_realized": 0.2,
            "severity": 0.8,
            "continuity_ok": False,
            "replay_identity": "r1",
        },
    )
    realized_swapped = (realized_base[1], realized_base[0])
    drift_series = ()

    kernel_a = run_governance_forecast_reconciliation(
        scenario=build_forecast_reconciliation_scenario(
            scenario_id="row-order-a",
            forecast_series=forecast_series,
            realized_evolution_series=realized_base,
            realized_drift_series=drift_series,
        )
    )
    kernel_b = run_governance_forecast_reconciliation(
        scenario=build_forecast_reconciliation_scenario(
            scenario_id="row-order-b",
            forecast_series=forecast_series,
            realized_evolution_series=realized_swapped,
            realized_drift_series=drift_series,
        )
    )

    assert tuple(metric.metric_value for metric in kernel_a.metrics) == tuple(metric.metric_value for metric in kernel_b.metrics)


def test_validator_checks_full_receipt_integrity() -> None:
    f, e, d = _accurate_inputs()
    scenario = build_forecast_reconciliation_scenario(
        scenario_id="receipt-integrity",
        forecast_series=f,
        realized_evolution_series=e,
        realized_drift_series=d,
    )
    kernel = run_governance_forecast_reconciliation(scenario=scenario)
    forged_receipt = ForecastReconciliationReceipt(
        scenario_hash="0" * 64,
        metrics_hash=kernel.receipt.metrics_hash,
        reconciliation_hash="1" * 64,
        replay_reconciliation_score=kernel.receipt.replay_reconciliation_score,
        receipt_hash=kernel.receipt.receipt_hash,
    )
    forged_kernel = GovernanceForecastDriftReconciliationKernel(
        scenario=kernel.scenario,
        metrics=kernel.metrics,
        violations=kernel.violations,
        receipt=forged_receipt,
        reconciliation_hash=kernel.reconciliation_hash,
    )
    violations = validate_forecast_reconciliation(forged_kernel)
    assert "receipt_scenario_hash_mismatch" in violations
    assert "receipt_reconciliation_hash_mismatch" in violations
    assert "receipt_hash_mismatch" in violations
