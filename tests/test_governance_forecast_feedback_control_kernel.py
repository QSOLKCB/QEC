import json

from qec.orchestration.governance_forecast_feedback_control_kernel import (
    ForecastFeedbackControlMetric,
    ForecastFeedbackControlReceipt,
    ForecastFeedbackControlScenario,
    GovernanceForecastFeedbackControlKernel,
    build_feedback_control_receipt,
    build_feedback_control_scenario,
    compare_feedback_control_replay,
    run_governance_feedback_control,
    summarize_feedback_control,
    validate_feedback_control,
)


def _low_residual_inputs():
    reconciliation_series = (
        {
            "reconciliation_id": "r0",
            "residual_magnitude": 0.05,
            "stability_residual": 0.04,
            "continuity_residual": 0.03,
            "severity_residual": 0.05,
            "replay_residual": 0.02,
            "calibration_residual": 0.04,
        },
        {
            "reconciliation_id": "r1",
            "residual_magnitude": 0.06,
            "stability_residual": 0.05,
            "continuity_residual": 0.04,
            "severity_residual": 0.06,
            "replay_residual": 0.03,
            "calibration_residual": 0.05,
        },
    )
    drift_series = (
        {
            "drift_id": "d0",
            "from_basin": "A",
            "to_basin": "A",
            "transition_count": 0,
            "drift_magnitude": 0.05,
        },
    )
    return reconciliation_series, drift_series


def _high_residual_inputs():
    reconciliation_series = (
        {
            "reconciliation_id": "r0",
            "residual_magnitude": 0.95,
            "stability_residual": 0.90,
            "continuity_residual": 0.92,
            "severity_residual": 0.95,
            "replay_residual": 0.80,
            "calibration_residual": 0.85,
        },
        {
            "reconciliation_id": "r1",
            "residual_magnitude": 0.90,
            "stability_residual": 0.88,
            "continuity_residual": 0.90,
            "severity_residual": 0.92,
            "replay_residual": 0.75,
            "calibration_residual": 0.88,
        },
    )
    drift_series = (
        {
            "drift_id": "d0",
            "from_basin": "A",
            "to_basin": "B",
            "transition_count": 4,
            "drift_magnitude": 0.95,
        },
        {
            "drift_id": "d1",
            "from_basin": "B",
            "to_basin": "C",
            "transition_count": 6,
            "drift_magnitude": 0.90,
        },
    )
    return reconciliation_series, drift_series


def _metric_map(kernel):
    return {m.metric_name: m.metric_value for m in kernel.metrics}


def test_deterministic_repeated_analysis() -> None:
    r, d = _low_residual_inputs()
    scenario = build_feedback_control_scenario(
        scenario_id="repeatable",
        reconciliation_series=r,
        drift_series=d,
    )
    a = run_governance_feedback_control(scenario=scenario)
    b = run_governance_feedback_control(scenario=scenario)
    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.control_hash == b.control_hash
    assert a.receipt.receipt_hash == b.receipt.receipt_hash


def test_stable_hash_reproducibility() -> None:
    r, d = _low_residual_inputs()
    scenario_a = build_feedback_control_scenario(
        scenario_id="hash-repro",
        reconciliation_series=r,
        drift_series=d,
    )
    scenario_b = build_feedback_control_scenario(
        scenario_id="hash-repro",
        reconciliation_series=r,
        drift_series=d,
    )
    assert scenario_a.stable_hash() == scenario_b.stable_hash()


def test_low_residual_low_control_pressure() -> None:
    r, d = _low_residual_inputs()
    kernel = run_governance_feedback_control(
        scenario=build_feedback_control_scenario(
            scenario_id="low",
            reconciliation_series=r,
            drift_series=d,
        )
    )
    metrics = _metric_map(kernel)
    assert metrics["residual_control_pressure"] < 0.2
    assert metrics["bounded_control_confidence"] > 0.8
    assert kernel.aggregate_recommendation in ("observe_only", "monitor")
    for _rid, rec in kernel.recommendations:
        assert rec in ("observe_only", "monitor")


def test_high_residual_elevated_control_pressure() -> None:
    r, d = _high_residual_inputs()
    kernel = run_governance_feedback_control(
        scenario=build_feedback_control_scenario(
            scenario_id="high",
            reconciliation_series=r,
            drift_series=d,
        )
    )
    metrics = _metric_map(kernel)
    assert metrics["residual_control_pressure"] > 0.8
    assert metrics["severity_feedback_gain"] > 0.8
    assert metrics["bounded_control_confidence"] < 0.2
    assert kernel.aggregate_recommendation == "isolate"


def test_advisory_output_determinism() -> None:
    r, d = _high_residual_inputs()
    scenario = build_feedback_control_scenario(
        scenario_id="advisory-det",
        reconciliation_series=r,
        drift_series=d,
    )
    k1 = run_governance_feedback_control(scenario=scenario)
    k2 = run_governance_feedback_control(scenario=scenario)
    assert k1.recommendations == k2.recommendations
    assert k1.aggregate_recommendation == k2.aggregate_recommendation
    assert k1.advisory_only is True
    assert k2.advisory_only is True


def test_malformed_input_handling() -> None:
    scenario = build_feedback_control_scenario(
        scenario_id="",
        reconciliation_series={"not": "a-series"},
        drift_series=None,
    )
    kernel = run_governance_feedback_control(scenario=scenario)
    assert "empty_scenario_id" in kernel.violations
    assert "empty_reconciliation_series" in kernel.violations
    assert kernel.aggregate_recommendation == "observe_only"


def test_validator_never_raises_on_arbitrary_object() -> None:
    violations = validate_feedback_control(object())
    assert isinstance(violations, tuple)
    assert "missing_scenario" in violations


def test_validator_never_raises_on_none() -> None:
    violations = validate_feedback_control(None)
    assert isinstance(violations, tuple)
    assert "missing_scenario" in violations


def test_replay_comparison_stability() -> None:
    r, d = _high_residual_inputs()
    scenario = build_feedback_control_scenario(
        scenario_id="replay",
        reconciliation_series=r,
        drift_series=d,
    )
    baseline = run_governance_feedback_control(scenario=scenario)
    replay = run_governance_feedback_control(scenario=scenario)
    report = compare_feedback_control_replay(baseline, replay)
    assert report["is_stable_replay"] is True
    assert report["mismatches"] == ()
    assert report["baseline_aggregate_recommendation"] == (
        report["replay_aggregate_recommendation"]
    )


def test_canonical_json_round_trip() -> None:
    r, d = _high_residual_inputs()
    kernel = run_governance_feedback_control(
        scenario=build_feedback_control_scenario(
            scenario_id="roundtrip",
            reconciliation_series=r,
            drift_series=d,
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


def test_metric_bounds_and_ordering() -> None:
    r, d = _high_residual_inputs()
    kernel = run_governance_feedback_control(
        scenario=build_feedback_control_scenario(
            scenario_id="bounds",
            reconciliation_series=r,
            drift_series=d,
        )
    )
    names = tuple(m.metric_name for m in kernel.metrics)
    assert names == (
        "residual_control_pressure",
        "stability_correction_score",
        "continuity_recovery_priority",
        "severity_feedback_gain",
        "replay_control_stability",
        "calibration_feedback_signal",
        "bounded_control_confidence",
    )
    for m in kernel.metrics:
        assert 0.0 <= m.metric_value <= 1.0


def test_recommendation_stability_across_runs() -> None:
    r, d = _high_residual_inputs()
    scenario = build_feedback_control_scenario(
        scenario_id="rec-stable",
        reconciliation_series=r,
        drift_series=d,
    )
    runs = tuple(run_governance_feedback_control(scenario=scenario) for _ in range(5))
    aggregates = tuple(k.aggregate_recommendation for k in runs)
    assert len(set(aggregates)) == 1
    per_entry_sets = tuple(tuple(k.recommendations) for k in runs)
    assert len(set(per_entry_sets)) == 1
    for k in runs:
        for _rid, rec in k.recommendations:
            assert rec in {
                "observe_only",
                "monitor",
                "stabilize",
                "recalibrate",
                "isolate",
            }


def test_receipt_builder_determinism() -> None:
    r, d = _low_residual_inputs()
    scenario = build_feedback_control_scenario(
        scenario_id="receipt-det",
        reconciliation_series=r,
        drift_series=d,
    )
    kernel = run_governance_feedback_control(scenario=scenario)
    r1 = build_feedback_control_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        recommendations=kernel.recommendations,
        aggregate_recommendation=kernel.aggregate_recommendation,
        control_hash=kernel.control_hash,
    )
    r2 = build_feedback_control_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        recommendations=kernel.recommendations,
        aggregate_recommendation=kernel.aggregate_recommendation,
        control_hash=kernel.control_hash,
    )
    assert r1.to_canonical_json() == r2.to_canonical_json()
    assert r1.receipt_hash == r2.receipt_hash


def test_no_mutation_of_inputs() -> None:
    r, d = _low_residual_inputs()
    original_r = tuple(dict(item) for item in r)
    original_d = tuple(dict(item) for item in d)
    scenario = build_feedback_control_scenario(
        scenario_id="no-mutate",
        reconciliation_series=r,
        drift_series=d,
    )
    run_governance_feedback_control(scenario=scenario)
    assert r == original_r
    assert d == original_d


def test_summary_contains_hashes_and_recommendation() -> None:
    r, d = _high_residual_inputs()
    kernel = run_governance_feedback_control(
        scenario=build_feedback_control_scenario(
            scenario_id="summary",
            reconciliation_series=r,
            drift_series=d,
        )
    )
    text = summarize_feedback_control(kernel)
    assert kernel.control_hash in text
    assert kernel.receipt.receipt_hash in text
    assert f"aggregate_recommendation={kernel.aggregate_recommendation}" in text
    assert "advisory_only=True" in text


def test_receipt_integrity_mismatch_detection() -> None:
    r, d = _low_residual_inputs()
    scenario = build_feedback_control_scenario(
        scenario_id="integrity",
        reconciliation_series=r,
        drift_series=d,
    )
    kernel = run_governance_feedback_control(scenario=scenario)
    forged_receipt = ForecastFeedbackControlReceipt(
        scenario_hash="0" * 64,
        metrics_hash=kernel.receipt.metrics_hash,
        recommendations_hash="1" * 64,
        control_hash="2" * 64,
        bounded_control_confidence=kernel.receipt.bounded_control_confidence,
        receipt_hash=kernel.receipt.receipt_hash,
    )
    forged_kernel = GovernanceForecastFeedbackControlKernel(
        scenario=kernel.scenario,
        metrics=kernel.metrics,
        recommendations=kernel.recommendations,
        aggregate_recommendation=kernel.aggregate_recommendation,
        violations=kernel.violations,
        receipt=forged_receipt,
        control_hash=kernel.control_hash,
        advisory_only=True,
    )
    violations = validate_feedback_control(forged_kernel)
    assert "receipt_scenario_hash_mismatch" in violations
    assert "receipt_recommendations_hash_mismatch" in violations
    assert "receipt_control_hash_mismatch" in violations
    assert "receipt_hash_mismatch" in violations


def test_dataclass_support_methods() -> None:
    scenario = ForecastFeedbackControlScenario(
        scenario_id="x",
        reconciliation_series=(),
        drift_series=(),
    )
    metric = ForecastFeedbackControlMetric(
        metric_name="residual_control_pressure",
        metric_order=0,
        metric_value=0.0,
    )
    receipt = ForecastFeedbackControlReceipt(
        scenario_hash="a" * 64,
        metrics_hash="b" * 64,
        recommendations_hash="c" * 64,
        control_hash="d" * 64,
        bounded_control_confidence=0.0,
        receipt_hash="e" * 64,
    )
    kernel = GovernanceForecastFeedbackControlKernel(
        scenario=scenario,
        metrics=(metric,),
        recommendations=(),
        aggregate_recommendation="observe_only",
        violations=(),
        receipt=receipt,
        control_hash="f" * 64,
        advisory_only=True,
    )
    assert isinstance(scenario.to_dict(), dict)
    assert isinstance(metric.to_dict(), dict)
    assert isinstance(receipt.to_dict(), dict)
    assert isinstance(kernel.to_dict(), dict)
    assert scenario.stable_hash() == scenario.stable_hash()
    assert metric.stable_hash() == metric.stable_hash()
    assert receipt.stable_hash() == receipt.receipt_hash
    assert kernel.stable_hash() == kernel.control_hash
    assert kernel.advisory_only is True
