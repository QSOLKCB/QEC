import json
from pathlib import Path

from qec.orchestration.governance_forecast_topology_reconciliation_kernel import (
    ForecastTopologyReconciliationMetric,
    ForecastTopologyReconciliationReceipt,
    ForecastTopologyReconciliationScenario,
    GovernanceForecastTopologyReconciliationKernel,
    build_forecast_topology_reconciliation_receipt,
    build_forecast_topology_reconciliation_scenario,
    compare_forecast_topology_replay,
    run_governance_forecast_topology_reconciliation,
    summarize_forecast_topology_reconciliation,
    validate_forecast_topology_reconciliation,
)


def _metric_map(kernel):
    return {item.metric_name: item.metric_value for item in kernel.metrics}


def _stable_inputs():
    forecast_series = (
        {
            "forecast_id": "f0",
            "topology_id": "t0",
            "forecast_stability": 0.95,
            "forecast_alignment": 0.95,
            "forecast_pressure": 0.03,
            "replay_identity": "r0",
        },
        {
            "forecast_id": "f1",
            "topology_id": "t1",
            "forecast_stability": 0.94,
            "forecast_alignment": 0.96,
            "forecast_pressure": 0.04,
            "replay_identity": "r0",
        },
    )
    topology_series = (
        {
            "topology_id": "t0",
            "observed_stability": 0.95,
            "observed_alignment": 0.95,
            "observed_pressure": 0.03,
            "continuity_ok": True,
            "replay_identity": "r0",
        },
        {
            "topology_id": "t1",
            "observed_stability": 0.94,
            "observed_alignment": 0.96,
            "observed_pressure": 0.04,
            "continuity_ok": True,
            "replay_identity": "r0",
        },
    )
    replay_horizon_series = (
        {"horizon_id": "h1", "horizon_step": 1.0, "reconciliation_delta": 0.00},
        {"horizon_id": "h2", "horizon_step": 2.0, "reconciliation_delta": 0.00},
    )
    return forecast_series, topology_series, replay_horizon_series


def _minor_inputs():
    forecast_series = (
        {
            "forecast_id": "f0",
            "topology_id": "t0",
            "forecast_stability": 0.95,
            "forecast_alignment": 0.90,
            "forecast_pressure": 0.15,
            "replay_identity": "r0",
        },
    )
    topology_series = (
        {
            "topology_id": "t0",
            "observed_stability": 0.90,
            "observed_alignment": 0.95,
            "observed_pressure": 0.10,
            "continuity_ok": True,
            "replay_identity": "r0",
        },
    )
    replay_horizon_series = (
        {"horizon_id": "h1", "horizon_step": 1.0, "reconciliation_delta": 0.30},
    )
    return forecast_series, topology_series, replay_horizon_series


def _moderate_inputs():
    forecast_series = (
        {
            "forecast_id": "f0",
            "topology_id": "t0",
            "forecast_stability": 0.80,
            "forecast_alignment": 0.80,
            "forecast_pressure": 0.10,
            "replay_identity": "r0",
        },
    )
    topology_series = (
        {
            "topology_id": "t0",
            "observed_stability": 0.60,
            "observed_alignment": 0.60,
            "observed_pressure": 0.40,
            "continuity_ok": True,
            "replay_identity": "r0",
        },
    )
    replay_horizon_series = (
        {"horizon_id": "h1", "horizon_step": 1.0, "reconciliation_delta": 0.80},
    )
    return forecast_series, topology_series, replay_horizon_series


def _severe_inputs():
    forecast_series = (
        {
            "forecast_id": "f0",
            "topology_id": "t0",
            "forecast_stability": 1.00,
            "forecast_alignment": 1.00,
            "forecast_pressure": 0.00,
            "replay_identity": "r-good",
        },
    )
    topology_series = (
        {
            "topology_id": "t0",
            "observed_stability": 0.00,
            "observed_alignment": 0.00,
            "observed_pressure": 1.00,
            "continuity_ok": False,
            "replay_identity": "r-bad",
        },
    )
    replay_horizon_series = (
        {"horizon_id": "h1", "horizon_step": 1.0, "reconciliation_delta": 1.00},
    )
    return forecast_series, topology_series, replay_horizon_series


def test_deterministic_repeated_reconciliation() -> None:
    f, t, h = _stable_inputs()
    scenario = build_forecast_topology_reconciliation_scenario(
        scenario_id="repeatable",
        forecast_series=f,
        topology_series=t,
        replay_horizon_series=h,
    )
    first = run_governance_forecast_topology_reconciliation(scenario=scenario)
    second = run_governance_forecast_topology_reconciliation(scenario=scenario)
    assert first.to_canonical_json() == second.to_canonical_json()


def test_stable_hash_reproducibility() -> None:
    f, t, h = _stable_inputs()
    scenario_a = build_forecast_topology_reconciliation_scenario(
        scenario_id="hash-repro",
        forecast_series=f,
        topology_series=t,
        replay_horizon_series=h,
    )
    scenario_b = build_forecast_topology_reconciliation_scenario(
        scenario_id="hash-repro",
        forecast_series=f,
        topology_series=t,
        replay_horizon_series=h,
    )
    assert scenario_a.stable_hash() == scenario_b.stable_hash()


def test_all_advisory_bands() -> None:
    cases = (
        (_stable_inputs(), "stable_reconciliation"),
        (_minor_inputs(), "minor_reconciliation_variation"),
        (_moderate_inputs(), "moderate_reconciliation_instability"),
        (_severe_inputs(), "severe_reconciliation_instability"),
    )
    for (forecast_series, topology_series, replay_horizon_series), expected in cases:
        scenario = build_forecast_topology_reconciliation_scenario(
            scenario_id=f"band-{expected}",
            forecast_series=forecast_series,
            topology_series=topology_series,
            replay_horizon_series=replay_horizon_series,
        )
        kernel = run_governance_forecast_topology_reconciliation(scenario=scenario)
        assert kernel.advisory_output == expected


def test_canonical_json_round_trip() -> None:
    f, t, h = _severe_inputs()
    kernel = run_governance_forecast_topology_reconciliation(
        scenario=build_forecast_topology_reconciliation_scenario(
            scenario_id="roundtrip",
            forecast_series=f,
            topology_series=t,
            replay_horizon_series=h,
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


def test_validator_never_raises() -> None:
    violations = validate_forecast_topology_reconciliation(object())
    assert isinstance(violations, tuple)
    assert "missing_scenario" in violations


def test_malformed_input_handling() -> None:
    scenario = build_forecast_topology_reconciliation_scenario(
        scenario_id="",
        forecast_series={"bad": "type"},
        topology_series=None,
        replay_horizon_series=None,
    )
    kernel = run_governance_forecast_topology_reconciliation(scenario=scenario)
    assert "empty_scenario_id" in kernel.violations
    assert "empty_forecast_series" in kernel.violations
    assert "empty_topology_series" in kernel.violations


def test_metric_bounds_and_fixed_order() -> None:
    f, t, h = _severe_inputs()
    kernel = run_governance_forecast_topology_reconciliation(
        scenario=build_forecast_topology_reconciliation_scenario(
            scenario_id="order",
            forecast_series=f,
            topology_series=t,
            replay_horizon_series=h,
        )
    )
    names = tuple(metric.metric_name for metric in kernel.metrics)
    assert names == (
        "forecast_topology_delta_score",
        "reconciliation_alignment_score",
        "horizon_reconciliation_gradient",
        "forecast_drift_pressure",
        "replay_reconciliation_stability_score",
        "reconciliation_confidence_score",
    )
    values = _metric_map(kernel)
    for name in names:
        assert 0.0 <= values[name] <= 1.0


def test_replay_comparison_stability() -> None:
    f, t, h = _stable_inputs()
    scenario = build_forecast_topology_reconciliation_scenario(
        scenario_id="replay",
        forecast_series=f,
        topology_series=t,
        replay_horizon_series=h,
    )
    baseline = run_governance_forecast_topology_reconciliation(scenario=scenario)
    replay = run_governance_forecast_topology_reconciliation(scenario=scenario)
    report = compare_forecast_topology_replay(baseline, replay)
    assert report["is_stable_replay"] is True
    assert report["mismatches"] == ()


def test_no_input_mutation() -> None:
    forecast_series = [{"forecast_id": "f0", "topology_id": "t0", "forecast_stability": 1.0}]
    topology_series = [{"topology_id": "t0", "observed_stability": 0.9, "continuity_ok": True}]
    replay_horizon_series = [{"horizon_id": "h0", "horizon_step": 1.0, "reconciliation_delta": 0.1}]
    f_before = tuple(dict(row) for row in forecast_series)
    t_before = tuple(dict(row) for row in topology_series)
    h_before = tuple(dict(row) for row in replay_horizon_series)

    _ = build_forecast_topology_reconciliation_scenario(
        scenario_id="immut",
        forecast_series=forecast_series,
        topology_series=topology_series,
        replay_horizon_series=replay_horizon_series,
    )

    assert tuple(dict(row) for row in forecast_series) == f_before
    assert tuple(dict(row) for row in topology_series) == t_before
    assert tuple(dict(row) for row in replay_horizon_series) == h_before


def test_decoder_untouched_confirmation() -> None:
    module_path = Path("src/qec/orchestration/governance_forecast_topology_reconciliation_kernel.py")
    source = module_path.read_text(encoding="utf-8")
    assert "qec.decoder" not in source


def test_receipt_determinism() -> None:
    f, t, h = _moderate_inputs()
    scenario = build_forecast_topology_reconciliation_scenario(
        scenario_id="receipt",
        forecast_series=f,
        topology_series=t,
        replay_horizon_series=h,
    )
    kernel = run_governance_forecast_topology_reconciliation(scenario=scenario)
    a = build_forecast_topology_reconciliation_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        reconciliation_hash=kernel.reconciliation_hash,
        advisory_output=kernel.advisory_output,
    )
    b = build_forecast_topology_reconciliation_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        reconciliation_hash=kernel.reconciliation_hash,
        advisory_output=kernel.advisory_output,
    )
    assert a.to_canonical_json() == b.to_canonical_json()


def test_summary_content() -> None:
    f, t, h = _stable_inputs()
    kernel = run_governance_forecast_topology_reconciliation(
        scenario=build_forecast_topology_reconciliation_scenario(
            scenario_id="summary",
            forecast_series=f,
            topology_series=t,
            replay_horizon_series=h,
        )
    )
    text = summarize_forecast_topology_reconciliation(kernel)
    assert "scenario_id=summary" in text
    assert kernel.reconciliation_hash in text
    assert kernel.receipt.receipt_hash in text


def test_validator_recomputes_reconciliation_hash() -> None:
    f, t, h = _stable_inputs()
    scenario = build_forecast_topology_reconciliation_scenario(
        scenario_id="forged-hash",
        forecast_series=f,
        topology_series=t,
        replay_horizon_series=h,
    )
    kernel = run_governance_forecast_topology_reconciliation(scenario=scenario)
    forged_hash = "f" * 64
    forged_receipt = build_forecast_topology_reconciliation_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        reconciliation_hash=forged_hash,
        advisory_output=kernel.advisory_output,
    )
    forged_kernel = GovernanceForecastTopologyReconciliationKernel(
        scenario=kernel.scenario,
        metrics=kernel.metrics,
        reconciliation_analysis=kernel.reconciliation_analysis,
        advisory_output=kernel.advisory_output,
        violations=kernel.violations,
        receipt=forged_receipt,
        reconciliation_hash=forged_hash,
    )

    violations = validate_forecast_topology_reconciliation(forged_kernel)
    assert "receipt_reconciliation_hash_mismatch" in violations


def test_dataclass_support_methods() -> None:
    scenario = ForecastTopologyReconciliationScenario(
        scenario_id="x",
        forecast_series=(),
        topology_series=(),
        replay_horizon_series=(),
    )
    metric = ForecastTopologyReconciliationMetric(
        metric_name="forecast_topology_delta_score",
        metric_order=0,
        metric_value=0.0,
    )
    receipt = ForecastTopologyReconciliationReceipt(
        scenario_hash="a" * 64,
        metrics_hash="b" * 64,
        reconciliation_hash="c" * 64,
        reconciliation_confidence_score=1.0,
        advisory_output="stable_reconciliation",
        receipt_hash="d" * 64,
    )
    kernel = GovernanceForecastTopologyReconciliationKernel(
        scenario=scenario,
        metrics=(metric,),
        reconciliation_analysis={},
        advisory_output="stable_reconciliation",
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
