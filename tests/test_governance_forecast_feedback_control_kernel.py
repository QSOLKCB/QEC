import json

from qec.orchestration.governance_forecast_feedback_control_kernel import (
    FeedbackControlMetric,
    FeedbackControlReceipt,
    FeedbackControlScenario,
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
            "stability_accuracy": 0.95,
            "continuity_error": 0.05,
            "severity_residual": 0.05,
            "replay_score": 0.99,
            "confidence_calibration": 0.95,
        },
        {
            "reconciliation_id": "r1",
            "residual_magnitude": 0.04,
            "stability_accuracy": 0.96,
            "continuity_error": 0.04,
            "severity_residual": 0.06,
            "replay_score": 0.98,
            "confidence_calibration": 0.94,
        },
    )
    drift_series = (
        {
            "drift_id": "d0",
            "from_basin": "A",
            "to_basin": "A",
            "transition_count": 0,
            "drift_magnitude": 0.04,
            "drift_severity": 0.05,
        },
    )
    return reconciliation_series, drift_series


def _high_residual_inputs():
    reconciliation_series = (
        {
            "reconciliation_id": "r0",
            "residual_magnitude": 0.95,
            "stability_accuracy": 0.05,
            "continuity_error": 0.9,
            "severity_residual": 0.9,
            "replay_score": 0.1,
            "confidence_calibration": 0.1,
        },
        {
            "reconciliation_id": "r1",
            "residual_magnitude": 0.92,
            "stability_accuracy": 0.06,
            "continuity_error": 0.88,
            "severity_residual": 0.92,
            "replay_score": 0.12,
            "confidence_calibration": 0.12,
        },
    )
    drift_series = (
        {
            "drift_id": "d0",
            "from_basin": "A",
            "to_basin": "B",
            "transition_count": 4,
            "drift_magnitude": 0.95,
            "drift_severity": 0.9,
        },
        {
            "drift_id": "d1",
            "from_basin": "B",
            "to_basin": "C",
            "transition_count": 3,
            "drift_magnitude": 0.9,
            "drift_severity": 0.95,
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
    assert a.analysis_hash == b.analysis_hash
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
    assert metrics["residual_control_pressure"] < 0.15
    assert metrics["stability_correction_score"] < 0.15
    assert metrics["bounded_control_confidence"] > 0.75
    assert kernel.advisory_output in ("observe_only", "monitor")


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
    assert metrics["stability_correction_score"] > 0.8
    assert metrics["severity_feedback_gain"] > 0.8
    assert kernel.advisory_output == "isolate"


def test_advisory_output_determinism() -> None:
    r, d = _high_residual_inputs()
    scenario = build_feedback_control_scenario(
        scenario_id="advisory-det",
        reconciliation_series=r,
        drift_series=d,
    )
    outputs = tuple(
        run_governance_feedback_control(scenario=scenario).advisory_output
        for _ in range(5)
    )
    assert len(set(outputs)) == 1
    assert outputs[0] in (
        "observe_only",
        "monitor",
        "stabilize",
        "recalibrate",
        "isolate",
    )


def test_advisory_escalation_lattice_covers_expected_levels() -> None:
    # Nominal
    nominal_scenario = build_feedback_control_scenario(
        scenario_id="nominal",
        reconciliation_series=(
            {
                "reconciliation_id": "r0",
                "residual_magnitude": 0.02,
                "stability_accuracy": 0.98,
                "continuity_error": 0.02,
                "severity_residual": 0.02,
                "replay_score": 0.99,
                "confidence_calibration": 0.99,
            },
        ),
        drift_series=(
            {
                "drift_id": "d0",
                "from_basin": "A",
                "to_basin": "A",
                "drift_magnitude": 0.02,
                "drift_severity": 0.02,
            },
        ),
    )
    assert (
        run_governance_feedback_control(scenario=nominal_scenario).advisory_output
        == "observe_only"
    )

    # Critical
    critical_scenario = build_feedback_control_scenario(
        scenario_id="critical",
        reconciliation_series=(
            {
                "reconciliation_id": "r0",
                "residual_magnitude": 1.0,
                "stability_accuracy": 0.0,
                "continuity_error": 1.0,
                "severity_residual": 1.0,
                "replay_score": 0.0,
                "confidence_calibration": 0.0,
            },
        ),
        drift_series=(
            {
                "drift_id": "d0",
                "from_basin": "A",
                "to_basin": "B",
                "drift_magnitude": 1.0,
                "drift_severity": 1.0,
            },
        ),
    )
    assert (
        run_governance_feedback_control(scenario=critical_scenario).advisory_output
        == "isolate"
    )


def test_validator_never_raises_on_arbitrary_inputs() -> None:
    for bad_input in (
        object(),
        None,
        42,
        "kernel",
        {"scenario": None, "metrics": "not-a-list"},
        {"scenario": {"scenario_id": ""}, "metrics": ()},
    ):
        violations = validate_feedback_control(bad_input)
        assert isinstance(violations, tuple)


def test_validator_flags_missing_scenario() -> None:
    violations = validate_feedback_control(object())
    assert "missing_scenario" in violations


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


def test_replay_comparison_stability() -> None:
    r, d = _low_residual_inputs()
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
    for _, delta in report["metric_delta"]:
        assert delta == 0.0


def test_replay_comparison_detects_divergence() -> None:
    r, d = _low_residual_inputs()
    baseline = run_governance_feedback_control(
        scenario=build_feedback_control_scenario(
            scenario_id="replay-low",
            reconciliation_series=r,
            drift_series=d,
        )
    )
    r2, d2 = _high_residual_inputs()
    divergent = run_governance_feedback_control(
        scenario=build_feedback_control_scenario(
            scenario_id="replay-high",
            reconciliation_series=r2,
            drift_series=d2,
        )
    )
    report = compare_feedback_control_replay(baseline, divergent)
    assert report["is_stable_replay"] is False
    assert "scenario_hash" in report["mismatches"]
    assert "analysis_hash" in report["mismatches"]


def test_malformed_input_handling() -> None:
    scenario = build_feedback_control_scenario(
        scenario_id="",
        reconciliation_series={"bad": "type"},
        drift_series=None,
    )
    kernel = run_governance_feedback_control(scenario=scenario)
    assert "empty_scenario_id" in kernel.violations
    assert "empty_input_series" in kernel.violations
    assert kernel.advisory_output == "observe_only"


def test_no_mutation_of_inputs() -> None:
    r, d = _low_residual_inputs()
    original_r = tuple(dict(item) for item in r)
    original_d = tuple(dict(item) for item in d)
    scenario = build_feedback_control_scenario(
        scenario_id="no-mutate",
        reconciliation_series=r,
        drift_series=d,
    )

    original_scenario_json = scenario.to_canonical_json()
    original_scenario_hash = scenario.stable_hash()

    # First run should not mutate the input series or the scenario.
    run_governance_feedback_control(scenario=scenario)
    assert r == original_r
    assert d == original_d
    assert scenario.to_canonical_json() == original_scenario_json
    assert scenario.stable_hash() == original_scenario_hash

    # Second run on the same scenario instance should also preserve immutability.
    run_governance_feedback_control(scenario=scenario)
    assert r == original_r
    assert d == original_d
    assert scenario.to_canonical_json() == original_scenario_json
    assert scenario.stable_hash() == original_scenario_hash


def test_validator_accepts_dict_scenario_without_raising() -> None:
    # Build a real kernel then cross-validate with a dict-shaped scenario whose
    # receipt carries the same canonical scenario_hash — the validator must
    # derive the expected hash generically and not produce a spurious mismatch.
    r, d = _low_residual_inputs()
    scenario = build_feedback_control_scenario(
        scenario_id="dict-scenario",
        reconciliation_series=r,
        drift_series=d,
    )
    kernel = run_governance_feedback_control(scenario=scenario)
    dict_like_kernel = {
        "scenario": scenario.to_dict(),
        "metrics": kernel.metrics,
        "advisory_output": kernel.advisory_output,
        "advisory_rationale": kernel.advisory_rationale,
        "receipt": kernel.receipt,
        "analysis_hash": kernel.analysis_hash,
    }
    violations = validate_feedback_control(dict_like_kernel)
    assert isinstance(violations, tuple)
    assert "receipt_scenario_hash_mismatch" not in violations
    assert not any(v.startswith("validator_internal_error") for v in violations)


def test_decoder_untouched_confirmation() -> None:
    import qec.orchestration.governance_forecast_feedback_control_kernel as module

    source = module.__file__
    with open(source, "r", encoding="utf-8") as handle:
        content = handle.read()
    assert "qec.decoder" not in content
    assert "from qec.decoder" not in content
    assert "import qec.decoder" not in content


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
        "bounded_control_confidence",
    )
    for metric in kernel.metrics:
        assert 0.0 <= metric.metric_value <= 1.0


def test_receipt_builder_determinism() -> None:
    r, d = _low_residual_inputs()
    scenario = build_feedback_control_scenario(
        scenario_id="receipt",
        reconciliation_series=r,
        drift_series=d,
    )
    kernel = run_governance_feedback_control(scenario=scenario)
    r1 = build_feedback_control_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        advisory_output=kernel.advisory_output,
        advisory_rationale=kernel.advisory_rationale,
        analysis_hash=kernel.analysis_hash,
    )
    r2 = build_feedback_control_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        advisory_output=kernel.advisory_output,
        advisory_rationale=kernel.advisory_rationale,
        analysis_hash=kernel.analysis_hash,
    )
    assert r1.to_canonical_json() == r2.to_canonical_json()
    assert r1.stable_hash() == r2.stable_hash()


def test_summary_contains_hashes_and_advisory() -> None:
    r, d = _high_residual_inputs()
    kernel = run_governance_feedback_control(
        scenario=build_feedback_control_scenario(
            scenario_id="summary",
            reconciliation_series=r,
            drift_series=d,
        )
    )
    text = summarize_feedback_control(kernel)
    assert kernel.analysis_hash in text
    assert kernel.receipt.receipt_hash in text
    assert kernel.advisory_output in text


def test_validator_detects_forged_receipt_integrity() -> None:
    r, d = _low_residual_inputs()
    scenario = build_feedback_control_scenario(
        scenario_id="forged",
        reconciliation_series=r,
        drift_series=d,
    )
    kernel = run_governance_feedback_control(scenario=scenario)
    forged_receipt = FeedbackControlReceipt(
        scenario_hash="0" * 64,
        metrics_hash=kernel.receipt.metrics_hash,
        advisory_hash="1" * 64,
        analysis_hash="2" * 64,
        bounded_control_confidence=kernel.receipt.bounded_control_confidence,
        advisory_output=kernel.receipt.advisory_output,
        receipt_hash=kernel.receipt.receipt_hash,
    )
    forged_kernel = GovernanceForecastFeedbackControlKernel(
        scenario=kernel.scenario,
        metrics=kernel.metrics,
        advisory_output=kernel.advisory_output,
        advisory_rationale=kernel.advisory_rationale,
        violations=kernel.violations,
        receipt=forged_receipt,
        analysis_hash=kernel.analysis_hash,
    )
    violations = validate_feedback_control(forged_kernel)
    assert "receipt_scenario_hash_mismatch" in violations
    assert "receipt_advisory_hash_mismatch" in violations
    assert "receipt_analysis_hash_mismatch" in violations
    assert "receipt_hash_mismatch" in violations


def test_dataclass_support_methods() -> None:
    scenario = FeedbackControlScenario(
        scenario_id="x",
        reconciliation_series=(),
        drift_series=(),
    )
    metric = FeedbackControlMetric(
        metric_name="residual_control_pressure",
        metric_order=0,
        metric_value=0.0,
    )
    receipt = FeedbackControlReceipt(
        scenario_hash="a" * 64,
        metrics_hash="b" * 64,
        advisory_hash="c" * 64,
        analysis_hash="d" * 64,
        bounded_control_confidence=0.0,
        advisory_output="observe_only",
        receipt_hash="e" * 64,
    )
    kernel = GovernanceForecastFeedbackControlKernel(
        scenario=scenario,
        metrics=(metric,),
        advisory_output="observe_only",
        advisory_rationale=("no_inputs",),
        violations=(),
        receipt=receipt,
        analysis_hash="f" * 64,
    )
    assert isinstance(scenario.to_dict(), dict)
    assert isinstance(metric.to_dict(), dict)
    assert isinstance(receipt.to_dict(), dict)
    assert isinstance(kernel.to_dict(), dict)
    assert scenario.stable_hash() == scenario.stable_hash()
    assert metric.stable_hash() == metric.stable_hash()
    assert receipt.stable_hash() == receipt.receipt_hash
    assert kernel.stable_hash() == kernel.analysis_hash
