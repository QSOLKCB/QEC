import json
import os

from qec.orchestration.governance_control_stability_calibration_kernel import (
    ControlCalibrationMetric,
    ControlCalibrationReceipt,
    ControlCalibrationScenario,
    GovernanceControlStabilityCalibrationKernel,
    build_control_calibration_receipt,
    build_control_calibration_scenario,
    compare_control_calibration_replay,
    run_governance_control_calibration,
    summarize_control_calibration,
    validate_control_calibration,
)


def _rec(idx, level):
    return {
        "reconciliation_id": f"r{idx}",
        "residual_magnitude": level,
        "stability_residual": level,
        "continuity_residual": level,
        "severity_residual": level,
        "replay_residual": level,
        "calibration_residual": level,
    }


def _control(level):
    return {
        "residual_control_pressure": level,
        "stability_correction_score": level,
        "continuity_recovery_priority": level,
        "severity_feedback_gain": level,
        "replay_control_stability": 1.0 - level,
        "calibration_feedback_signal": level,
        "bounded_control_confidence": 1.0 - level,
        "aggregate_recommendation": "monitor" if level < 0.5 else "stabilize",
    }


def _scenario(level, sid="s"):
    return build_control_calibration_scenario(
        scenario_id=sid,
        reconciliation_series=(_rec(0, level), _rec(1, level)),
        control_analysis=_control(level),
    )


def _metric_map(kernel):
    return {m.metric_name: m.metric_value for m in kernel.metrics}


def test_deterministic_repeated_calibration():
    s = _scenario(0.05, "det")
    a = run_governance_control_calibration(scenario=s)
    b = run_governance_control_calibration(scenario=s)
    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.calibration_hash == b.calibration_hash
    assert a.receipt.receipt_hash == b.receipt.receipt_hash


def test_stable_hash_reproducibility():
    a = _scenario(0.30, "hash")
    b = _scenario(0.30, "hash")
    assert a.stable_hash() == b.stable_hash()
    ka = run_governance_control_calibration(scenario=a)
    kb = run_governance_control_calibration(scenario=b)
    assert ka.calibration_hash == kb.calibration_hash
    assert ka.receipt.receipt_hash == kb.receipt.receipt_hash


def test_low_residual_holds_thresholds():
    kernel = run_governance_control_calibration(scenario=_scenario(0.05, "low"))
    metrics = _metric_map(kernel)
    assert metrics["threshold_drift_score"] < 0.2
    assert metrics["control_gain_calibration"] < 0.2
    assert metrics["calibration_confidence_score"] > 0.8
    assert kernel.aggregate_advisory == "hold_thresholds"
    for _rid, adv in kernel.advisories:
        assert adv == "hold_thresholds"


def test_moderate_residual_soft_adjust():
    kernel = run_governance_control_calibration(scenario=_scenario(0.30, "mod"))
    metrics = _metric_map(kernel)
    assert 0.2 <= metrics["threshold_drift_score"] < 0.5
    assert kernel.aggregate_advisory == "soft_adjust"
    for _rid, adv in kernel.advisories:
        assert adv == "soft_adjust"


def test_elevated_residual_moderate_recalibration():
    kernel = run_governance_control_calibration(scenario=_scenario(0.60, "elev"))
    metrics = _metric_map(kernel)
    assert 0.5 <= metrics["threshold_drift_score"] < 0.8
    assert kernel.aggregate_advisory == "moderate_recalibration"
    for _rid, adv in kernel.advisories:
        assert adv == "moderate_recalibration"


def test_high_residual_aggressive_recalibration():
    kernel = run_governance_control_calibration(scenario=_scenario(0.90, "high"))
    metrics = _metric_map(kernel)
    assert metrics["threshold_drift_score"] >= 0.8
    assert metrics["calibration_confidence_score"] < 0.2
    assert kernel.aggregate_advisory == "aggressive_recalibration"
    for _rid, adv in kernel.advisories:
        assert adv == "aggressive_recalibration"


def test_advisory_output_determinism():
    s = _scenario(0.6, "adv")
    runs = tuple(run_governance_control_calibration(scenario=s) for _ in range(5))
    aggregates = tuple(k.aggregate_advisory for k in runs)
    assert len(set(aggregates)) == 1
    sets = tuple(tuple(k.advisories) for k in runs)
    assert len(set(sets)) == 1
    for k in runs:
        assert k.advisory_only is True
        for _rid, adv in k.advisories:
            assert adv in {
                "hold_thresholds",
                "soft_adjust",
                "moderate_recalibration",
                "aggressive_recalibration",
            }


def test_validator_never_raises_on_arbitrary_inputs():
    assert isinstance(validate_control_calibration(None), tuple)
    assert isinstance(validate_control_calibration(object()), tuple)
    assert isinstance(validate_control_calibration("not-a-kernel"), tuple)
    assert isinstance(validate_control_calibration(123), tuple)
    assert isinstance(validate_control_calibration([1, 2, 3]), tuple)
    violations = validate_control_calibration(None)
    assert "missing_scenario" in violations


def test_canonical_json_round_trip():
    kernel = run_governance_control_calibration(scenario=_scenario(0.6, "rt"))
    encoded = kernel.to_canonical_json()
    parsed = json.loads(encoded)
    re_encoded = json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    assert encoded == re_encoded


def test_replay_comparison_stability():
    s = _scenario(0.6, "rep")
    a = run_governance_control_calibration(scenario=s)
    b = run_governance_control_calibration(scenario=s)
    report = compare_control_calibration_replay(a, b)
    assert report["is_stable_replay"] is True
    assert report["mismatches"] == ()
    assert report["baseline_aggregate_advisory"] == report["replay_aggregate_advisory"]
    for _name, delta in report["metric_delta"]:
        assert delta == 0.0


def test_malformed_input_handling():
    scenario = build_control_calibration_scenario(
        scenario_id="",
        reconciliation_series={"not": "a-series"},
        control_analysis=None,
    )
    kernel = run_governance_control_calibration(scenario=scenario)
    assert "empty_scenario_id" in kernel.violations
    assert "empty_reconciliation_series" in kernel.violations
    assert kernel.aggregate_advisory == "hold_thresholds"
    assert kernel.advisory_only is True
    # Validator must remain pure-tuple even on this defensive path.
    assert isinstance(kernel.violations, tuple)


def test_metric_bounds_and_ordering():
    for level in (0.0, 0.05, 0.30, 0.60, 0.90, 1.0):
        kernel = run_governance_control_calibration(
            scenario=_scenario(level, f"b{level}")
        )
        names = tuple(m.metric_name for m in kernel.metrics)
        assert names == (
            "threshold_drift_score",
            "control_gain_calibration",
            "residual_confidence_adjustment",
            "continuity_threshold_pressure",
            "stability_gain_normalization",
            "calibration_confidence_score",
        )
        for m in kernel.metrics:
            assert 0.0 <= m.metric_value <= 1.0


def test_decoder_untouched_confirmation():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_path = os.path.join(
        here,
        "src",
        "qec",
        "orchestration",
        "governance_control_stability_calibration_kernel.py",
    )
    with open(module_path, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert "qec.decoder" not in source
    assert "src/qec/decoder" not in source
    assert "from qec.decoder" not in source
    assert "import qec.decoder" not in source


def test_receipt_integrity_mismatch_detection():
    kernel = run_governance_control_calibration(scenario=_scenario(0.05, "integ"))
    forged = ControlCalibrationReceipt(
        scenario_hash="0" * 64,
        metrics_hash=kernel.receipt.metrics_hash,
        advisories_hash="1" * 64,
        calibration_hash="2" * 64,
        calibration_confidence_score=kernel.receipt.calibration_confidence_score,
        receipt_hash=kernel.receipt.receipt_hash,
    )
    forged_kernel = GovernanceControlStabilityCalibrationKernel(
        scenario=kernel.scenario,
        metrics=kernel.metrics,
        advisories=kernel.advisories,
        aggregate_advisory=kernel.aggregate_advisory,
        violations=kernel.violations,
        receipt=forged,
        calibration_hash=kernel.calibration_hash,
        advisory_only=True,
    )
    violations = validate_control_calibration(forged_kernel)
    assert "receipt_scenario_hash_mismatch" in violations
    assert "receipt_advisories_hash_mismatch" in violations
    assert "receipt_calibration_hash_mismatch" in violations
    assert "receipt_hash_mismatch" in violations


def test_dataclass_support_methods():
    scenario = ControlCalibrationScenario(
        scenario_id="x",
        reconciliation_series=(),
        control_analysis={},
    )
    metric = ControlCalibrationMetric(
        metric_name="threshold_drift_score",
        metric_order=0,
        metric_value=0.0,
    )
    receipt = ControlCalibrationReceipt(
        scenario_hash="a" * 64,
        metrics_hash="b" * 64,
        advisories_hash="c" * 64,
        calibration_hash="d" * 64,
        calibration_confidence_score=0.0,
        receipt_hash="e" * 64,
    )
    kernel = GovernanceControlStabilityCalibrationKernel(
        scenario=scenario,
        metrics=(metric,),
        advisories=(),
        aggregate_advisory="hold_thresholds",
        violations=(),
        receipt=receipt,
        calibration_hash="f" * 64,
        advisory_only=True,
    )
    assert isinstance(scenario.to_dict(), dict)
    assert isinstance(metric.to_dict(), dict)
    assert isinstance(receipt.to_dict(), dict)
    assert isinstance(kernel.to_dict(), dict)
    assert isinstance(scenario.to_canonical_json(), str)
    assert isinstance(metric.to_canonical_json(), str)
    assert isinstance(receipt.to_canonical_json(), str)
    assert isinstance(kernel.to_canonical_json(), str)
    assert scenario.stable_hash() == scenario.stable_hash()
    assert metric.stable_hash() == metric.stable_hash()
    assert receipt.stable_hash() == receipt.receipt_hash
    assert kernel.stable_hash() == kernel.calibration_hash
    assert kernel.advisory_only is True


def test_no_mutation_of_inputs():
    rec = (_rec(0, 0.30), _rec(1, 0.30))
    ctrl = _control(0.30)
    original_rec = tuple(dict(item) for item in rec)
    original_ctrl = dict(ctrl)
    scenario = build_control_calibration_scenario(
        scenario_id="no-mutate",
        reconciliation_series=rec,
        control_analysis=ctrl,
    )
    run_governance_control_calibration(scenario=scenario)
    assert rec == original_rec
    assert ctrl == original_ctrl


def test_summary_contains_key_fields():
    kernel = run_governance_control_calibration(scenario=_scenario(0.6, "summary"))
    text = summarize_control_calibration(kernel)
    assert kernel.calibration_hash in text
    assert kernel.receipt.receipt_hash in text
    assert f"aggregate_advisory={kernel.aggregate_advisory}" in text
    assert "advisory_only=True" in text
    assert "threshold_drift_score" in text


def test_receipt_builder_determinism():
    s = _scenario(0.05, "receipt-det")
    kernel = run_governance_control_calibration(scenario=s)
    r1 = build_control_calibration_receipt(
        scenario=s,
        metrics=kernel.metrics,
        advisories=kernel.advisories,
        aggregate_advisory=kernel.aggregate_advisory,
        calibration_hash=kernel.calibration_hash,
    )
    r2 = build_control_calibration_receipt(
        scenario=s,
        metrics=kernel.metrics,
        advisories=kernel.advisories,
        aggregate_advisory=kernel.aggregate_advisory,
        calibration_hash=kernel.calibration_hash,
    )
    assert r1.to_canonical_json() == r2.to_canonical_json()
    assert r1.receipt_hash == r2.receipt_hash
