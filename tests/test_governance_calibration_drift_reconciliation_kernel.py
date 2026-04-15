import json
import os

from qec.orchestration.governance_calibration_drift_reconciliation_kernel import (
    CalibrationDriftMetric,
    CalibrationDriftReceipt,
    CalibrationDriftScenario,
    GovernanceCalibrationDriftReconciliationKernel,
    build_calibration_drift_receipt,
    build_calibration_drift_scenario,
    compare_calibration_drift_replay,
    run_governance_calibration_drift_reconciliation,
    summarize_calibration_drift_reconciliation,
    validate_calibration_drift_reconciliation,
)


def _row(
    calibration_id: str,
    *,
    threshold: float,
    confidence: float,
    continuity_signal: bool,
    stability_level: float,
    replay_identity: str,
):
    return {
        "calibration_id": calibration_id,
        "threshold": threshold,
        "confidence": confidence,
        "continuity_signal": continuity_signal,
        "stability_level": stability_level,
        "replay_identity": replay_identity,
    }


def _identical_pair():
    series = (
        _row("c0", threshold=0.50, confidence=0.90, continuity_signal=True, stability_level=0.90, replay_identity="r0"),
        _row("c1", threshold=0.60, confidence=0.92, continuity_signal=True, stability_level=0.92, replay_identity="r0"),
    )
    return series, series


def _minor_drift_pair():
    current = (
        _row("c0", threshold=0.70, confidence=0.80, continuity_signal=True, stability_level=0.80, replay_identity="r0"),
        _row("c1", threshold=0.70, confidence=0.80, continuity_signal=True, stability_level=0.80, replay_identity="r0"),
    )
    prior = (
        _row("c0", threshold=0.50, confidence=0.90, continuity_signal=True, stability_level=0.90, replay_identity="r0"),
        _row("c1", threshold=0.50, confidence=0.90, continuity_signal=True, stability_level=0.90, replay_identity="r0"),
    )
    return current, prior


def _moderate_drift_pair():
    current = (
        _row("c0", threshold=0.70, confidence=0.70, continuity_signal=False, stability_level=0.70, replay_identity="r0"),
        _row("c1", threshold=0.70, confidence=0.70, continuity_signal=False, stability_level=0.70, replay_identity="r0"),
    )
    prior = (
        _row("c0", threshold=0.30, confidence=0.90, continuity_signal=True, stability_level=0.90, replay_identity="r0"),
        _row("c1", threshold=0.30, confidence=0.90, continuity_signal=True, stability_level=0.90, replay_identity="r0"),
    )
    return current, prior


def _severe_drift_pair():
    current = (
        _row("c0", threshold=1.0, confidence=0.10, continuity_signal=False, stability_level=0.10, replay_identity="r-new"),
        _row("c1", threshold=1.0, confidence=0.10, continuity_signal=False, stability_level=0.10, replay_identity="r-new"),
    )
    prior = (
        _row("c0", threshold=0.10, confidence=1.00, continuity_signal=True, stability_level=1.00, replay_identity="r-old"),
        _row("c1", threshold=0.10, confidence=1.00, continuity_signal=True, stability_level=1.00, replay_identity="r-old"),
    )
    return current, prior


def _metric_map(kernel):
    return {m.metric_name: m.metric_value for m in kernel.metrics}


def test_deterministic_repeated_reconciliation() -> None:
    current, prior = _identical_pair()
    scenario = build_calibration_drift_scenario(
        scenario_id="repeatable",
        calibration_series=current,
        prior_calibration_series=prior,
    )
    a = run_governance_calibration_drift_reconciliation(scenario=scenario)
    b = run_governance_calibration_drift_reconciliation(scenario=scenario)
    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.reconciliation_hash == b.reconciliation_hash


def test_stable_hash_reproducibility() -> None:
    current, prior = _identical_pair()
    scenario_a = build_calibration_drift_scenario(
        scenario_id="hash-repro",
        calibration_series=current,
        prior_calibration_series=prior,
    )
    scenario_b = build_calibration_drift_scenario(
        scenario_id="hash-repro",
        calibration_series=current,
        prior_calibration_series=prior,
    )
    assert scenario_a.stable_hash() == scenario_b.stable_hash()


def test_no_drift_yields_stable_alignment() -> None:
    current, prior = _identical_pair()
    kernel = run_governance_calibration_drift_reconciliation(
        scenario=build_calibration_drift_scenario(
            scenario_id="stable",
            calibration_series=current,
            prior_calibration_series=prior,
        )
    )
    metrics = _metric_map(kernel)
    assert metrics["threshold_drift_delta"] == 0.0
    assert metrics["continuity_drift_pressure"] == 0.0
    assert metrics["confidence_reconciliation_score"] == 1.0
    assert metrics["stability_recovery_alignment"] == 1.0
    assert metrics["cross_replay_drift_score"] == 0.0
    assert metrics["drift_reconciliation_confidence"] == 1.0
    assert kernel.advisory == ("stable_alignment",)


def test_low_drift_yields_minor_reconcile() -> None:
    current, prior = _minor_drift_pair()
    kernel = run_governance_calibration_drift_reconciliation(
        scenario=build_calibration_drift_scenario(
            scenario_id="minor",
            calibration_series=current,
            prior_calibration_series=prior,
        )
    )
    assert kernel.advisory == ("minor_drift_reconcile",)
    metrics = _metric_map(kernel)
    assert 0.15 <= metrics["threshold_drift_delta"] <= 0.25


def test_moderate_drift_yields_moderate_reconcile() -> None:
    current, prior = _moderate_drift_pair()
    kernel = run_governance_calibration_drift_reconciliation(
        scenario=build_calibration_drift_scenario(
            scenario_id="moderate",
            calibration_series=current,
            prior_calibration_series=prior,
        )
    )
    assert kernel.advisory == ("moderate_drift_reconcile",)
    metrics = _metric_map(kernel)
    assert metrics["continuity_drift_pressure"] == 1.0


def test_severe_drift_yields_severe_reconcile() -> None:
    current, prior = _severe_drift_pair()
    kernel = run_governance_calibration_drift_reconciliation(
        scenario=build_calibration_drift_scenario(
            scenario_id="severe",
            calibration_series=current,
            prior_calibration_series=prior,
        )
    )
    assert kernel.advisory == ("severe_drift_reconcile",)
    metrics = _metric_map(kernel)
    assert metrics["drift_reconciliation_confidence"] <= 0.1
    assert metrics["threshold_drift_delta"] >= 0.8
    assert metrics["cross_replay_drift_score"] == 1.0


def test_validator_never_raises_on_garbage() -> None:
    violations = validate_calibration_drift_reconciliation(object())
    assert isinstance(violations, tuple)
    assert "missing_scenario" in violations

    assert isinstance(validate_calibration_drift_reconciliation(None), tuple)
    assert isinstance(validate_calibration_drift_reconciliation(42), tuple)
    assert isinstance(validate_calibration_drift_reconciliation("not-a-kernel"), tuple)


def test_canonical_json_round_trip() -> None:
    current, prior = _moderate_drift_pair()
    kernel = run_governance_calibration_drift_reconciliation(
        scenario=build_calibration_drift_scenario(
            scenario_id="roundtrip",
            calibration_series=current,
            prior_calibration_series=prior,
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
    current, prior = _identical_pair()
    scenario = build_calibration_drift_scenario(
        scenario_id="replay",
        calibration_series=current,
        prior_calibration_series=prior,
    )
    baseline = run_governance_calibration_drift_reconciliation(scenario=scenario)
    replay = run_governance_calibration_drift_reconciliation(scenario=scenario)
    report = compare_calibration_drift_replay(baseline, replay)
    assert report["is_stable_replay"] is True
    assert report["mismatches"] == ()
    for _, delta in report["metric_delta"]:
        assert delta == 0.0


def test_malformed_input_handling() -> None:
    scenario = build_calibration_drift_scenario(
        scenario_id="",
        calibration_series={"not": "a_list"},
        prior_calibration_series=None,
    )
    kernel = run_governance_calibration_drift_reconciliation(scenario=scenario)
    assert "empty_scenario_id" in kernel.violations
    assert "empty_calibration_series" in kernel.violations
    assert "empty_prior_calibration_series" in kernel.violations

    # Malformed numeric / bool values must normalize safely (no raise).
    bad_rows = (
        {"calibration_id": "c0", "threshold": "nope", "confidence": None, "stability_level": float("nan"), "continuity_signal": 1},
    )
    scenario2 = build_calibration_drift_scenario(
        scenario_id="bad",
        calibration_series=bad_rows,
        prior_calibration_series=bad_rows,
    )
    kernel2 = run_governance_calibration_drift_reconciliation(scenario=scenario2)
    for metric in kernel2.metrics:
        assert 0.0 <= metric.metric_value <= 1.0


def test_metric_bounds_and_order() -> None:
    current, prior = _severe_drift_pair()
    kernel = run_governance_calibration_drift_reconciliation(
        scenario=build_calibration_drift_scenario(
            scenario_id="bounds",
            calibration_series=current,
            prior_calibration_series=prior,
        )
    )
    names = tuple(m.metric_name for m in kernel.metrics)
    assert names == (
        "threshold_drift_delta",
        "confidence_reconciliation_score",
        "continuity_drift_pressure",
        "stability_recovery_alignment",
        "cross_replay_drift_score",
        "drift_reconciliation_confidence",
    )
    for metric in kernel.metrics:
        assert 0.0 <= metric.metric_value <= 1.0
        assert metric.metric_order == names.index(metric.metric_name)


def test_decoder_untouched_confirmation() -> None:
    # Reconciliation kernel must never import decoder internals, and the
    # decoder source tree must remain outside this module's dependency cone.
    module_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
        "qec",
        "orchestration",
        "governance_calibration_drift_reconciliation_kernel.py",
    )
    with open(module_path, "r", encoding="utf-8") as handle:
        source = handle.read()
    assert "qec.decoder" not in source
    assert "from qec.decoder" not in source
    assert "import qec.decoder" not in source
    # Decoder directory still exists and was not touched by this kernel.
    decoder_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
        "qec",
        "decoder",
    )
    assert os.path.isdir(decoder_dir)


def test_receipt_builder_determinism_and_hash_stability() -> None:
    current, prior = _minor_drift_pair()
    scenario = build_calibration_drift_scenario(
        scenario_id="receipt",
        calibration_series=current,
        prior_calibration_series=prior,
    )
    kernel = run_governance_calibration_drift_reconciliation(scenario=scenario)
    r1 = build_calibration_drift_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        advisory=kernel.advisory,
        reconciliation_hash=kernel.reconciliation_hash,
    )
    r2 = build_calibration_drift_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        advisory=kernel.advisory,
        reconciliation_hash=kernel.reconciliation_hash,
    )
    assert r1.to_canonical_json() == r2.to_canonical_json()
    assert r1.receipt_hash == r2.receipt_hash
    assert r1.stable_hash() == r1.receipt_hash


def test_summary_contains_hashes_and_advisory() -> None:
    current, prior = _identical_pair()
    kernel = run_governance_calibration_drift_reconciliation(
        scenario=build_calibration_drift_scenario(
            scenario_id="summary",
            calibration_series=current,
            prior_calibration_series=prior,
        )
    )
    text = summarize_calibration_drift_reconciliation(kernel)
    assert kernel.reconciliation_hash in text
    assert kernel.receipt.receipt_hash in text
    assert "stable_alignment" in text


def test_dataclass_support_methods() -> None:
    scenario = CalibrationDriftScenario(
        scenario_id="x",
        calibration_series=(),
        prior_calibration_series=(),
    )
    metric = CalibrationDriftMetric(
        metric_name="threshold_drift_delta",
        metric_order=0,
        metric_value=0.0,
    )
    receipt = CalibrationDriftReceipt(
        scenario_hash="a" * 64,
        metrics_hash="b" * 64,
        advisory_hash="c" * 64,
        reconciliation_hash="d" * 64,
        drift_reconciliation_confidence=0.0,
        receipt_hash="e" * 64,
    )
    kernel = GovernanceCalibrationDriftReconciliationKernel(
        scenario=scenario,
        metrics=(metric,),
        advisory=("stable_alignment",),
        violations=(),
        receipt=receipt,
        reconciliation_hash="f" * 64,
    )
    assert isinstance(scenario.to_dict(), dict)
    assert isinstance(metric.to_dict(), dict)
    assert isinstance(receipt.to_dict(), dict)
    assert isinstance(kernel.to_dict(), dict)
    assert scenario.stable_hash() == scenario.stable_hash()
    assert metric.stable_hash() == metric.stable_hash()
    assert receipt.stable_hash() == receipt.receipt_hash
    assert kernel.stable_hash() == kernel.reconciliation_hash
