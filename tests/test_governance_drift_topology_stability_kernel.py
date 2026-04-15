import json
import os

from qec.orchestration.governance_drift_topology_stability_kernel import (
    DriftTopologyMetric,
    DriftTopologyReceipt,
    DriftTopologyScenario,
    GovernanceDriftTopologyStabilityKernel,
    build_drift_topology_receipt,
    build_drift_topology_scenario,
    compare_drift_topology_replay,
    run_governance_drift_topology_stability,
    summarize_drift_topology_stability,
    validate_drift_topology_stability,
)


def _drift_row(
    reconciliation_id: str,
    *,
    drift_reconciliation_confidence: float,
    threshold_drift_delta: float = 0.0,
    continuity_drift_pressure: float = 0.0,
    stability_recovery_alignment: float = 1.0,
    cross_replay_drift_score: float = 0.0,
    advisory_label: str = "stable_alignment",
    replay_identity: str = "r0",
):
    return {
        "reconciliation_id": reconciliation_id,
        "drift_reconciliation_confidence": drift_reconciliation_confidence,
        "threshold_drift_delta": threshold_drift_delta,
        "continuity_drift_pressure": continuity_drift_pressure,
        "stability_recovery_alignment": stability_recovery_alignment,
        "cross_replay_drift_score": cross_replay_drift_score,
        "advisory_label": advisory_label,
        "replay_identity": replay_identity,
    }


def _horizon_row(
    horizon_id: str,
    *,
    horizon_index: int,
    continuity_flag: bool = True,
    replay_identity: str = "r0",
    weight: float = 1.0,
):
    return {
        "horizon_id": horizon_id,
        "horizon_index": horizon_index,
        "continuity_flag": continuity_flag,
        "replay_identity": replay_identity,
        "weight": weight,
    }


def _stable_inputs():
    drifts = (
        _drift_row("d0", drift_reconciliation_confidence=1.0),
        _drift_row("d1", drift_reconciliation_confidence=1.0),
        _drift_row("d2", drift_reconciliation_confidence=1.0),
    )
    horizons = (
        _horizon_row("h0", horizon_index=0),
        _horizon_row("h1", horizon_index=1),
        _horizon_row("h2", horizon_index=2),
    )
    return drifts, horizons


def _minor_inputs():
    drifts = (
        _drift_row(
            "d0",
            drift_reconciliation_confidence=0.80,
            threshold_drift_delta=0.20,
            continuity_drift_pressure=0.20,
            stability_recovery_alignment=0.85,
            cross_replay_drift_score=0.10,
        ),
        _drift_row(
            "d1",
            drift_reconciliation_confidence=0.70,
            threshold_drift_delta=0.20,
            continuity_drift_pressure=0.20,
            stability_recovery_alignment=0.85,
            cross_replay_drift_score=0.10,
        ),
    )
    horizons = (
        _horizon_row("h0", horizon_index=0, continuity_flag=True),
        _horizon_row("h1", horizon_index=1, continuity_flag=True),
    )
    return drifts, horizons


def _moderate_inputs():
    drifts = (
        _drift_row(
            "d0",
            drift_reconciliation_confidence=0.40,
            threshold_drift_delta=0.60,
            continuity_drift_pressure=0.60,
            stability_recovery_alignment=0.40,
            cross_replay_drift_score=0.60,
        ),
        _drift_row(
            "d1",
            drift_reconciliation_confidence=0.40,
            threshold_drift_delta=0.60,
            continuity_drift_pressure=0.60,
            stability_recovery_alignment=0.40,
            cross_replay_drift_score=0.60,
        ),
    )
    horizons = (
        _horizon_row("h0", horizon_index=0, continuity_flag=True),
        _horizon_row("h1", horizon_index=1, continuity_flag=False),
    )
    return drifts, horizons


def _severe_inputs():
    drifts = (
        _drift_row(
            "d0",
            drift_reconciliation_confidence=0.10,
            threshold_drift_delta=1.0,
            continuity_drift_pressure=1.0,
            stability_recovery_alignment=0.10,
            cross_replay_drift_score=1.0,
            replay_identity="r-old",
        ),
        _drift_row(
            "d1",
            drift_reconciliation_confidence=0.90,
            threshold_drift_delta=1.0,
            continuity_drift_pressure=1.0,
            stability_recovery_alignment=0.10,
            cross_replay_drift_score=1.0,
            replay_identity="r-new",
        ),
    )
    horizons = (
        _horizon_row("h0", horizon_index=0, continuity_flag=False),
        _horizon_row("h1", horizon_index=1, continuity_flag=False),
    )
    return drifts, horizons


def _metric_map(kernel):
    return {m.metric_name: m.metric_value for m in kernel.metrics}


def _run(scenario_id, drifts, horizons):
    return run_governance_drift_topology_stability(
        scenario=build_drift_topology_scenario(
            scenario_id=scenario_id,
            drift_reconciliation_series=drifts,
            replay_horizon_series=horizons,
        )
    )


def test_deterministic_repeated_topology_analysis() -> None:
    drifts, horizons = _stable_inputs()
    scenario = build_drift_topology_scenario(
        scenario_id="repeatable",
        drift_reconciliation_series=drifts,
        replay_horizon_series=horizons,
    )
    a = run_governance_drift_topology_stability(scenario=scenario)
    b = run_governance_drift_topology_stability(scenario=scenario)
    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.topology_hash == b.topology_hash
    assert a.receipt.receipt_hash == b.receipt.receipt_hash


def test_stable_hash_reproducibility() -> None:
    drifts, horizons = _stable_inputs()
    scenario_a = build_drift_topology_scenario(
        scenario_id="hash-repro",
        drift_reconciliation_series=drifts,
        replay_horizon_series=horizons,
    )
    scenario_b = build_drift_topology_scenario(
        scenario_id="hash-repro",
        drift_reconciliation_series=drifts,
        replay_horizon_series=horizons,
    )
    assert scenario_a.stable_hash() == scenario_b.stable_hash()


def test_stable_topology_band() -> None:
    drifts, horizons = _stable_inputs()
    kernel = _run("stable", drifts, horizons)
    assert kernel.advisory == ("stable_topology",)
    metrics = _metric_map(kernel)
    assert metrics["topology_coherence_score"] == 1.0
    assert metrics["horizon_stability_gradient"] == 1.0
    assert metrics["drift_surface_pressure"] == 0.0
    assert metrics["continuity_topology_alignment"] == 1.0
    assert metrics["cross_horizon_stability_score"] == 1.0
    assert metrics["topology_confidence_score"] == 1.0


def test_minor_topology_variation_band() -> None:
    drifts, horizons = _minor_inputs()
    kernel = _run("minor", drifts, horizons)
    assert kernel.advisory == ("minor_topology_variation",)
    for metric in kernel.metrics:
        assert 0.0 <= metric.metric_value <= 1.0


def test_moderate_topology_instability_band() -> None:
    drifts, horizons = _moderate_inputs()
    kernel = _run("moderate", drifts, horizons)
    assert kernel.advisory == ("moderate_topology_instability",)


def test_severe_topology_instability_band() -> None:
    drifts, horizons = _severe_inputs()
    kernel = _run("severe", drifts, horizons)
    assert kernel.advisory == ("severe_topology_instability",)
    metrics = _metric_map(kernel)
    assert metrics["drift_surface_pressure"] >= 0.5
    assert metrics["continuity_topology_alignment"] == 0.0


def test_canonical_json_round_trip() -> None:
    drifts, horizons = _moderate_inputs()
    kernel = _run("roundtrip", drifts, horizons)
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


def test_validator_never_raises_on_garbage() -> None:
    assert isinstance(validate_drift_topology_stability(object()), tuple)
    assert isinstance(validate_drift_topology_stability(None), tuple)
    assert isinstance(validate_drift_topology_stability(42), tuple)
    assert isinstance(validate_drift_topology_stability("not-a-kernel"), tuple)
    violations = validate_drift_topology_stability(object())
    assert "missing_scenario" in violations


def test_malformed_input_handling() -> None:
    scenario = build_drift_topology_scenario(
        scenario_id="",
        drift_reconciliation_series={"not": "a_list"},
        replay_horizon_series=None,
    )
    kernel = run_governance_drift_topology_stability(scenario=scenario)
    assert "empty_scenario_id" in kernel.violations
    assert "empty_drift_reconciliation_series" in kernel.violations
    assert "empty_replay_horizon_series" in kernel.violations

    bad_drifts = (
        {
            "reconciliation_id": "d0",
            "drift_reconciliation_confidence": "not-a-number",
            "threshold_drift_delta": None,
            "continuity_drift_pressure": float("nan"),
            "stability_recovery_alignment": "bad",
            "cross_replay_drift_score": True,
        },
    )
    bad_horizons = (
        {"horizon_id": "h0", "horizon_index": "zero", "continuity_flag": 1, "weight": "w"},
    )
    scenario2 = build_drift_topology_scenario(
        scenario_id="bad",
        drift_reconciliation_series=bad_drifts,
        replay_horizon_series=bad_horizons,
    )
    kernel2 = run_governance_drift_topology_stability(scenario=scenario2)
    for metric in kernel2.metrics:
        assert 0.0 <= metric.metric_value <= 1.0


def test_metric_bounds_and_fixed_order() -> None:
    drifts, horizons = _severe_inputs()
    kernel = _run("bounds", drifts, horizons)
    names = tuple(m.metric_name for m in kernel.metrics)
    assert names == (
        "topology_coherence_score",
        "horizon_stability_gradient",
        "drift_surface_pressure",
        "continuity_topology_alignment",
        "cross_horizon_stability_score",
        "topology_confidence_score",
    )
    for metric in kernel.metrics:
        assert 0.0 <= metric.metric_value <= 1.0
        assert metric.metric_order == names.index(metric.metric_name)


def test_replay_comparison_stability() -> None:
    drifts, horizons = _stable_inputs()
    scenario = build_drift_topology_scenario(
        scenario_id="replay",
        drift_reconciliation_series=drifts,
        replay_horizon_series=horizons,
    )
    baseline = run_governance_drift_topology_stability(scenario=scenario)
    replay = run_governance_drift_topology_stability(scenario=scenario)
    report = compare_drift_topology_replay(baseline, replay)
    assert report["is_stable_replay"] is True
    assert report["mismatches"] == ()
    for _, delta in report["metric_delta"]:
        assert delta == 0.0
    assert report["baseline_hash"] == report["replay_hash"]


def test_no_input_mutation() -> None:
    drifts, horizons = _minor_inputs()
    drift_snapshot = json.dumps(drifts, sort_keys=True)
    horizon_snapshot = json.dumps(horizons, sort_keys=True)
    scenario = build_drift_topology_scenario(
        scenario_id="no-mut",
        drift_reconciliation_series=drifts,
        replay_horizon_series=horizons,
    )
    run_governance_drift_topology_stability(scenario=scenario)
    assert json.dumps(drifts, sort_keys=True) == drift_snapshot
    assert json.dumps(horizons, sort_keys=True) == horizon_snapshot


def test_decoder_untouched_confirmation() -> None:
    module_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
        "qec",
        "orchestration",
        "governance_drift_topology_stability_kernel.py",
    )
    with open(module_path, "r", encoding="utf-8") as handle:
        source = handle.read()
    assert "qec.decoder" not in source
    assert "from qec.decoder" not in source
    assert "import qec.decoder" not in source
    decoder_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src",
        "qec",
        "decoder",
    )
    assert os.path.isdir(decoder_dir)


def test_receipt_builder_determinism_and_hash_stability() -> None:
    drifts, horizons = _minor_inputs()
    scenario = build_drift_topology_scenario(
        scenario_id="receipt",
        drift_reconciliation_series=drifts,
        replay_horizon_series=horizons,
    )
    kernel = run_governance_drift_topology_stability(scenario=scenario)
    r1 = build_drift_topology_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        advisory=kernel.advisory,
        topology_hash=kernel.topology_hash,
    )
    r2 = build_drift_topology_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        advisory=kernel.advisory,
        topology_hash=kernel.topology_hash,
    )
    assert r1.to_canonical_json() == r2.to_canonical_json()
    assert r1.receipt_hash == r2.receipt_hash
    assert r1.stable_hash() == r1.receipt_hash


def test_summary_contains_hashes_and_advisory() -> None:
    drifts, horizons = _stable_inputs()
    kernel = _run("summary", drifts, horizons)
    text = summarize_drift_topology_stability(kernel)
    assert kernel.topology_hash in text
    assert kernel.receipt.receipt_hash in text
    assert "stable_topology" in text
    assert "scenario_id=summary" in text
    assert "topology_coherence_score" in text


def test_dataclass_support_methods() -> None:
    scenario = DriftTopologyScenario(
        scenario_id="x",
        drift_reconciliation_series=(),
        replay_horizon_series=(),
    )
    metric = DriftTopologyMetric(
        metric_name="topology_coherence_score",
        metric_order=0,
        metric_value=1.0,
    )
    receipt = DriftTopologyReceipt(
        scenario_hash="a" * 64,
        metrics_hash="b" * 64,
        advisory_hash="c" * 64,
        topology_hash="d" * 64,
        topology_confidence_score=1.0,
        receipt_hash="e" * 64,
    )
    kernel = GovernanceDriftTopologyStabilityKernel(
        scenario=scenario,
        metrics=(metric,),
        advisory=("stable_topology",),
        violations=(),
        receipt=receipt,
        topology_hash="f" * 64,
    )
    assert isinstance(scenario.to_dict(), dict)
    assert isinstance(metric.to_dict(), dict)
    assert isinstance(receipt.to_dict(), dict)
    assert isinstance(kernel.to_dict(), dict)
    assert scenario.to_canonical_json() == scenario.to_canonical_json()
    assert metric.stable_hash() == metric.stable_hash()
    assert receipt.stable_hash() == receipt.receipt_hash
    assert kernel.stable_hash() == kernel.topology_hash
