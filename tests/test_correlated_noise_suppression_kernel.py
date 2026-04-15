import copy
import json
import math

from qec.orchestration.correlated_noise_suppression_kernel import (
    CorrelatedNoiseSuppressionKernel,
    CorrelatedNoiseNode,
    SuppressionMetric,
    SuppressionReceipt,
    build_correlated_noise_scenario,
    build_suppression_receipt,
    compare_suppression_replay,
    run_correlated_noise_suppression,
    summarize_correlated_noise_suppression,
    validate_correlated_noise_suppression,
)


def _base_series() -> dict:
    return {
        "noise_series": [
            {
                "noise_id": "n1",
                "topology_id": "t1",
                "temporal_index": 0,
                "noise_level": 0.10,
                "correlation_hint": 0.10,
                "residual_noise": 0.01,
                "replay_identity": "rA",
            },
            {
                "noise_id": "n2",
                "topology_id": "t1",
                "temporal_index": 1,
                "noise_level": 0.12,
                "correlation_hint": 0.12,
                "residual_noise": 0.01,
                "replay_identity": "rA",
            },
        ],
        "topology_series": [
            {"topology_id": "t1", "adjacency_pressure": 0.02, "coupling_strength": 0.02},
        ],
        "replay_horizon": [
            {"horizon_id": "h1", "horizon_step": 1.0, "expected_suppression": 0.10},
        ],
    }


def _build_scenario() -> CorrelatedNoiseNode:
    base = _base_series()
    return build_correlated_noise_scenario(
        scenario_id="scenario.alpha",
        noise_series=base["noise_series"],
        topology_series=base["topology_series"],
        replay_horizon=base["replay_horizon"],
    )


def test_deterministic_repeated_runs() -> None:
    scenario = _build_scenario()
    a = run_correlated_noise_suppression(scenario=scenario)
    b = run_correlated_noise_suppression(scenario=scenario)
    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.stable_hash() == b.stable_hash()


def test_stable_hash_reproducibility() -> None:
    scenario_a = _build_scenario()
    scenario_b = _build_scenario()
    assert scenario_a.stable_hash() == scenario_b.stable_hash()
    run_a = run_correlated_noise_suppression(scenario=scenario_a)
    run_b = run_correlated_noise_suppression(scenario=scenario_b)
    assert run_a.stable_hash() == run_b.stable_hash()


def test_canonical_json_round_trip() -> None:
    scenario = _build_scenario()
    as_json = scenario.to_canonical_json()
    rebuilt = build_correlated_noise_scenario(
        scenario_id=json.loads(as_json)["scenario_id"],
        noise_series=json.loads(as_json)["noise_series"],
        topology_series=json.loads(as_json)["topology_series"],
        replay_horizon=json.loads(as_json)["replay_horizon"],
    )
    assert scenario.to_canonical_json() == rebuilt.to_canonical_json()


def test_validator_never_raises() -> None:
    class Broken:
        @property
        def scenario(self):
            raise RuntimeError("boom")

    violations = validate_correlated_noise_suppression(Broken())
    assert isinstance(violations, tuple)
    assert "missing_scenario" in violations


def test_malformed_input_normalization() -> None:
    scenario = build_correlated_noise_scenario(
        scenario_id=123,
        noise_series=[{"noise_id": None, "noise_level": "bad", "correlation_hint": "NaN"}],
        topology_series=[{"topology_id": None, "adjacency_pressure": "Inf", "coupling_strength": -1}],
        replay_horizon=[{"horizon_id": None, "horizon_step": 0, "expected_suppression": "-inf"}],
    )
    assert scenario.scenario_id == "123"
    assert scenario.noise_series[0]["noise_level"] == 0.0
    assert scenario.topology_series[0]["adjacency_pressure"] == 0.0
    assert scenario.replay_horizon[0]["horizon_step"] == 1.0


def test_metric_bounds_and_fixed_order() -> None:
    kernel = run_correlated_noise_suppression(scenario=_build_scenario())
    names = tuple(metric.metric_name for metric in kernel.metrics)
    assert names == (
        "spatial_correlation_score",
        "temporal_correlation_score",
        "topology_noise_pressure",
        "suppression_alignment_score",
        "residual_noise_score",
        "suppression_confidence_score",
    )
    for metric in kernel.metrics:
        assert 0.0 <= metric.metric_value <= 1.0


def test_all_advisory_bands() -> None:
    cases = [
        ((0.0, 1.0, 0.0, 0.0), "no_suppression_required"),
        ((0.0, 0.8, 0.0, 0.0), "mild_correlated_suppression"),
        ((0.0, 0.4, 0.0, 0.0), "moderate_correlated_suppression"),
        ((0.0, 0.1, 0.9, 0.9), "severe_correlated_suppression"),
    ]
    for (left_level, right_level, topology_pressure, residual), expected in cases:
        scenario = build_correlated_noise_scenario(
            scenario_id=f"band.{left_level}.{right_level}.{topology_pressure}.{residual}",
            noise_series=[
                {
                    "noise_id": "n1",
                    "topology_id": "t1",
                    "temporal_index": 0,
                    "noise_level": left_level,
                    "correlation_hint": left_level,
                    "residual_noise": residual,
                },
                {
                    "noise_id": "n2",
                    "topology_id": "t1",
                    "temporal_index": 1,
                    "noise_level": right_level,
                    "correlation_hint": right_level,
                    "residual_noise": residual,
                },
            ],
            topology_series=[
                {
                    "topology_id": "t1",
                    "adjacency_pressure": topology_pressure,
                    "coupling_strength": topology_pressure,
                }
            ],
            replay_horizon=[{"horizon_id": "h1", "horizon_step": 1, "expected_suppression": right_level}],
        )
        assert run_correlated_noise_suppression(scenario=scenario).advisory_output == expected


def test_replay_comparison_stability() -> None:
    scenario = _build_scenario()
    a = run_correlated_noise_suppression(scenario=scenario)
    b = run_correlated_noise_suppression(scenario=scenario)
    report_a = compare_suppression_replay(a, b)
    report_b = compare_suppression_replay(a, b)
    assert report_a == report_b
    assert report_a["hash_match"] is True


def test_no_input_mutation() -> None:
    base = _base_series()
    frozen = copy.deepcopy(base)
    _ = build_correlated_noise_scenario(
        scenario_id="immutability",
        noise_series=base["noise_series"],
        topology_series=base["topology_series"],
        replay_horizon=base["replay_horizon"],
    )
    assert base == frozen


def test_decoder_untouched_confirmation() -> None:
    kernel = run_correlated_noise_suppression(scenario=_build_scenario())
    assert kernel.sideband_only is True
    assert kernel.decoder_untouched is True
    assert kernel.suppression_analysis["decoder_semantics_modified"] is False




def test_validator_detects_tampered_receipt_payload() -> None:
    kernel = run_correlated_noise_suppression(scenario=_build_scenario())
    tampered_receipt = SuppressionReceipt(
        scenario_hash=kernel.suppression_receipt.scenario_hash,
        metrics_hash=kernel.suppression_receipt.metrics_hash,
        suppression_hash=kernel.suppression_receipt.suppression_hash,
        suppression_confidence_score=kernel.suppression_receipt.suppression_confidence_score,
        advisory_output="severe_correlated_suppression",
        sideband_only=kernel.suppression_receipt.sideband_only,
        receipt_hash=kernel.suppression_receipt.receipt_hash,
    )
    tampered_kernel = CorrelatedNoiseSuppressionKernel(
        scenario=kernel.scenario,
        metrics=kernel.metrics,
        suppression_analysis=kernel.suppression_analysis,
        advisory_output=kernel.advisory_output,
        violations=kernel.violations,
        suppression_receipt=tampered_receipt,
        suppression_hash=kernel.suppression_hash,
        sideband_only=kernel.sideband_only,
        decoder_untouched=kernel.decoder_untouched,
    )

    violations = validate_correlated_noise_suppression(tampered_kernel)
    assert "receipt_hash_mismatch" in violations

def test_suppression_receipt_determinism() -> None:
    scenario = _build_scenario()
    kernel = run_correlated_noise_suppression(scenario=scenario)
    a = build_suppression_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        advisory_output=kernel.advisory_output,
        suppression_hash=kernel.suppression_hash,
    )
    b = build_suppression_receipt(
        scenario=scenario,
        metrics=kernel.metrics,
        advisory_output=kernel.advisory_output,
        suppression_hash=kernel.suppression_hash,
    )
    assert a.to_canonical_json() == b.to_canonical_json()
    assert a.stable_hash() == a.receipt_hash


def test_summary_content() -> None:
    kernel = run_correlated_noise_suppression(scenario=_build_scenario())
    summary = summarize_correlated_noise_suppression(kernel)
    assert "scenario_id=scenario.alpha" in summary
    assert f"advisory_output={kernel.advisory_output}" in summary
    assert "decoder_untouched=True" in summary


def test_malformed_float_handling_nan_inf_sentinels() -> None:
    scenario = build_correlated_noise_scenario(
        scenario_id="float.sentinel",
        noise_series=[
            {
                "noise_id": "n1",
                "topology_id": "t1",
                "temporal_index": 0,
                "noise_level": math.nan,
                "correlation_hint": math.inf,
                "residual_noise": -math.inf,
            }
        ],
        topology_series=[{"topology_id": "t1", "adjacency_pressure": math.nan, "coupling_strength": math.inf}],
        replay_horizon=[{"horizon_id": "h", "horizon_step": math.nan, "expected_suppression": math.inf}],
    )
    kernel = run_correlated_noise_suppression(scenario=scenario)
    assert all(0.0 <= m.metric_value <= 1.0 for m in kernel.metrics)


def test_topology_ordering_determinism() -> None:
    first = build_correlated_noise_scenario(
        scenario_id="topo.order",
        noise_series=_base_series()["noise_series"],
        topology_series=[
            {"topology_id": "t2", "adjacency_pressure": 0.2, "coupling_strength": 0.2},
            {"topology_id": "t1", "adjacency_pressure": 0.1, "coupling_strength": 0.1},
        ],
        replay_horizon=_base_series()["replay_horizon"],
    )
    second = build_correlated_noise_scenario(
        scenario_id="topo.order",
        noise_series=_base_series()["noise_series"],
        topology_series=[
            {"topology_id": "t1", "adjacency_pressure": 0.1, "coupling_strength": 0.1},
            {"topology_id": "t2", "adjacency_pressure": 0.2, "coupling_strength": 0.2},
        ],
        replay_horizon=_base_series()["replay_horizon"],
    )
    assert first.topology_series == second.topology_series
    assert first.stable_hash() == second.stable_hash()


def test_required_dataclass_methods_exist_and_return_types() -> None:
    scenario = _build_scenario()
    kernel = run_correlated_noise_suppression(scenario=scenario)

    metric = SuppressionMetric("spatial_correlation_score", 0, 0.5)
    receipt = SuppressionReceipt(
        scenario_hash="a" * 64,
        metrics_hash="b" * 64,
        suppression_hash="c" * 64,
        suppression_confidence_score=0.8,
        advisory_output="mild_correlated_suppression",
        sideband_only=True,
        receipt_hash="d" * 64,
    )
    bare_kernel = CorrelatedNoiseSuppressionKernel(
        scenario=scenario,
        metrics=(metric,),
        suppression_analysis={"x": 1},
        advisory_output="mild_correlated_suppression",
        violations=(),
        suppression_receipt=receipt,
        suppression_hash="e" * 64,
        sideband_only=True,
        decoder_untouched=True,
    )
    for obj in (scenario, metric, receipt, bare_kernel, kernel):
        assert isinstance(obj.to_dict(), dict)
        assert isinstance(obj.to_canonical_json(), str)
        assert isinstance(obj.stable_hash(), str)
