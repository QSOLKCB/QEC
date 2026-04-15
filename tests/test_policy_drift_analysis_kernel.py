"""Tests for v137.19.1 — Policy Drift Analysis Kernel."""

from __future__ import annotations

import json

from qec.orchestration.policy_drift_analysis_kernel import (
    _METRIC_ORDER,
    PolicyDriftAnalysisKernel,
    PolicyDriftMetric,
    PolicyDriftReceipt,
    PolicyDriftScenario,
    build_policy_drift_receipt,
    build_policy_drift_scenario,
    compare_policy_drift_replay,
    run_policy_drift_analysis,
    summarize_policy_drift,
    validate_policy_drift_analysis,
)


# Source metric order from the kernel itself to avoid drift between code and
# tests if new metrics are added in canonical order.
_EXPECTED_METRIC_ORDER = _METRIC_ORDER


def _benchmark(
    *,
    benchmark_id: str = "bench.alpha",
    allow_count: int = 80,
    deny_count: int = 20,
    decision_surface=(("k1", "allow"), ("k2", "deny"), ("k3", "allow")),
    boundary_failures: int = 0,
    continuity_ok: bool = True,
    replay_identity: str = "ri.alpha",
    trace_length: int = 100,
):
    return {
        "benchmark_id": benchmark_id,
        "allow_count": allow_count,
        "deny_count": deny_count,
        "decision_surface": decision_surface,
        "boundary_failures": boundary_failures,
        "continuity_ok": continuity_ok,
        "replay_identity": replay_identity,
        "trace_length": trace_length,
    }


def _scenario(**overrides_b):
    benchmark_a = _benchmark(benchmark_id="bench.a")
    benchmark_b = _benchmark(benchmark_id="bench.b", **overrides_b)
    return build_policy_drift_scenario(
        scenario_id="scn.basic",
        benchmark_a=benchmark_a,
        benchmark_b=benchmark_b,
    )


def test_build_scenario_returns_frozen_dataclass():
    scenario = _scenario()
    assert isinstance(scenario, PolicyDriftScenario)
    assert scenario.scenario_id == "scn.basic"
    assert scenario.benchmark_a.benchmark_id == "bench.a"
    assert scenario.benchmark_b.benchmark_id == "bench.b"
    # canonical json round trip
    blob = scenario.to_canonical_json()
    assert json.loads(blob)["scenario_id"] == "scn.basic"
    assert scenario.stable_hash() == scenario.stable_hash()


def test_deterministic_repeated_analysis():
    scenario = _scenario()
    first = run_policy_drift_analysis(scenario)
    second = run_policy_drift_analysis(scenario)
    assert first.analysis_hash == second.analysis_hash
    assert first.receipt.receipt_hash == second.receipt.receipt_hash
    assert first.to_canonical_json() == second.to_canonical_json()


def test_stable_hash_reproducibility_across_builds():
    scenario_a = _scenario()
    scenario_b = build_policy_drift_scenario(
        scenario_id="scn.basic",
        benchmark_a=_benchmark(benchmark_id="bench.a"),
        benchmark_b=_benchmark(benchmark_id="bench.b"),
    )
    assert scenario_a.stable_hash() == scenario_b.stable_hash()
    analysis_a = run_policy_drift_analysis(scenario_a)
    analysis_b = run_policy_drift_analysis(scenario_b)
    assert analysis_a.analysis_hash == analysis_b.analysis_hash


def test_zero_drift_when_benchmarks_identical():
    scenario = build_policy_drift_scenario(
        scenario_id="scn.zero",
        benchmark_a=_benchmark(benchmark_id="same", replay_identity="ri.same"),
        benchmark_b=_benchmark(benchmark_id="same", replay_identity="ri.same"),
    )
    analysis = run_policy_drift_analysis(scenario)
    deltas = {metric.metric_name: metric.delta for metric in analysis.metrics}
    for name in _EXPECTED_METRIC_ORDER:
        assert deltas[name] == 0.0, f"{name} should be zero for identical benchmarks"


def test_nonzero_drift_on_differing_benchmarks():
    scenario = _scenario(
        allow_count=20,
        deny_count=80,
        decision_surface=(("k1", "deny"), ("k4", "allow")),
        boundary_failures=5,
        continuity_ok=False,
        replay_identity="ri.beta",
        trace_length=40,
    )
    analysis = run_policy_drift_analysis(scenario)
    deltas = {metric.metric_name: metric.delta for metric in analysis.metrics}
    assert deltas["allow_drift_rate"] > 0.0
    assert deltas["deny_drift_rate"] > 0.0
    assert deltas["decision_surface_delta"] > 0.0
    assert deltas["boundary_failure_delta"] == 5.0
    assert deltas["continuity_delta"] == 1.0
    assert deltas["replay_stability_delta"] == 1.0
    assert deltas["trace_length_delta"] == 60.0
    assert deltas["drift_severity_score"] > 0.0


def test_allow_deny_drift_rates_symmetric():
    scenario = _scenario(allow_count=20, deny_count=80)
    analysis = run_policy_drift_analysis(scenario)
    deltas = {metric.metric_name: metric.delta for metric in analysis.metrics}
    # allow/deny drift rates must agree when totals are equal
    assert abs(deltas["allow_drift_rate"] - deltas["deny_drift_rate"]) < 1e-12
    assert abs(deltas["allow_drift_rate"] - 0.6) < 1e-12


def test_decision_surface_jaccard_delta():
    scenario = build_policy_drift_scenario(
        scenario_id="scn.surface",
        benchmark_a=_benchmark(
            benchmark_id="a",
            decision_surface=(("k1", "allow"), ("k2", "allow")),
        ),
        benchmark_b=_benchmark(
            benchmark_id="b",
            decision_surface=(("k2", "allow"), ("k3", "deny")),
        ),
    )
    analysis = run_policy_drift_analysis(scenario)
    metric = next(m for m in analysis.metrics if m.metric_name == "decision_surface_delta")
    # |symmetric_diff| = 2, |union| = 3 -> 2/3
    assert abs(metric.delta - (2.0 / 3.0)) < 1e-12
    assert metric.value_a == 2.0
    assert metric.value_b == 2.0


def test_malformed_scenario_input_never_raises():
    # run_policy_drift_analysis must accept anything
    analysis = run_policy_drift_analysis(None)
    assert isinstance(analysis, PolicyDriftAnalysisKernel)
    assert "malformed_scenario_input" in analysis.violations
    analysis2 = run_policy_drift_analysis(42)
    assert "malformed_scenario_input" in analysis2.violations
    analysis3 = run_policy_drift_analysis("not-a-scenario")
    assert "malformed_scenario_input" in analysis3.violations


def test_build_scenario_coerces_junk_without_raising():
    scenario = build_policy_drift_scenario(
        scenario_id="scn.junk",
        benchmark_a={"allow_count": "bad", "deny_count": -5, "decision_surface": 7},
        benchmark_b=None,
    )
    assert scenario.benchmark_a.allow_count == 0
    assert scenario.benchmark_a.deny_count == 0
    assert scenario.benchmark_a.decision_surface == ()
    assert scenario.benchmark_b.benchmark_id == "benchmark_b"
    assert scenario.benchmark_b.continuity_ok is False


def test_validator_never_raises_on_garbage():
    assert validate_policy_drift_analysis(None) == ("malformed_policy_drift_analysis",)
    assert validate_policy_drift_analysis("x") == ("malformed_policy_drift_analysis",)
    assert validate_policy_drift_analysis({"kind": "not-analysis"}) == (
        "malformed_policy_drift_analysis",
    )
    assert validate_policy_drift_analysis(123) == ("malformed_policy_drift_analysis",)


def test_validator_clean_on_valid_analysis():
    analysis = run_policy_drift_analysis(_scenario(allow_count=10, deny_count=90))
    assert validate_policy_drift_analysis(analysis) == ()


def test_replay_comparison_stability_identical():
    analysis_a = run_policy_drift_analysis(_scenario())
    analysis_b = run_policy_drift_analysis(_scenario())
    report = compare_policy_drift_replay(analysis_a, analysis_b)
    assert report["match"] is True
    assert report["mismatch_fields"] == ()
    assert report["analysis_a_hash"] == report["analysis_b_hash"]


def test_replay_comparison_detects_divergence():
    analysis_a = run_policy_drift_analysis(_scenario())
    analysis_b = run_policy_drift_analysis(_scenario(allow_count=1, deny_count=99))
    report = compare_policy_drift_replay(analysis_a, analysis_b)
    assert report["match"] is False
    assert "analysis_hash" in report["mismatch_fields"]
    assert "metrics" in report["mismatch_fields"]


def test_replay_comparison_handles_malformed_inputs():
    report = compare_policy_drift_replay("x", None)
    assert report["match"] is False
    assert "type" in report["mismatch_fields"]


def test_canonical_json_round_trip_on_all_types():
    analysis = run_policy_drift_analysis(_scenario())
    for obj in (analysis.scenario, analysis.metrics[0], analysis.receipt, analysis):
        blob = obj.to_canonical_json()
        data = json.loads(blob)
        assert isinstance(data, dict)
        assert obj.stable_hash() == obj.stable_hash()


def test_deterministic_metric_ordering():
    analysis = run_policy_drift_analysis(_scenario(allow_count=50, deny_count=50))
    names = tuple(metric.metric_name for metric in analysis.metrics)
    assert names == _EXPECTED_METRIC_ORDER
    # metric_order field must match positional index
    for idx, metric in enumerate(analysis.metrics):
        assert metric.metric_order == idx


def test_summarize_policy_drift_valid():
    analysis = run_policy_drift_analysis(_scenario(allow_count=10, deny_count=90))
    summary = summarize_policy_drift(analysis)
    assert summary["valid"] is True
    assert summary["scenario_id"] == "scn.basic"
    assert summary["analysis_hash"] == analysis.analysis_hash
    assert summary["receipt_hash"] == analysis.receipt.receipt_hash
    assert set(summary["metric_deltas"].keys()) == set(_EXPECTED_METRIC_ORDER)


def test_summarize_policy_drift_malformed():
    summary = summarize_policy_drift("nope")
    assert summary["valid"] is False
    assert summary["violations"] == ("malformed_policy_drift_analysis",)
    assert summary["analysis_hash"] == ""


def test_build_receipt_is_deterministic_standalone():
    receipt_a = build_policy_drift_receipt(
        scenario_hash="sh",
        metrics_hash="mh",
        drift_severity_score=0.25,
        analysis_hash="ah",
    )
    receipt_b = build_policy_drift_receipt(
        scenario_hash="sh",
        metrics_hash="mh",
        drift_severity_score=0.25,
        analysis_hash="ah",
    )
    assert isinstance(receipt_a, PolicyDriftReceipt)
    assert receipt_a.receipt_hash == receipt_b.receipt_hash
    assert receipt_a.stable_hash() == receipt_a.receipt_hash


def test_scenario_from_object_equivalent_to_from_mapping():
    class _B:
        benchmark_id = "bench.a"
        allow_count = 80
        deny_count = 20
        decision_surface = (("k1", "allow"), ("k2", "deny"), ("k3", "allow"))
        boundary_failures = 0
        continuity_ok = True
        replay_identity = "ri.alpha"
        trace_length = 100

    scenario_obj = build_policy_drift_scenario(
        scenario_id="scn.basic",
        benchmark_a=_B(),
        benchmark_b=_B(),
    )
    scenario_map = build_policy_drift_scenario(
        scenario_id="scn.basic",
        benchmark_a=_benchmark(benchmark_id="bench.a"),
        benchmark_b=_benchmark(benchmark_id="bench.a"),
    )
    assert scenario_obj.stable_hash() == scenario_map.stable_hash()


def test_input_benchmarks_are_not_mutated():
    original_a = _benchmark(benchmark_id="a")
    original_b = _benchmark(benchmark_id="b")
    snapshot_a = json.dumps(original_a, sort_keys=True)
    snapshot_b = json.dumps(original_b, sort_keys=True)
    scenario = build_policy_drift_scenario(
        scenario_id="scn.nomut",
        benchmark_a=original_a,
        benchmark_b=original_b,
    )
    _ = run_policy_drift_analysis(scenario)
    assert json.dumps(original_a, sort_keys=True) == snapshot_a
    assert json.dumps(original_b, sort_keys=True) == snapshot_b


def test_validator_detects_tampered_analysis_hash():
    analysis = run_policy_drift_analysis(_scenario())
    tampered = PolicyDriftAnalysisKernel(
        scenario=analysis.scenario,
        metrics=analysis.metrics,
        violations=analysis.violations,
        receipt=analysis.receipt,
        analysis_hash="0" * 64,
    )
    violations = validate_policy_drift_analysis(tampered)
    assert "analysis_hash_body_mismatch" in violations or "analysis_hash_linkage_mismatch" in violations


def test_severity_score_bounded_and_monotonic():
    zero = run_policy_drift_analysis(
        build_policy_drift_scenario(
            scenario_id="s0",
            benchmark_a=_benchmark(benchmark_id="x"),
            benchmark_b=_benchmark(benchmark_id="x"),
        )
    )
    partial = run_policy_drift_analysis(
        _scenario(allow_count=60, deny_count=40)
    )
    heavy = run_policy_drift_analysis(
        _scenario(
            allow_count=0,
            deny_count=100,
            decision_surface=(("q1", "deny"),),
            boundary_failures=50,
            continuity_ok=False,
            replay_identity="ri.zzz",
            trace_length=9999,
        )
    )
    zero_score = zero.receipt.drift_severity_score
    partial_score = partial.receipt.drift_severity_score
    heavy_score = heavy.receipt.drift_severity_score
    assert zero_score == 0.0
    assert 0.0 <= partial_score <= 1.0
    assert 0.0 <= heavy_score <= 1.0
    assert partial_score < heavy_score


def test_bool_inputs_rejected_for_numeric_counts():
    # Booleans must not be silently coerced into 1/0 counts. Also continuity_ok
    # must be strictly the literal True — truthy junk must not flip it.
    scenario = build_policy_drift_scenario(
        scenario_id="scn.bool",
        benchmark_a={
            "benchmark_id": "a",
            "allow_count": True,
            "deny_count": False,
            "boundary_failures": True,
            "trace_length": True,
            "continuity_ok": "yes",  # truthy but not the literal True
        },
        benchmark_b={
            "benchmark_id": "b",
            "allow_count": 10,
            "deny_count": 0,
            "continuity_ok": True,
        },
    )
    assert scenario.benchmark_a.allow_count == 0
    assert scenario.benchmark_a.deny_count == 0
    assert scenario.benchmark_a.boundary_failures == 0
    assert scenario.benchmark_a.trace_length == 0
    assert scenario.benchmark_a.continuity_ok is False
    assert scenario.benchmark_b.continuity_ok is True


def test_summarize_reports_invalid_for_tampered_hash():
    analysis = run_policy_drift_analysis(_scenario())
    tampered = PolicyDriftAnalysisKernel(
        scenario=analysis.scenario,
        metrics=analysis.metrics,
        violations=analysis.violations,
        receipt=analysis.receipt,
        analysis_hash="0" * 64,
    )
    summary = summarize_policy_drift(tampered)
    assert summary["valid"] is False
    # The integrity-check violation must surface in the summary
    assert any("hash" in v for v in summary["violations"])


def test_metric_stable_hash_reproducible():
    analysis_a = run_policy_drift_analysis(_scenario())
    analysis_b = run_policy_drift_analysis(_scenario())
    hashes_a = tuple(metric.stable_hash() for metric in analysis_a.metrics)
    hashes_b = tuple(metric.stable_hash() for metric in analysis_b.metrics)
    assert hashes_a == hashes_b
    # Each metric has an isinstance check
    for metric in analysis_a.metrics:
        assert isinstance(metric, PolicyDriftMetric)
