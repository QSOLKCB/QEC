from __future__ import annotations

import hashlib
from pathlib import Path

from qec.analysis.optimized_telemetry_receipts import (
    TelemetrySourceDeclaration,
    TelemetryMetricDeclaration,
    TelemetrySample,
    build_telemetry_source_declaration,
    build_telemetry_metric_declaration,
    build_telemetry_sample,
    build_telemetry_aggregation,
    build_telemetry_threshold_evaluation,
    build_telemetry_claim,
    build_optimized_telemetry_receipt,
    validate_optimized_telemetry_receipt,
)


def _h(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _src() -> TelemetrySourceDeclaration:
    return build_telemetry_source_declaration(
        source_index=0,
        source_name="s",
        source_status="TELEMETRY_SOURCE_DECLARED",
        source_kind="OFFLINE_TELEMETRY_SOURCE",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="x",
        source_optimized_simulation_spec_hash=_h("spec"),
        source_backend_equivalence_replay_receipt_hash=_h("replay"),
        source_optimized_qec_benchmark_receipt_hash=_h("bench"),
        telemetry_source_identity_hash=_h("i"),
        telemetry_origin_hash=_h("o"),
        telemetry_collection_policy_hash=_h("p"),
        live_collection_allowed=False,
        emission_allowed=False,
        reason="ok",
    )


def _metric(src_hash: str) -> TelemetryMetricDeclaration:
    return build_telemetry_metric_declaration(
        metric_index=0,
        metric_name="m",
        metric_status="TELEMETRY_METRIC_DECLARED",
        metric_kind="LATENCY_NS",
        metric_unit="ns",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="x",
        source_telemetry_source_hash=src_hash,
        source_optimized_simulation_spec_hash=_h("spec"),
        source_backend_equivalence_replay_receipt_hash=_h("replay"),
        source_optimized_qec_benchmark_receipt_hash=_h("bench"),
        metric_identity_hash=_h("mi"),
        metric_precision_policy_hash=None,
        bounded_sample_count=1,
        aggregation_required=True,
        threshold_required=True,
        reason="ok",
    )


def _sample(src_hash: str, metric_hash: str) -> TelemetrySample:
    return build_telemetry_sample(
        sample_index=0,
        sample_name="a",
        sample_role="AGGREGATE_SAMPLE",
        metric_kind="LATENCY_NS",
        metric_unit="ns",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="x",
        source_telemetry_source_hash=src_hash,
        source_metric_declaration_hash=metric_hash,
        source_optimized_simulation_spec_hash=_h("spec"),
        source_backend_equivalence_replay_receipt_hash=_h("replay"),
        source_optimized_qec_benchmark_receipt_hash=_h("bench"),
        logical_sample_index=0,
        logical_tick=0,
        value_numerator=1,
        value_denominator=1,
        sample_series_hash=_h("ss"),
        sample_payload_hash=_h("sp"),
        declared_unavailable_reason=None,
        declared_error_code=None,
        reason="ok",
    )


def test_optimized_telemetry_child_hash_stability() -> None:
    s = _src()
    m = _metric(s.telemetry_source_declaration_hash)
    sm = _sample(
        s.telemetry_source_declaration_hash, m.telemetry_metric_declaration_hash
    )
    assert len(s.telemetry_source_declaration_hash) == 64
    assert len(m.telemetry_metric_declaration_hash) == 64
    assert isinstance(sm.to_dict(), dict)


def test_optimized_telemetry_receipt_hash_stability() -> None:
    s = _src()
    m = _metric(s.telemetry_source_declaration_hash)
    sm = _sample(
        s.telemetry_source_declaration_hash, m.telemetry_metric_declaration_hash
    )
    ag = build_telemetry_aggregation(
        aggregation_index=0,
        aggregation_name="a",
        aggregation_kind="EXACT_VALUE",
        metric_kind="LATENCY_NS",
        metric_unit="ns",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="x",
        source_metric_declaration_hash=m.telemetry_metric_declaration_hash,
        source_sample_hashes=(sm.telemetry_sample_hash,),
        source_optimized_qec_benchmark_receipt_hash=_h("bench"),
        aggregate_value_numerator=1,
        aggregate_value_denominator=1,
        sample_count=1,
        min_value_numerator=1,
        min_value_denominator=1,
        max_value_numerator=1,
        max_value_denominator=1,
        median_value_numerator=1,
        median_value_denominator=1,
        aggregation_series_hash=_h("as"),
        reason="ok",
    )
    th = build_telemetry_threshold_evaluation(
        threshold_index=0,
        threshold_name="t",
        threshold_direction="EXACTLY_EQUAL",
        threshold_status="TELEMETRY_THRESHOLD_PASSED",
        metric_kind="LATENCY_NS",
        metric_unit="ns",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="x",
        source_metric_declaration_hash=m.telemetry_metric_declaration_hash,
        source_aggregation_hash=ag.telemetry_aggregation_hash,
        threshold_lower_numerator=1,
        threshold_lower_denominator=1,
        threshold_upper_numerator=1,
        threshold_upper_denominator=1,
        observed_value_numerator=1,
        observed_value_denominator=1,
        threshold_passed=True,
        failure_code=None,
        reason="ok",
    )
    cl = build_telemetry_claim(
        claim_index=0,
        claim_name="c",
        claim_kind="TELEMETRY_NO_REGRESSION",
        claim_status="TELEMETRY_CLAIM_ACCEPTED",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="x",
        source_threshold_evaluation_hash=th.telemetry_threshold_evaluation_hash,
        source_aggregation_hash=ag.telemetry_aggregation_hash,
        source_optimized_qec_benchmark_receipt_hash=_h("bench"),
        source_backend_equivalence_replay_receipt_hash=_h("replay"),
        claim_scope_hash=_h("cs"),
        telemetry_is_proof=False,
        live_collection_used=False,
        emission_used=False,
        benchmark_receipt_required=True,
        benchmark_receipt_present=True,
        replay_equivalence_required=True,
        replay_equivalence_passed=True,
        failure_code=None,
        reason="ok",
    )
    r = build_optimized_telemetry_receipt(
        schema_version="OPTIMIZED_TELEMETRY_RECEIPT_V1",
        telemetry_mode="BENCHMARK_BOUND_TELEMETRY",
        telemetry_status="OPTIMIZED_TELEMETRY_DRAFT",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="x",
        source_heavy_dependency_discovery_manifest_hash=_h("1"),
        source_dependency_hotpath_receipt_hash=_h("2"),
        source_backend_invariant_candidate_receipt_hash=_h("3"),
        source_cross_backend_equivalence_receipt_hash=_h("4"),
        source_optimization_opportunity_index_hash=_h("5"),
        source_optimization_contract_hash=_h("6"),
        source_lightweight_adapter_spec_hash=_h("7"),
        source_cached_canonical_kernel_receipt_hash=_h("8"),
        source_fast_path_equivalence_receipt_hash=_h("9"),
        source_optimization_implementation_receipt_hash=_h("10"),
        source_dependency_reduction_receipt_hash=_h("11"),
        source_optimized_simulation_spec_hash=_h("spec"),
        source_backend_equivalence_replay_receipt_hash=_h("replay"),
        source_optimized_qec_benchmark_receipt_hash=_h("bench"),
        source_count=1,
        metric_count=1,
        sample_count=1,
        aggregation_count=1,
        threshold_evaluation_count=1,
        claim_count=1,
        sources=(s,),
        metrics=(m,),
        samples=(sm,),
        aggregations=(ag,),
        threshold_evaluations=(th,),
        claims=(cl,),
        first_source_hash=s.telemetry_source_declaration_hash,
        final_source_hash=s.telemetry_source_declaration_hash,
        first_metric_hash=m.telemetry_metric_declaration_hash,
        final_metric_hash=m.telemetry_metric_declaration_hash,
        first_sample_hash=sm.telemetry_sample_hash,
        final_sample_hash=sm.telemetry_sample_hash,
        first_aggregation_hash=ag.telemetry_aggregation_hash,
        final_aggregation_hash=ag.telemetry_aggregation_hash,
        first_threshold_evaluation_hash=th.telemetry_threshold_evaluation_hash,
        final_threshold_evaluation_hash=th.telemetry_threshold_evaluation_hash,
        first_claim_hash=cl.telemetry_claim_hash,
        final_claim_hash=cl.telemetry_claim_hash,
        all_sources_declared=True,
        all_metrics_declared=True,
        all_samples_declared=True,
        all_aggregations_valid=True,
        all_thresholds_passed=True,
        all_claims_accepted=True,
        optimized_qec_benchmark_receipt_present=True,
        backend_equivalence_replay_passed=True,
        optimized_simulation_spec_ready=True,
        live_collection_used=False,
        emission_used=False,
        telemetry_is_not_proof=True,
    )
    assert len(r.optimized_telemetry_receipt_hash) == 64
    validate_optimized_telemetry_receipt(r)


def test_optimized_telemetry_source_scan_and_decoder_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "src/qec/analysis/optimized_telemetry_receipts.py").read_text(
        encoding="utf-8"
    )
    for bad in [
        "numpy",
        "requests",
        "subprocess",
        "time.time",
        "perf_counter",
        "datetime.now",
    ]:
        assert bad not in text
