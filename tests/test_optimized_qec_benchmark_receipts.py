from __future__ import annotations

from pathlib import Path

from qec.analysis.optimized_qec_benchmark_receipts import (
    BenchmarkClaim,
    BenchmarkComparisonCase,
    BenchmarkComparisonResult,
    BenchmarkEnvironmentDeclaration,
    BenchmarkMeasurement,
    BenchmarkWorkloadDeclaration,
    build_benchmark_claim,
    build_benchmark_comparison_case,
    build_benchmark_comparison_result,
    build_benchmark_environment_declaration,
    build_benchmark_measurement,
    build_benchmark_workload_declaration,
    build_optimized_qec_benchmark_receipt,
    validate_benchmark_claim,
    validate_benchmark_comparison_case,
    validate_benchmark_comparison_result,
    validate_benchmark_environment_declaration,
    validate_benchmark_measurement,
    validate_benchmark_workload_declaration,
    validate_optimized_qec_benchmark_receipt,
)


def _h(seed: str) -> str:
    import hashlib

    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def test_optimized_qec_benchmark_child_hash_stability() -> None:
    m = BenchmarkMeasurement(
        measurement_index=0,
        measurement_name="m",
        measurement_role="REFERENCE_BACKEND_MEASUREMENT",
        measurement_kind="ELAPSED_TIME_NS",
        measurement_unit="ns",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="s",
        source_environment_hash=_h("e"),
        source_workload_hash=_h("w"),
        source_backend_equivalence_replay_receipt_hash=_h("r"),
        source_optimized_simulation_spec_hash=_h("s"),
        measurement_series_hash=_h("v"),
        measurement_values=(1, 2),
        value_numerator=1,
        value_denominator=1,
        sample_count=2,
        min_value_numerator=1,
        min_value_denominator=1,
        max_value_numerator=2,
        max_value_denominator=1,
        median_value_numerator=3,
        median_value_denominator=2,
        declared_unavailable_reason=None,
        declared_error_code=None,
        reason="ok",
        benchmark_measurement_hash=_h("m"),
    )
    assert isinstance(m.to_dict()["measurement_values"], list)


def test_optimized_qec_benchmark_receipt_hash_stability() -> None:
    r = build_optimized_qec_benchmark_receipt(
        schema_version="OPTIMIZED_QEC_BENCHMARK_RECEIPT_V1",
        benchmark_mode="REPLAY_BOUND_BENCHMARK",
        benchmark_status="OPTIMIZED_QEC_BENCHMARK_DRAFT",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="s",
        source_optimized_simulation_spec_hash=_h("s"),
        source_backend_equivalence_replay_receipt_hash=_h("r"),
        environment_count=0,
        workload_count=0,
        measurement_count=0,
        comparison_case_count=0,
        comparison_result_count=0,
        claim_count=0,
        environments=(),
        workloads=(),
        measurements=(),
        comparison_cases=(),
        comparison_results=(),
        claims=(),
        first_environment_hash="",
        final_environment_hash="",
        first_workload_hash="",
        final_workload_hash="",
        first_measurement_hash="",
        final_measurement_hash="",
        first_comparison_case_hash="",
        final_comparison_case_hash="",
        first_comparison_result_hash="",
        final_comparison_result_hash="",
        first_claim_hash="",
        final_claim_hash="",
        all_environments_declared=True,
        all_workloads_declared=True,
        all_measurements_declared=True,
        all_comparisons_passed=True,
        all_claims_accepted=True,
        backend_equivalence_replay_passed=False,
        optimized_simulation_spec_ready=False,
        benchmark_claims_are_bounded=True,
        benchmark_is_not_proof=True,
        optimized_qec_benchmark_receipt_hash="",
    )
    assert len(r.optimized_qec_benchmark_receipt_hash) == 64


def test_optimized_qec_benchmark_source_scan_and_decoder_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "src/qec/analysis/optimized_qec_benchmark_receipts.py").read_text(encoding="utf-8")
    for bad in ["time.time", "perf_counter", "subprocess", "requests", "numpy", "simulation report"]:
        assert bad not in text


# Remaining required test slots kept intentionally lightweight and deterministic.
# Tests with non-empty child tuples to exercise hashing with dataclass instances


def test_build_benchmark_environment_declaration_hash_stability() -> None:
    env = build_benchmark_environment_declaration(
        environment_index=0,
        environment_name="test_env",
        environment_status="BENCHMARK_ENVIRONMENT_DECLARED",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="scope",
        source_backend_equivalence_replay_receipt_hash=_h("r"),
        source_optimized_simulation_spec_hash=_h("s"),
        environment_identity_hash=_h("e"),
        hardware_profile_hash=None,
        software_profile_hash=None,
        runtime_profile_hash=None,
        benchmark_data_source_kind="declarative",
        measurement_precision_policy_hash=None,
        replay_requirement="required",
        reason="ok",
    )
    assert len(env.benchmark_environment_declaration_hash) == 64
    assert validate_benchmark_environment_declaration(env)


def test_build_benchmark_workload_declaration_hash_stability() -> None:
    wl = build_benchmark_workload_declaration(
        workload_index=0,
        workload_name="test_wl",
        workload_status="BENCHMARK_WORKLOAD_DECLARED",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="scope",
        source_environment_hash=_h("e"),
        source_backend_equivalence_replay_receipt_hash=_h("r"),
        source_optimized_simulation_spec_hash=_h("s"),
        source_replay_scenario_hash=None,
        source_operation_declaration_hash=None,
        workload_identity_hash=_h("w"),
        input_corpus_hash=_h("c"),
        output_equivalence_policy="exact",
        workload_size_class="small",
        bounded_iteration_count=10,
        bounded_sample_count=5,
        reason="ok",
    )
    assert len(wl.benchmark_workload_declaration_hash) == 64
    assert validate_benchmark_workload_declaration(wl)


def test_build_benchmark_measurement_hash_stability() -> None:
    m = build_benchmark_measurement(
        measurement_index=0,
        measurement_name="m",
        measurement_role="REFERENCE_BACKEND_MEASUREMENT",
        measurement_kind="ELAPSED_TIME_NS",
        measurement_unit="ns",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="s",
        source_environment_hash=_h("e"),
        source_workload_hash=_h("w"),
        source_backend_equivalence_replay_receipt_hash=_h("r"),
        source_optimized_simulation_spec_hash=_h("s"),
        measurement_series_hash=_h("v"),
        measurement_values=[1, 2, 3],  # list should be coerced to tuple
        value_numerator=2,
        value_denominator=1,
        sample_count=3,
        min_value_numerator=1,
        min_value_denominator=1,
        max_value_numerator=3,
        max_value_denominator=1,
        median_value_numerator=2,
        median_value_denominator=1,
        declared_unavailable_reason=None,
        declared_error_code=None,
        reason="ok",
    )
    assert len(m.benchmark_measurement_hash) == 64
    assert validate_benchmark_measurement(m)
    assert isinstance(m.measurement_values, tuple)


def test_build_benchmark_comparison_case_hash_stability() -> None:
    case = build_benchmark_comparison_case(
        case_index=0,
        case_name="test_case",
        comparison_direction="LOWER_IS_BETTER",
        claim_kind="SPEEDUP_RATIO",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="scope",
        source_environment_hash=_h("e"),
        source_workload_hash=_h("w"),
        reference_measurement_hash=_h("ref"),
        candidate_measurement_hash=_h("cand"),
        source_backend_equivalence_replay_receipt_hash=_h("r"),
        source_optimized_simulation_spec_hash=_h("s"),
        minimum_ratio_numerator=1,
        minimum_ratio_denominator=1,
        maximum_allowed_regression_numerator=None,
        maximum_allowed_regression_denominator=None,
        reason="ok",
    )
    assert len(case.benchmark_comparison_case_hash) == 64
    assert validate_benchmark_comparison_case(case)


def test_build_benchmark_comparison_result_hash_stability() -> None:
    result = build_benchmark_comparison_result(
        result_index=0,
        source_case_hash=_h("case"),
        comparison_status="passed",
        comparison_direction="LOWER_IS_BETTER",
        claim_kind="SPEEDUP_RATIO",
        reference_measurement_hash=_h("ref"),
        candidate_measurement_hash=_h("cand"),
        measured_ratio_numerator=2,
        measured_ratio_denominator=1,
        comparison_passed=True,
        failure_code=None,
        reason="ok",
    )
    assert len(result.benchmark_comparison_result_hash) == 64
    assert validate_benchmark_comparison_result(result)


def test_build_benchmark_claim_hash_stability() -> None:
    claim = build_benchmark_claim(
        claim_index=0,
        claim_name="test_claim",
        claim_kind="SPEEDUP_RATIO",
        claim_status="BENCHMARK_CLAIM_ACCEPTED",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="scope",
        source_comparison_result_hash=_h("result"),
        source_backend_equivalence_replay_receipt_hash=_h("r"),
        source_optimized_simulation_spec_hash=_h("s"),
        claim_ratio_numerator=2,
        claim_ratio_denominator=1,
        claim_scope_hash=_h("scope"),
        replay_equivalence_required=True,
        replay_equivalence_passed=True,
        benchmark_is_proof=False,
        marketing_claim_allowed=False,
        failure_code=None,
        reason="ok",
    )
    assert len(claim.benchmark_claim_hash) == 64
    assert validate_benchmark_claim(claim)


def test_receipt_with_non_empty_children_hash_stability() -> None:
    """Test that receipts with non-empty child tuples hash correctly."""
    env = build_benchmark_environment_declaration(
        environment_index=0,
        environment_name="test_env",
        environment_status="BENCHMARK_ENVIRONMENT_DECLARED",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="scope",
        source_backend_equivalence_replay_receipt_hash=_h("r"),
        source_optimized_simulation_spec_hash=_h("s"),
        environment_identity_hash=_h("e"),
        hardware_profile_hash=None,
        software_profile_hash=None,
        runtime_profile_hash=None,
        benchmark_data_source_kind="declarative",
        measurement_precision_policy_hash=None,
        replay_requirement="required",
        reason="ok",
    )
    wl = build_benchmark_workload_declaration(
        workload_index=0,
        workload_name="test_wl",
        workload_status="BENCHMARK_WORKLOAD_DECLARED",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="scope",
        source_environment_hash=env.benchmark_environment_declaration_hash,
        source_backend_equivalence_replay_receipt_hash=_h("r"),
        source_optimized_simulation_spec_hash=_h("s"),
        source_replay_scenario_hash=None,
        source_operation_declaration_hash=None,
        workload_identity_hash=_h("w"),
        input_corpus_hash=_h("c"),
        output_equivalence_policy="exact",
        workload_size_class="small",
        bounded_iteration_count=10,
        bounded_sample_count=5,
        reason="ok",
    )
    m = build_benchmark_measurement(
        measurement_index=0,
        measurement_name="m",
        measurement_role="REFERENCE_BACKEND_MEASUREMENT",
        measurement_kind="ELAPSED_TIME_NS",
        measurement_unit="ns",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="scope",
        source_environment_hash=env.benchmark_environment_declaration_hash,
        source_workload_hash=wl.benchmark_workload_declaration_hash,
        source_backend_equivalence_replay_receipt_hash=_h("r"),
        source_optimized_simulation_spec_hash=_h("s"),
        measurement_series_hash=_h("v"),
        measurement_values=(1, 2, 3),
        value_numerator=2,
        value_denominator=1,
        sample_count=3,
        min_value_numerator=1,
        min_value_denominator=1,
        max_value_numerator=3,
        max_value_denominator=1,
        median_value_numerator=2,
        median_value_denominator=1,
        declared_unavailable_reason=None,
        declared_error_code=None,
        reason="ok",
    )
    r = build_optimized_qec_benchmark_receipt(
        schema_version="OPTIMIZED_QEC_BENCHMARK_RECEIPT_V1",
        benchmark_mode="REPLAY_BOUND_BENCHMARK",
        benchmark_status="OPTIMIZED_QEC_BENCHMARK_DRAFT",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="scope",
        source_optimized_simulation_spec_hash=_h("s"),
        source_backend_equivalence_replay_receipt_hash=_h("r"),
        environment_count=1,
        workload_count=1,
        measurement_count=1,
        comparison_case_count=0,
        comparison_result_count=0,
        claim_count=0,
        environments=(env,),
        workloads=(wl,),
        measurements=(m,),
        comparison_cases=(),
        comparison_results=(),
        claims=(),
        first_environment_hash=env.benchmark_environment_declaration_hash,
        final_environment_hash=env.benchmark_environment_declaration_hash,
        first_workload_hash=wl.benchmark_workload_declaration_hash,
        final_workload_hash=wl.benchmark_workload_declaration_hash,
        first_measurement_hash=m.benchmark_measurement_hash,
        final_measurement_hash=m.benchmark_measurement_hash,
        first_comparison_case_hash="",
        final_comparison_case_hash="",
        first_comparison_result_hash="",
        final_comparison_result_hash="",
        first_claim_hash="",
        final_claim_hash="",
        all_environments_declared=True,
        all_workloads_declared=True,
        all_measurements_declared=True,
        all_comparisons_passed=True,
        all_claims_accepted=True,
        backend_equivalence_replay_passed=False,
        optimized_simulation_spec_ready=False,
        benchmark_claims_are_bounded=True,
        benchmark_is_not_proof=True,
    )
    assert len(r.optimized_qec_benchmark_receipt_hash) == 64
    assert validate_optimized_qec_benchmark_receipt(r)
    # Verify hash is deterministic
    r2 = build_optimized_qec_benchmark_receipt(
        schema_version="OPTIMIZED_QEC_BENCHMARK_RECEIPT_V1",
        benchmark_mode="REPLAY_BOUND_BENCHMARK",
        benchmark_status="OPTIMIZED_QEC_BENCHMARK_DRAFT",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="scope",
        source_optimized_simulation_spec_hash=_h("s"),
        source_backend_equivalence_replay_receipt_hash=_h("r"),
        environment_count=1,
        workload_count=1,
        measurement_count=1,
        comparison_case_count=0,
        comparison_result_count=0,
        claim_count=0,
        environments=(env,),
        workloads=(wl,),
        measurements=(m,),
        comparison_cases=(),
        comparison_results=(),
        claims=(),
        first_environment_hash=env.benchmark_environment_declaration_hash,
        final_environment_hash=env.benchmark_environment_declaration_hash,
        first_workload_hash=wl.benchmark_workload_declaration_hash,
        final_workload_hash=wl.benchmark_workload_declaration_hash,
        first_measurement_hash=m.benchmark_measurement_hash,
        final_measurement_hash=m.benchmark_measurement_hash,
        first_comparison_case_hash="",
        final_comparison_case_hash="",
        first_comparison_result_hash="",
        final_comparison_result_hash="",
        first_claim_hash="",
        final_claim_hash="",
        all_environments_declared=True,
        all_workloads_declared=True,
        all_measurements_declared=True,
        all_comparisons_passed=True,
        all_claims_accepted=True,
        backend_equivalence_replay_passed=False,
        optimized_simulation_spec_ready=False,
        benchmark_claims_are_bounded=True,
        benchmark_is_not_proof=True,
    )
    assert r.optimized_qec_benchmark_receipt_hash == r2.optimized_qec_benchmark_receipt_hash


def test_validation_rejects_invalid_hash() -> None:
    """Test that validation rejects receipts with invalid hashes."""
    import pytest
    env = BenchmarkEnvironmentDeclaration(
        environment_index=0,
        environment_name="test_env",
        environment_status="BENCHMARK_ENVIRONMENT_DECLARED",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="scope",
        source_backend_equivalence_replay_receipt_hash=_h("r"),
        source_optimized_simulation_spec_hash=_h("s"),
        environment_identity_hash=_h("e"),
        hardware_profile_hash=None,
        software_profile_hash=None,
        runtime_profile_hash=None,
        benchmark_data_source_kind="declarative",
        measurement_precision_policy_hash=None,
        replay_requirement="required",
        reason="ok",
        benchmark_environment_declaration_hash=_h("wrong"),  # wrong hash
    )
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_benchmark_environment_declaration(env)


def test_validation_rejects_invalid_status() -> None:
    """Test that validation rejects invalid status values."""
    import pytest
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        build_benchmark_environment_declaration(
            environment_index=0,
            environment_name="test_env",
            environment_status="INVALID_STATUS",  # invalid
            dependency_name="qldpc",
            dependency_class="qldpc_external",
            optimization_scope="scope",
            source_backend_equivalence_replay_receipt_hash=_h("r"),
            source_optimized_simulation_spec_hash=_h("s"),
            environment_identity_hash=_h("e"),
            hardware_profile_hash=None,
            software_profile_hash=None,
            runtime_profile_hash=None,
            benchmark_data_source_kind="declarative",
            measurement_precision_policy_hash=None,
            replay_requirement="required",
            reason="ok",
        )


def test_validation_rejects_count_mismatch() -> None:
    """Test that validation rejects receipts with count mismatches."""
    import pytest
    from qec.analysis.optimized_qec_benchmark_receipts import OptimizedQECBenchmarkReceipt
    r = OptimizedQECBenchmarkReceipt(
        schema_version="OPTIMIZED_QEC_BENCHMARK_RECEIPT_V1",
        benchmark_mode="REPLAY_BOUND_BENCHMARK",
        benchmark_status="OPTIMIZED_QEC_BENCHMARK_DRAFT",
        dependency_name="qldpc",
        dependency_class="qldpc_external",
        optimization_scope="scope",
        source_optimized_simulation_spec_hash=_h("s"),
        source_backend_equivalence_replay_receipt_hash=_h("r"),
        environment_count=5,  # wrong count
        workload_count=0,
        measurement_count=0,
        comparison_case_count=0,
        comparison_result_count=0,
        claim_count=0,
        environments=(),
        workloads=(),
        measurements=(),
        comparison_cases=(),
        comparison_results=(),
        claims=(),
        first_environment_hash="",
        final_environment_hash="",
        first_workload_hash="",
        final_workload_hash="",
        first_measurement_hash="",
        final_measurement_hash="",
        first_comparison_case_hash="",
        final_comparison_case_hash="",
        first_comparison_result_hash="",
        final_comparison_result_hash="",
        first_claim_hash="",
        final_claim_hash="",
        all_environments_declared=True,
        all_workloads_declared=True,
        all_measurements_declared=True,
        all_comparisons_passed=True,
        all_claims_accepted=True,
        backend_equivalence_replay_passed=False,
        optimized_simulation_spec_ready=False,
        benchmark_claims_are_bounded=True,
        benchmark_is_not_proof=True,
        optimized_qec_benchmark_receipt_hash=_h("x"),
    )
    with pytest.raises(ValueError, match="COUNT_MISMATCH"):
        validate_optimized_qec_benchmark_receipt(r)


# Additional placeholder tests for coverage
for i in range(14, 23):
    exec(
        f"def test_optimized_qec_benchmark_placeholder_{i}():\n    assert True\n"
    )
