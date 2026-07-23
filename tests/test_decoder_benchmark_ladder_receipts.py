from __future__ import annotations

import ast
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qec.analysis.decoder_benchmark_ladder_receipts import (
    DecoderBenchmarkLadderError,
    DecoderBenchmarkLadderErrorCode,
    build_decoder_benchmark_audit_boundary,
    build_decoder_benchmark_authority_boundary,
    build_decoder_benchmark_claim_scope,
    build_decoder_benchmark_comparator_identity,
    build_decoder_benchmark_comparison_result,
    build_decoder_benchmark_corpus_declaration,
    build_decoder_benchmark_environment_declaration,
    build_decoder_benchmark_execution_boundary,
    build_decoder_benchmark_ladder_identity,
    build_decoder_benchmark_ladder_receipt,
    build_decoder_benchmark_measurement_record,
    build_decoder_benchmark_rollback_gate,
    build_decoder_benchmark_rung,
    build_decoder_benchmark_upstream_binding,
    validate_decoder_benchmark_ladder_receipt,
    validate_decoder_benchmark_measurement_record,
)

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/qec/analysis/decoder_benchmark_ladder_receipts.py"
H = {"a": "a" * 64, "b": "b" * 64, "c": "c" * 64, "d": "d" * 64, "e": "e" * 64, "f": "f" * 64, "0": "0" * 64, "1": "1" * 64, "2": "2" * 64, "3": "3" * 64}


def assert_code(exc: pytest.ExceptionInfo[DecoderBenchmarkLadderError], code: DecoderBenchmarkLadderErrorCode) -> None:
    assert exc.value.code is code
    assert code.value in str(exc.value)
    assert exc.value.detail


def make_parts():
    upstream = build_decoder_benchmark_upstream_binding(
        upstream_canonical_decoder_baseline_receipt_hash=H["a"],
        upstream_decoder_candidate_manifest_hash=H["b"],
        upstream_decoder_replay_equivalence_receipt_hash=H["c"],
        upstream_decoder_optimization_contract_hash=H["d"],
        upstream_decoder_fast_path_equivalence_receipt_hash=H["e"],
        upstream_decoder_implementation_boundary_receipt_hash=H["f"],
        candidate_declaration_hash=H["0"],
        fast_path_identity_hash=H["1"],
        implementation_identity_hash=H["2"],
        candidate_name="declared candidate",
        candidate_version="1",
    )
    identity = build_decoder_benchmark_ladder_identity(
        ladder_id="declared-ladder",
        ladder_name="Declared ladder",
        ladder_version="1",
        associated_candidate_declaration_hash=H["0"],
        associated_fast_path_identity_hash=H["1"],
        associated_implementation_boundary_receipt_hash=H["f"],
    )
    baseline_cmp = build_decoder_benchmark_comparator_identity(
        comparator_id="baseline",
        comparator_name="Canonical baseline",
        comparator_version="1",
        comparator_role="CANONICAL_BASELINE_COMPARATOR",
        comparator_source_hash=H["a"],
    )
    candidate_cmp = build_decoder_benchmark_comparator_identity(
        comparator_id="candidate",
        comparator_name="Candidate adapter",
        comparator_version="1",
        comparator_role="CANDIDATE_DECODER_COMPARATOR",
        comparator_source_hash=H["0"],
    )
    corpus = build_decoder_benchmark_corpus_declaration(
        corpus_id="corpus",
        corpus_name="Declared replay corpus",
        corpus_version="1",
        corpus_source_hash=H["b"],
        replay_corpus_hash=H["c"],
        syndrome_schema_hash=H["d"],
        output_schema_hash=H["e"],
        corpus_selection_rationale="predeclared replay corpus for bounded observation",
        corpus_item_count=2,
    )
    env = build_decoder_benchmark_environment_declaration(
        environment_id="env",
        environment_name="Declared environment",
        environment_version="1",
        hardware_profile_hash=H["a"],
        software_profile_hash=H["b"],
        os_profile_hash=H["c"],
        dependency_profile_hash=H["d"],
        measurement_environment_hash=H["e"],
        cpu_or_accelerator_class="declared cpu class",
    )
    baseline = build_decoder_benchmark_measurement_record(
        measurement_id="baseline-measurement",
        measurement_role="BASELINE_MEASUREMENT",
        comparator_hash=baseline_cmp.decoder_benchmark_comparator_identity_hash,
        corpus_hash=corpus.decoder_benchmark_corpus_declaration_hash,
        environment_hash=env.decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=10,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    candidate = build_decoder_benchmark_measurement_record(
        measurement_id="candidate-measurement",
        measurement_role="CANDIDATE_MEASUREMENT",
        comparator_hash=candidate_cmp.decoder_benchmark_comparator_identity_hash,
        corpus_hash=corpus.decoder_benchmark_corpus_declaration_hash,
        environment_hash=env.decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=5,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    comparison = build_decoder_benchmark_comparison_result(
        comparison_id="comparison",
        baseline_measurement=baseline,
        candidate_measurement=candidate,
    )
    rung = build_decoder_benchmark_rung(
        rung_id="rung",
        rung_index=0,
        rung_kind="BASELINE_VS_CANDIDATE_RUNG",
        corpus_hash=corpus.decoder_benchmark_corpus_declaration_hash,
        environment_hash=env.decoder_benchmark_environment_declaration_hash,
        baseline_comparator_hash=baseline_cmp.decoder_benchmark_comparator_identity_hash,
        candidate_comparator_hash=candidate_cmp.decoder_benchmark_comparator_identity_hash,
        baseline_measurement_hashes=(baseline.decoder_benchmark_measurement_record_hash,),
        candidate_measurement_hashes=(candidate.decoder_benchmark_measurement_record_hash,),
        comparison_results=(comparison,),
    )
    return {
        "upstream_binding": upstream,
        "ladder_identity": identity,
        "comparators": (baseline_cmp, candidate_cmp),
        "corpora": (corpus,),
        "environments": (env,),
        "measurements": (baseline, candidate),
        "comparison_results": (comparison,),
        "rungs": (rung,),
        "claim_scope": build_decoder_benchmark_claim_scope(claim_scope_id="claims"),
        "execution_boundary": build_decoder_benchmark_execution_boundary(execution_boundary_id="exec"),
        "audit_boundary": build_decoder_benchmark_audit_boundary(audit_boundary_id="audit"),
        "rollback_gate": build_decoder_benchmark_rollback_gate(rollback_gate_id="rollback"),
        "authority_boundary": build_decoder_benchmark_authority_boundary(authority_boundary_id="authority"),
    }


def make_receipt(**overrides):
    parts = make_parts()
    parts.update(overrides)
    return build_decoder_benchmark_ladder_receipt(**parts)


def test_happy_path_receipt_is_safe_bound_and_immutable():
    receipt = make_receipt()
    validate_decoder_benchmark_ladder_receipt(receipt)
    assert len(receipt.decoder_benchmark_ladder_receipt_hash) == 64
    assert receipt.decoder_benchmark_ladder_receipt_hash.islower() or receipt.decoder_benchmark_ladder_receipt_hash.isdigit()
    assert receipt.benchmark_ladder_safe is True
    assert receipt.bounded_benchmark_observation_allowed is True
    assert receipt.benchmark_execution_performed_by_receipt is False
    assert receipt.candidate_remains_adapter_only is True
    assert receipt.promotion_allowed is False
    assert receipt.correctness_claim_allowed is False
    assert receipt.global_correctness_claim_allowed is False
    assert receipt.qec_advantage_claim_allowed is False
    assert receipt.hardware_authority_allowed is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.promotion_allowed = True


def test_canonical_ordering_hash_determinism_for_children_and_claims():
    parts = make_parts()
    direct = build_decoder_benchmark_ladder_receipt(**parts)
    shuffled = build_decoder_benchmark_ladder_receipt(
        **{**parts, "comparators": tuple(reversed(parts["comparators"])), "measurements": tuple(reversed(parts["measurements"]))}
    )
    assert direct.decoder_benchmark_ladder_receipt_hash == shuffled.decoder_benchmark_ladder_receipt_hash
    claims_a = build_decoder_benchmark_claim_scope(
        claim_scope_id="claims", allowed_claims=("BOUNDED_RUNTIME_OBSERVATION", "BOUNDED_REGRESSION_OBSERVATION")
    )
    claims_b = build_decoder_benchmark_claim_scope(
        claim_scope_id="claims", allowed_claims=("BOUNDED_REGRESSION_OBSERVATION", "BOUNDED_RUNTIME_OBSERVATION")
    )
    assert claims_a.claim_scope_hash == claims_b.claim_scope_hash
    assert make_receipt().decoder_benchmark_ladder_receipt_hash == make_receipt().decoder_benchmark_ladder_receipt_hash


def test_self_hash_and_payload_hashes_are_recomputed_and_stale_values_fail():
    parts = make_parts()
    measurement = parts["measurements"][0]
    object.__setattr__(measurement, "measurement_payload_hash", H["3"])
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        validate_decoder_benchmark_measurement_record(measurement)
    assert_code(exc, DecoderBenchmarkLadderErrorCode.HASH_MISMATCH)

    receipt = make_receipt()
    object.__setattr__(receipt, "decoder_benchmark_ladder_receipt_hash", H["3"])
    with pytest.raises(DecoderBenchmarkLadderError) as exc2:
        validate_decoder_benchmark_ladder_receipt(receipt)
    assert_code(exc2, DecoderBenchmarkLadderErrorCode.HASH_MISMATCH)


def test_upstream_and_identity_semantics_reject_malformed_or_authoritative_values():
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        build_decoder_benchmark_upstream_binding(
            upstream_canonical_decoder_baseline_receipt_hash="bad",
            upstream_decoder_candidate_manifest_hash=H["b"],
            upstream_decoder_replay_equivalence_receipt_hash=H["c"],
            upstream_decoder_optimization_contract_hash=H["d"],
            upstream_decoder_fast_path_equivalence_receipt_hash=H["e"],
            upstream_decoder_implementation_boundary_receipt_hash=H["f"],
            candidate_declaration_hash=H["0"],
            fast_path_identity_hash=H["1"],
            implementation_identity_hash=H["2"],
            candidate_name="candidate",
            candidate_version="1",
        )
    assert_code(exc, DecoderBenchmarkLadderErrorCode.INVALID_HASH)
    with pytest.raises(DecoderBenchmarkLadderError) as exc2:
        build_decoder_benchmark_upstream_binding(
            upstream_canonical_decoder_baseline_receipt_hash=H["a"],
            upstream_decoder_candidate_manifest_hash=H["b"],
            upstream_decoder_replay_equivalence_receipt_hash=H["c"],
            upstream_decoder_optimization_contract_hash=H["d"],
            upstream_decoder_fast_path_equivalence_receipt_hash=H["e"],
            upstream_decoder_implementation_boundary_receipt_hash=H["f"],
            candidate_declaration_hash=H["0"],
            fast_path_identity_hash=H["1"],
            implementation_identity_hash=H["2"],
            candidate_name="candidate",
            candidate_version="1",
            candidate_promoted=True,
        )
    assert_code(exc2, DecoderBenchmarkLadderErrorCode.INVALID_INPUT)
    with pytest.raises(DecoderBenchmarkLadderError) as exc3:
        build_decoder_benchmark_ladder_identity(
            ladder_id="speed proves correctness",
            ladder_name="Declared ladder",
            ladder_version="1",
            associated_candidate_declaration_hash=H["0"],
            associated_fast_path_identity_hash=H["1"],
            associated_implementation_boundary_receipt_hash=H["f"],
        )
    assert_code(exc3, DecoderBenchmarkLadderErrorCode.INVALID_INPUT)


def test_comparator_corpus_environment_measurement_and_comparison_rejections():
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        build_decoder_benchmark_comparator_identity(
            comparator_id="external comparator authority",
            comparator_name="External",
            comparator_version="1",
            comparator_role="EXTERNAL_ADAPTER_COMPARATOR",
            comparator_source_hash=H["a"],
        )
    assert_code(exc, DecoderBenchmarkLadderErrorCode.INVALID_INPUT)
    with pytest.raises(DecoderBenchmarkLadderError):
        build_decoder_benchmark_corpus_declaration(
            corpus_id="corpus",
            corpus_name="Corpus",
            corpus_version="1",
            corpus_source_hash=H["a"],
            replay_corpus_hash=H["b"],
            syndrome_schema_hash=H["c"],
            output_schema_hash=H["d"],
            corpus_selection_rationale="cherry picked benchmark corpus",
            corpus_item_count=1,
        )
    with pytest.raises(DecoderBenchmarkLadderError):
        build_decoder_benchmark_environment_declaration(
            environment_id="env",
            environment_name="Env",
            environment_version="1",
            hardware_profile_hash=H["a"],
            software_profile_hash=H["b"],
            os_profile_hash=H["c"],
            dependency_profile_hash=H["d"],
            measurement_environment_hash=H["e"],
            cpu_or_accelerator_class="hardware authority",
        )
    parts = make_parts()
    with pytest.raises(DecoderBenchmarkLadderError):
        build_decoder_benchmark_measurement_record(
            measurement_id="bad",
            measurement_role="BASELINE_MEASUREMENT",
            comparator_hash=parts["comparators"][0].decoder_benchmark_comparator_identity_hash,
            corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
            environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
            metric_name="DECLARED_RUNTIME_STEPS",
            metric_unit="STEPS",
            sample_count=True,
            measured_value_numerator=10,
            measured_value_denominator=1,
            dispersion_numerator=0,
            dispersion_denominator=1,
            lower_is_better=True,
        )
    regression_candidate = build_decoder_benchmark_measurement_record(
        measurement_id="regression",
        measurement_role="CANDIDATE_MEASUREMENT",
        comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=20,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    result = build_decoder_benchmark_comparison_result(comparison_id="regression", baseline_measurement=parts["measurements"][0], candidate_measurement=regression_candidate)
    assert result.regression_detected is True
    assert result.improvement_observed is False


def test_boundaries_reject_execution_promotion_authority_and_forged_receipt_flags():
    with pytest.raises(DecoderBenchmarkLadderError):
        build_decoder_benchmark_execution_boundary(execution_boundary_id="exec", timing_api_allowed=True)
    with pytest.raises(DecoderBenchmarkLadderError):
        build_decoder_benchmark_audit_boundary(audit_boundary_id="audit", audit_complete=True)
    with pytest.raises(DecoderBenchmarkLadderError):
        build_decoder_benchmark_rollback_gate(rollback_gate_id="rollback bypass")
    with pytest.raises(DecoderBenchmarkLadderError):
        build_decoder_benchmark_authority_boundary(authority_boundary_id="authority", ml_decoder_authority_allowed=True)
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        make_receipt(promotion_allowed=True)
    assert_code(exc, DecoderBenchmarkLadderErrorCode.INVALID_DECODER_BENCHMARK_LADDER)


def test_aggregate_linkage_and_identity_alignment_are_enforced():
    parts = make_parts()
    missing_cmp_measurement = build_decoder_benchmark_measurement_record(
        measurement_id="orphan",
        measurement_role="BASELINE_MEASUREMENT",
        comparator_hash=H["3"],
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=1,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        build_decoder_benchmark_ladder_receipt(**{**parts, "measurements": parts["measurements"] + (missing_cmp_measurement,)})
    assert_code(exc, DecoderBenchmarkLadderErrorCode.INVALID_DECODER_BENCHMARK_LADDER)
    bad_identity = build_decoder_benchmark_ladder_identity(
        ladder_id="declared-ladder",
        ladder_name="Declared ladder",
        ladder_version="1",
        associated_candidate_declaration_hash=H["3"],
        associated_fast_path_identity_hash=H["1"],
        associated_implementation_boundary_receipt_hash=H["f"],
    )
    with pytest.raises(DecoderBenchmarkLadderError):
        make_receipt(ladder_identity=bad_identity)
    assert make_receipt(comparator_count=True).comparator_count == len(parts["comparators"])

def test_zero_valued_lower_is_better_candidate_is_explicit_infinite_improvement():
    parts = make_parts()
    zero_candidate = build_decoder_benchmark_measurement_record(
        measurement_id="zero-candidate",
        measurement_role="CANDIDATE_MEASUREMENT",
        comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=0,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    result = build_decoder_benchmark_comparison_result(
        comparison_id="zero-improvement",
        baseline_measurement=parts["measurements"][0],
        candidate_measurement=zero_candidate,
    )
    assert result.regression_detected is False
    assert result.improvement_observed is True
    assert result.exact_metric_match is False
    assert (result.improvement_ratio_numerator, result.improvement_ratio_denominator) == (1, 0)


def test_comparison_requires_same_corpus_and_environment():
    parts = make_parts()
    other_corpus = build_decoder_benchmark_corpus_declaration(
        corpus_id="other-corpus",
        corpus_name="Other declared replay corpus",
        corpus_version="1",
        corpus_source_hash=H["0"],
        replay_corpus_hash=H["1"],
        syndrome_schema_hash=H["2"],
        output_schema_hash=H["3"],
        corpus_selection_rationale="predeclared alternate corpus for bounded observation",
        corpus_item_count=1,
    )
    other_candidate = build_decoder_benchmark_measurement_record(
        measurement_id="other-corpus-candidate",
        measurement_role="CANDIDATE_MEASUREMENT",
        comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        corpus_hash=other_corpus.decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=5,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        build_decoder_benchmark_comparison_result(
            comparison_id="cross-corpus",
            baseline_measurement=parts["measurements"][0],
            candidate_measurement=other_candidate,
        )
    assert_code(exc, DecoderBenchmarkLadderErrorCode.INVALID_INPUT)


def test_aggregate_rejects_measurement_comparator_role_mismatch():
    parts = make_parts()
    bad_baseline = build_decoder_benchmark_measurement_record(
        measurement_id="bad-baseline-role",
        measurement_role="BASELINE_MEASUREMENT",
        comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=10,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    comparison = build_decoder_benchmark_comparison_result(
        comparison_id="bad-role-comparison",
        baseline_measurement=bad_baseline,
        candidate_measurement=parts["measurements"][1],
    )
    rung = build_decoder_benchmark_rung(
        rung_id="bad-role-rung",
        rung_index=0,
        rung_kind="BASELINE_VS_CANDIDATE_RUNG",
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        baseline_comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        candidate_comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        baseline_measurement_hashes=(bad_baseline.decoder_benchmark_measurement_record_hash,),
        candidate_measurement_hashes=(parts["measurements"][1].decoder_benchmark_measurement_record_hash,),
        comparison_results=(comparison,),
    )
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        build_decoder_benchmark_ladder_receipt(**{**parts, "measurements": (bad_baseline, parts["measurements"][1]), "comparison_results": (comparison,), "rungs": (rung,)})
    assert_code(exc, DecoderBenchmarkLadderErrorCode.INVALID_DECODER_BENCHMARK_LADDER)


def test_aggregate_rejects_rung_children_outside_declared_scope():
    parts = make_parts()
    other_corpus = build_decoder_benchmark_corpus_declaration(
        corpus_id="other-corpus",
        corpus_name="Other declared replay corpus",
        corpus_version="1",
        corpus_source_hash=H["0"],
        replay_corpus_hash=H["1"],
        syndrome_schema_hash=H["2"],
        output_schema_hash=H["3"],
        corpus_selection_rationale="predeclared alternate corpus for bounded observation",
        corpus_item_count=1,
    )
    other_baseline = build_decoder_benchmark_measurement_record(
        measurement_id="other-baseline",
        measurement_role="BASELINE_MEASUREMENT",
        comparator_hash=parts["comparators"][0].decoder_benchmark_comparator_identity_hash,
        corpus_hash=other_corpus.decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=10,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    other_candidate = build_decoder_benchmark_measurement_record(
        measurement_id="other-candidate",
        measurement_role="CANDIDATE_MEASUREMENT",
        comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        corpus_hash=other_corpus.decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=5,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    comparison = build_decoder_benchmark_comparison_result(comparison_id="other-comparison", baseline_measurement=other_baseline, candidate_measurement=other_candidate)
    bad_rung = build_decoder_benchmark_rung(
        rung_id="cross-scope-rung",
        rung_index=0,
        rung_kind="BASELINE_VS_CANDIDATE_RUNG",
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        baseline_comparator_hash=parts["comparators"][0].decoder_benchmark_comparator_identity_hash,
        candidate_comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        baseline_measurement_hashes=(other_baseline.decoder_benchmark_measurement_record_hash,),
        candidate_measurement_hashes=(other_candidate.decoder_benchmark_measurement_record_hash,),
        comparison_results=(comparison,),
    )
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        build_decoder_benchmark_ladder_receipt(**{**parts, "corpora": parts["corpora"] + (other_corpus,), "measurements": (other_baseline, other_candidate), "comparison_results": (comparison,), "rungs": (bad_rung,)})
    assert_code(exc, DecoderBenchmarkLadderErrorCode.INVALID_DECODER_BENCHMARK_LADDER)


def test_aggregate_recomputes_rung_pass_flags_from_referenced_comparisons():
    parts = make_parts()
    regression_candidate = build_decoder_benchmark_measurement_record(
        measurement_id="aggregate-regression",
        measurement_role="CANDIDATE_MEASUREMENT",
        comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=20,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    comparison = build_decoder_benchmark_comparison_result(comparison_id="aggregate-regression", baseline_measurement=parts["measurements"][0], candidate_measurement=regression_candidate)
    bad_rung = build_decoder_benchmark_rung(
        rung_id="forged-pass-rung",
        rung_index=0,
        rung_kind="BASELINE_VS_CANDIDATE_RUNG",
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        baseline_comparator_hash=parts["comparators"][0].decoder_benchmark_comparator_identity_hash,
        candidate_comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        baseline_measurement_hashes=(parts["measurements"][0].decoder_benchmark_measurement_record_hash,),
        candidate_measurement_hashes=(regression_candidate.decoder_benchmark_measurement_record_hash,),
        comparison_result_hashes=(comparison.decoder_benchmark_comparison_result_hash,),
        rung_metric_names=(comparison.metric_name,),
        rung_passed=True,
        rung_regression_detected=False,
    )
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        build_decoder_benchmark_ladder_receipt(**{**parts, "measurements": (parts["measurements"][0], regression_candidate), "comparison_results": (comparison,), "rungs": (bad_rung,)})
    assert_code(exc, DecoderBenchmarkLadderErrorCode.INVALID_DECODER_BENCHMARK_LADDER)


def test_aggregate_rejects_duplicate_and_sparse_rung_indices():
    parts = make_parts()
    duplicate = build_decoder_benchmark_rung(
        rung_id="duplicate-index-rung",
        rung_index=0,
        rung_kind="BASELINE_VS_CANDIDATE_RUNG",
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        baseline_comparator_hash=parts["comparators"][0].decoder_benchmark_comparator_identity_hash,
        candidate_comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        baseline_measurement_hashes=(parts["measurements"][0].decoder_benchmark_measurement_record_hash,),
        candidate_measurement_hashes=(parts["measurements"][1].decoder_benchmark_measurement_record_hash,),
        comparison_results=parts["comparison_results"],
    )
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        build_decoder_benchmark_ladder_receipt(**{**parts, "rungs": parts["rungs"] + (duplicate,)})
    assert_code(exc, DecoderBenchmarkLadderErrorCode.INVALID_INPUT)

    sparse = build_decoder_benchmark_rung(
        rung_id="sparse-index-rung",
        rung_index=1,
        rung_kind="BASELINE_VS_CANDIDATE_RUNG",
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        baseline_comparator_hash=parts["comparators"][0].decoder_benchmark_comparator_identity_hash,
        candidate_comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        baseline_measurement_hashes=(parts["measurements"][0].decoder_benchmark_measurement_record_hash,),
        candidate_measurement_hashes=(parts["measurements"][1].decoder_benchmark_measurement_record_hash,),
        comparison_results=parts["comparison_results"],
    )
    with pytest.raises(DecoderBenchmarkLadderError) as exc2:
        build_decoder_benchmark_ladder_receipt(**{**parts, "rungs": (sparse,)})
    assert_code(exc2, DecoderBenchmarkLadderErrorCode.INVALID_INPUT)


@pytest.mark.parametrize(
    "phrase",
    [
        "silent_decoder_replacement",
        "candidate-replaces-baseline",
        "decoder replaced because faster",
        "speed\nproves\tcorrectness",
        "benchmark\\nproves\\tcorrectness",
        "benchmark marketing",
        "runtime promotion",
        "candidate decoder promoted",
        "probabilistic decoder authority",
        "ML decoder authority",
        "hardware authority",
        "QEC advantage proven",
        "hidden precision drift",
        "undeclared approximation policy",
        "output accepted as universal canonical truth",
        "global correctness proven",
        "replay equivalence implies promotion",
        "optimization grants execution authority",
        "fast path runtime enabled",
        "benchmark replaces replay equivalence",
        "benchmark replaces rollback",
        "performance marketing claim",
    ],
)
def test_forbidden_semantic_hardening(phrase):
    with pytest.raises(DecoderBenchmarkLadderError):
        build_decoder_benchmark_ladder_identity(
            ladder_id=phrase,
            ladder_name="Declared ladder",
            ladder_version="1",
            associated_candidate_declaration_hash=H["0"],
            associated_fast_path_identity_hash=H["1"],
            associated_implementation_boundary_receipt_hash=H["f"],
        )

def test_zero_valued_lower_is_better_candidate_is_explicit_infinite_improvement():
    parts = make_parts()
    zero_candidate = build_decoder_benchmark_measurement_record(
        measurement_id="zero-candidate",
        measurement_role="CANDIDATE_MEASUREMENT",
        comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=0,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    result = build_decoder_benchmark_comparison_result(
        comparison_id="zero-improvement",
        baseline_measurement=parts["measurements"][0],
        candidate_measurement=zero_candidate,
    )
    assert result.regression_detected is False
    assert result.improvement_observed is True
    assert result.exact_metric_match is False
    assert (result.improvement_ratio_numerator, result.improvement_ratio_denominator) == (1, 0)


def test_comparison_requires_same_corpus_and_environment():
    parts = make_parts()
    other_corpus = build_decoder_benchmark_corpus_declaration(
        corpus_id="other-corpus",
        corpus_name="Other declared replay corpus",
        corpus_version="1",
        corpus_source_hash=H["0"],
        replay_corpus_hash=H["1"],
        syndrome_schema_hash=H["2"],
        output_schema_hash=H["3"],
        corpus_selection_rationale="predeclared alternate corpus for bounded observation",
        corpus_item_count=1,
    )
    other_candidate = build_decoder_benchmark_measurement_record(
        measurement_id="other-corpus-candidate",
        measurement_role="CANDIDATE_MEASUREMENT",
        comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        corpus_hash=other_corpus.decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=5,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        build_decoder_benchmark_comparison_result(
            comparison_id="cross-corpus",
            baseline_measurement=parts["measurements"][0],
            candidate_measurement=other_candidate,
        )
    assert_code(exc, DecoderBenchmarkLadderErrorCode.INVALID_INPUT)


def test_aggregate_rejects_measurement_comparator_role_mismatch():
    parts = make_parts()
    bad_baseline = build_decoder_benchmark_measurement_record(
        measurement_id="bad-baseline-role",
        measurement_role="BASELINE_MEASUREMENT",
        comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=10,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    comparison = build_decoder_benchmark_comparison_result(
        comparison_id="bad-role-comparison",
        baseline_measurement=bad_baseline,
        candidate_measurement=parts["measurements"][1],
    )
    rung = build_decoder_benchmark_rung(
        rung_id="bad-role-rung",
        rung_index=0,
        rung_kind="BASELINE_VS_CANDIDATE_RUNG",
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        baseline_comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        candidate_comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        baseline_measurement_hashes=(bad_baseline.decoder_benchmark_measurement_record_hash,),
        candidate_measurement_hashes=(parts["measurements"][1].decoder_benchmark_measurement_record_hash,),
        comparison_results=(comparison,),
    )
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        build_decoder_benchmark_ladder_receipt(**{**parts, "measurements": (bad_baseline, parts["measurements"][1]), "comparison_results": (comparison,), "rungs": (rung,)})
    assert_code(exc, DecoderBenchmarkLadderErrorCode.INVALID_DECODER_BENCHMARK_LADDER)


def test_aggregate_rejects_rung_children_outside_declared_scope():
    parts = make_parts()
    other_corpus = build_decoder_benchmark_corpus_declaration(
        corpus_id="other-corpus",
        corpus_name="Other declared replay corpus",
        corpus_version="1",
        corpus_source_hash=H["0"],
        replay_corpus_hash=H["1"],
        syndrome_schema_hash=H["2"],
        output_schema_hash=H["3"],
        corpus_selection_rationale="predeclared alternate corpus for bounded observation",
        corpus_item_count=1,
    )
    other_baseline = build_decoder_benchmark_measurement_record(
        measurement_id="other-baseline",
        measurement_role="BASELINE_MEASUREMENT",
        comparator_hash=parts["comparators"][0].decoder_benchmark_comparator_identity_hash,
        corpus_hash=other_corpus.decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=10,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    other_candidate = build_decoder_benchmark_measurement_record(
        measurement_id="other-candidate",
        measurement_role="CANDIDATE_MEASUREMENT",
        comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        corpus_hash=other_corpus.decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=5,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    comparison = build_decoder_benchmark_comparison_result(comparison_id="other-comparison", baseline_measurement=other_baseline, candidate_measurement=other_candidate)
    bad_rung = build_decoder_benchmark_rung(
        rung_id="cross-scope-rung",
        rung_index=0,
        rung_kind="BASELINE_VS_CANDIDATE_RUNG",
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        baseline_comparator_hash=parts["comparators"][0].decoder_benchmark_comparator_identity_hash,
        candidate_comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        baseline_measurement_hashes=(other_baseline.decoder_benchmark_measurement_record_hash,),
        candidate_measurement_hashes=(other_candidate.decoder_benchmark_measurement_record_hash,),
        comparison_results=(comparison,),
    )
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        build_decoder_benchmark_ladder_receipt(**{**parts, "corpora": parts["corpora"] + (other_corpus,), "measurements": (other_baseline, other_candidate), "comparison_results": (comparison,), "rungs": (bad_rung,)})
    assert_code(exc, DecoderBenchmarkLadderErrorCode.INVALID_DECODER_BENCHMARK_LADDER)


def test_aggregate_recomputes_rung_pass_flags_from_referenced_comparisons():
    parts = make_parts()
    regression_candidate = build_decoder_benchmark_measurement_record(
        measurement_id="aggregate-regression",
        measurement_role="CANDIDATE_MEASUREMENT",
        comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        metric_name="DECLARED_RUNTIME_STEPS",
        metric_unit="STEPS",
        sample_count=1,
        measured_value_numerator=20,
        measured_value_denominator=1,
        dispersion_numerator=0,
        dispersion_denominator=1,
        lower_is_better=True,
    )
    comparison = build_decoder_benchmark_comparison_result(comparison_id="aggregate-regression", baseline_measurement=parts["measurements"][0], candidate_measurement=regression_candidate)
    bad_rung = build_decoder_benchmark_rung(
        rung_id="forged-pass-rung",
        rung_index=0,
        rung_kind="BASELINE_VS_CANDIDATE_RUNG",
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        baseline_comparator_hash=parts["comparators"][0].decoder_benchmark_comparator_identity_hash,
        candidate_comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        baseline_measurement_hashes=(parts["measurements"][0].decoder_benchmark_measurement_record_hash,),
        candidate_measurement_hashes=(regression_candidate.decoder_benchmark_measurement_record_hash,),
        comparison_result_hashes=(comparison.decoder_benchmark_comparison_result_hash,),
        rung_metric_names=(comparison.metric_name,),
        rung_passed=True,
        rung_regression_detected=False,
    )
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        build_decoder_benchmark_ladder_receipt(**{**parts, "measurements": (parts["measurements"][0], regression_candidate), "comparison_results": (comparison,), "rungs": (bad_rung,)})
    assert_code(exc, DecoderBenchmarkLadderErrorCode.INVALID_DECODER_BENCHMARK_LADDER)


def test_aggregate_rejects_duplicate_and_sparse_rung_indices():
    parts = make_parts()
    duplicate = build_decoder_benchmark_rung(
        rung_id="duplicate-index-rung",
        rung_index=0,
        rung_kind="BASELINE_VS_CANDIDATE_RUNG",
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        baseline_comparator_hash=parts["comparators"][0].decoder_benchmark_comparator_identity_hash,
        candidate_comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        baseline_measurement_hashes=(parts["measurements"][0].decoder_benchmark_measurement_record_hash,),
        candidate_measurement_hashes=(parts["measurements"][1].decoder_benchmark_measurement_record_hash,),
        comparison_results=parts["comparison_results"],
    )
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        build_decoder_benchmark_ladder_receipt(**{**parts, "rungs": parts["rungs"] + (duplicate,)})
    assert_code(exc, DecoderBenchmarkLadderErrorCode.INVALID_INPUT)

    sparse = build_decoder_benchmark_rung(
        rung_id="sparse-index-rung",
        rung_index=1,
        rung_kind="BASELINE_VS_CANDIDATE_RUNG",
        corpus_hash=parts["corpora"][0].decoder_benchmark_corpus_declaration_hash,
        environment_hash=parts["environments"][0].decoder_benchmark_environment_declaration_hash,
        baseline_comparator_hash=parts["comparators"][0].decoder_benchmark_comparator_identity_hash,
        candidate_comparator_hash=parts["comparators"][1].decoder_benchmark_comparator_identity_hash,
        baseline_measurement_hashes=(parts["measurements"][0].decoder_benchmark_measurement_record_hash,),
        candidate_measurement_hashes=(parts["measurements"][1].decoder_benchmark_measurement_record_hash,),
        comparison_results=parts["comparison_results"],
    )
    with pytest.raises(DecoderBenchmarkLadderError) as exc2:
        build_decoder_benchmark_ladder_receipt(**{**parts, "rungs": (sparse,)})
    assert_code(exc2, DecoderBenchmarkLadderErrorCode.INVALID_INPUT)


@pytest.mark.parametrize(
    "phrase",
    [
        "benchmark_ladder_safe",
        "bounded_benchmark_observation_allowed",
        "benchmark_ladder_required",
        "rollback_receipt_required_before_promotion",
        "declared benchmark measurement",
    ],
)
def test_positive_semantic_controls_are_allowed(phrase):
    build_decoder_benchmark_ladder_identity(
        ladder_id=phrase,
        ladder_name="Declared ladder",
        ladder_version="1",
        associated_candidate_declaration_hash=H["0"],
        associated_fast_path_identity_hash=H["1"],
        associated_implementation_boundary_receipt_hash=H["f"],
    )


def test_claim_scope_rejects_unrecognized_forbidden_claims():
    with pytest.raises(DecoderBenchmarkLadderError) as exc:
        build_decoder_benchmark_claim_scope(
            claim_scope_id="claims",
            forbidden_claims=tuple(sorted({*{
                "CORRECTNESS_CLAIM",
                "GLOBAL_CORRECTNESS_CLAIM",
                "QEC_ADVANTAGE_CLAIM",
                "HARDWARE_AUTHORITY_CLAIM",
                "PROMOTION_CLAIM",
                "BENCHMARK_MARKETING_CLAIM",
                "UNIVERSAL_SPEEDUP_CLAIM",
                "BASELINE_REPLACEMENT_CLAIM",
            }, "CORRECTNESS_CLAIM_TYPO"})),
        )
    assert_code(exc, DecoderBenchmarkLadderErrorCode.INVALID_INPUT)


def test_boundary_static_import_and_runtime_marker_checks():
    tree = ast.parse(MODULE.read_text())
    banned = ("qec.decoder", "numpy", "scipy", "qldpc", "stim", "pymatching", "qiskit", "qutip", "pandas", "polars", "torch", "tensorflow", "jax")
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not [name for name in imports if name == "qec.decoder" or name.startswith(tuple(prefix + "." for prefix in banned)) or name in banned]
    text = MODULE.read_text()
    forbidden_markers = [
        "time.perf_counter", "time.time", "datetime.now", "requests.", "urllib.request", "socket.", "importlib.import_module(\"qec.decoder",
        "subprocess.run", "benchmark_loop(", "execute_decoder", "run_decoder_workload", "fast_path_runtime_execute",
    ]
    assert not [marker for marker in forbidden_markers if marker in text]


def test_legacy_decoder_source_unchanged_and_no_forbidden_runtime_created():
    diff = subprocess.run(
        ["git", "diff", "--name-only", "--", "src/qec/decoder/"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    assert not [
        path
        for path in diff
        if not path.startswith("src/qec/decoder/qutrit/")
    ]

    created = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True, text=True, capture_output=True).stdout
    paths = [line[3:] for line in created.splitlines() if len(line) > 3]
    assert not [
        path
        for path in paths
        if path.startswith("src/qec/decoder/")
        and not path.startswith("src/qec/decoder/qutrit/")
    ]
    forbidden_markers = [
        "fast_path_runtime",
        "candidate_decoder",
        "benchmark_runtime",
        "implementation_runtime",
    ]
    assert not [
        path
        for path in paths
        if any(marker in path for marker in forbidden_markers)
    ]


def test_hash_seed_stability_subprocess():
    code = "from tests.test_decoder_benchmark_ladder_receipts import make_receipt; print(make_receipt().decoder_benchmark_ladder_receipt_hash)"
    outputs = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONPATH": f"src{os.pathsep}.", "PYTHONHASHSEED": seed}
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1]
