from __future__ import annotations

import json

import pytest

from qec.verification.formal_benchmark_interface import (
    CATEGORY_ORDER,
    FormalBenchmarkInterface,
    FormalBenchmarkThresholdSet,
    run_formal_benchmark_interface,
    validate_formal_benchmark_report,
)


def _thresholds(*, acceptance_required: bool = False) -> FormalBenchmarkThresholdSet:
    return FormalBenchmarkThresholdSet(
        replay_integrity_min=1.0,
        proof_contract_completeness_min=1.0,
        latency_compliance_min=0.95,
        throughput_compliance_min=0.95,
        equivalence_required=True,
        suppression_receipt_completeness_min=1.0,
        benchmark_acceptance_floor=0.90,
        benchmark_acceptance_required=acceptance_required,
    )


def _inputs() -> dict:
    return {
        "benchmark_summary": {
            "logical_pass_ratio": 1.0,
            "benchmark_pass_ratio": 1.0,
        },
        "proof_contract_receipts": {"completeness_ratio": 1.0},
        "suppression_receipts": {"completeness_ratio": 1.0},
        "latency_throughput_receipts": {
            "latency_compliance_ratio": 1.0,
            "throughput_compliance_ratio": 1.0,
        },
        "equivalence_checks": {
            "replay_integrity_ratio": 1.0,
            "offline_realtime_equivalent": True,
        },
    }


def test_happy_path_merge_ready() -> None:
    report, receipt = run_formal_benchmark_interface(**_inputs(), thresholds=_thresholds())
    assert report.overall_decision == "pass"
    assert receipt.version == "v137.21.3"
    assert receipt.merge_ready is True
    assert receipt.merge_readiness == "merge_ready"


def test_required_logical_failure_blocks_merge() -> None:
    payload = _inputs()
    payload["benchmark_summary"]["logical_pass_ratio"] = 0.75
    report, receipt = run_formal_benchmark_interface(**payload, thresholds=_thresholds())
    assert report.logical_gate_passed is False
    assert "logical_pass_ratio" in report.failing_required_checks
    assert receipt.merge_readiness == "merge_blocked"


def test_required_timing_failure_distinct_from_logical_gate() -> None:
    payload = _inputs()
    payload["latency_throughput_receipts"]["latency_compliance_ratio"] = 0.90
    report, receipt = run_formal_benchmark_interface(**payload, thresholds=_thresholds())
    assert report.logical_gate_passed is True
    assert report.physical_gate_passed is False
    assert receipt.gate_decision == "fail"
    assert receipt.merge_ready is False


def test_advisory_only_warning_path_merge_still_ready() -> None:
    payload = _inputs()
    payload["benchmark_summary"]["benchmark_pass_ratio"] = 0.5
    report, receipt = run_formal_benchmark_interface(
        **payload,
        thresholds=_thresholds(acceptance_required=False),
    )
    assert report.overall_decision == "warn"
    assert report.advisory_warnings == ("benchmark_pass_ratio",)
    assert receipt.merge_ready is True


def test_canonical_ordering_stable_under_shuffled_mappings() -> None:
    payload_a = _inputs()
    payload_b = {
        "benchmark_summary": {"benchmark_pass_ratio": 1.0, "logical_pass_ratio": 1.0},
        "proof_contract_receipts": dict(reversed(list(payload_a["proof_contract_receipts"].items()))),
        "suppression_receipts": dict(reversed(list(payload_a["suppression_receipts"].items()))),
        "latency_throughput_receipts": {
            "throughput_compliance_ratio": 1.0,
            "latency_compliance_ratio": 1.0,
        },
        "equivalence_checks": {
            "offline_realtime_equivalent": True,
            "replay_integrity_ratio": 1.0,
        },
    }
    report_a, receipt_a = run_formal_benchmark_interface(**payload_a, thresholds=_thresholds())
    report_b, receipt_b = run_formal_benchmark_interface(**payload_b, thresholds=_thresholds())
    assert report_a.to_canonical_json() == report_b.to_canonical_json()
    assert receipt_a.to_canonical_json() == receipt_b.to_canonical_json()
    assert report_a.stable_hash() == report_b.stable_hash()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()


def test_malformed_input_fails_fast() -> None:
    payload = _inputs()
    payload["proof_contract_receipts"] = {"wrong": 1.0}
    with pytest.raises(ValueError, match="proof_contract_receipts missing required field"):
        run_formal_benchmark_interface(**payload, thresholds=_thresholds())


def test_stable_hash_repeatability() -> None:
    report_a, receipt_a = run_formal_benchmark_interface(**_inputs(), thresholds=_thresholds())
    report_b, receipt_b = run_formal_benchmark_interface(**_inputs(), thresholds=_thresholds())
    assert report_a.stable_hash() == report_b.stable_hash()
    assert receipt_a.stable_hash() == receipt_b.stable_hash()


def test_threshold_boundary_equality_is_pass() -> None:
    payload = _inputs()
    payload["latency_throughput_receipts"]["latency_compliance_ratio"] = 0.95
    payload["latency_throughput_receipts"]["throughput_compliance_ratio"] = 0.95
    payload["benchmark_summary"]["benchmark_pass_ratio"] = 0.90
    report, receipt = run_formal_benchmark_interface(
        **payload,
        thresholds=_thresholds(acceptance_required=True),
    )
    assert report.overall_decision == "pass"
    assert receipt.gate_decision == "pass"


def test_mixed_category_aggregation_fixed_order_and_counts() -> None:
    payload = _inputs()
    payload["latency_throughput_receipts"]["throughput_compliance_ratio"] = 0.50
    payload["benchmark_summary"]["benchmark_pass_ratio"] = 0.50
    report, _ = run_formal_benchmark_interface(**payload, thresholds=_thresholds())
    assert tuple(report.category_summaries.keys()) == CATEGORY_ORDER
    assert report.category_summaries["physical_timing"]["failed"] == 1
    assert report.category_summaries["benchmark_acceptance"]["advisory_failed"] == 1
    assert report.counts_by_status["failed"] == 2


def test_release_is_additive_and_decoder_untouched() -> None:
    import qec.verification.formal_benchmark_interface as mod

    with open(mod.__file__, "r", encoding="utf-8") as handle:
        source = handle.read()
    assert "qec.decoder" not in source
    assert "decoder-safe" in (mod.__doc__ or "")


def test_validate_report_contract() -> None:
    report, _ = run_formal_benchmark_interface(**_inputs(), thresholds=_thresholds())
    validation = validate_formal_benchmark_report(report)
    assert validation == {"valid": True, "violations": ()}
    rehydrated = json.loads(report.to_canonical_json())
    assert "checks" in rehydrated


def test_interface_threshold_builder_validation() -> None:
    with pytest.raises(TypeError, match="equivalence_required"):
        FormalBenchmarkInterface.build_threshold_set(
            {
                "replay_integrity_min": 1.0,
                "proof_contract_completeness_min": 1.0,
                "latency_compliance_min": 1.0,
                "throughput_compliance_min": 1.0,
                "equivalence_required": "yes",
                "suppression_receipt_completeness_min": 1.0,
                "benchmark_acceptance_floor": 1.0,
            }
        )
