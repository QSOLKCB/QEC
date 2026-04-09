"""Tests for v137.13.4 — Signal Abstraction Certification Battery."""

from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.hybrid_signal_interface import build_hybrid_signal_trace
from qec.analysis.morphology_transition_kernel import run_morphology_transition_kernel
from qec.analysis.neuromorphic_substrate_simulator import compile_substrate_report
from qec.analysis.phase_boundary_topology_kernel import run_phase_boundary_topology_kernel
from qec.analysis.region_correspondence_kernel import run_region_correspondence_kernel
from qec.analysis.signal_abstraction_certification_battery import (
    SCHEMA_VERSION,
    SignalAbstractionCertificationResult,
    SignalAbstractionEvidence,
    build_ascii_certification_summary,
    build_certification_report,
    export_certification_report_json,
    run_signal_abstraction_certification_battery,
)
from qec.analysis.synthetic_signal_geometry_kernel import run_signal_geometry_kernel


def _base_input(sim_id: str, signal: list[int]) -> dict[str, object]:
    return {
        "simulation_id": sim_id,
        "node_count": 2,
        "input_signal": signal,
        "threshold": 5,
        "time_steps": 4,
        "decay_factor": 0.5,
        "epoch_id": f"epoch-{sim_id}",
        "schema_version": "v137.12.0",
    }


def _make_stack():
    report_a = compile_substrate_report(_base_input("sim-a", [1, 2, 3, 4]))
    trace_a = build_hybrid_signal_trace(report_a)

    report_b = compile_substrate_report(_base_input("sim-b", [4, 3, 2, 1]))
    trace_b = build_hybrid_signal_trace(report_b)

    geometry_a, _ = run_signal_geometry_kernel(trace_a)
    geometry_b, _ = run_signal_geometry_kernel(trace_b)

    morphology_a, _ = run_morphology_transition_kernel(geometry_a.trajectory)
    morphology_b, _ = run_morphology_transition_kernel(geometry_b.trajectory)

    topology_a, _ = run_phase_boundary_topology_kernel(morphology_a.path)
    topology_b, _ = run_phase_boundary_topology_kernel(morphology_b.path)

    correspondence, _ = run_region_correspondence_kernel((topology_a.path, topology_b.path))

    return geometry_a, morphology_a, topology_a, correspondence


def test_same_input_same_bytes() -> None:
    stack = _make_stack()
    report_a, receipt_a = run_signal_abstraction_certification_battery(*stack)
    report_b, receipt_b = run_signal_abstraction_certification_battery(*stack)
    assert report_a.to_canonical_bytes() == report_b.to_canonical_bytes()
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_same_input_same_hash() -> None:
    stack = _make_stack()
    report_a, receipt_a = run_signal_abstraction_certification_battery(*stack)
    report_b, receipt_b = run_signal_abstraction_certification_battery(*stack)
    assert report_a.report_hash == report_b.report_hash
    assert receipt_a.receipt_hash == receipt_b.receipt_hash


def test_repeated_run_byte_identity() -> None:
    stack = _make_stack()
    outputs = tuple(run_signal_abstraction_certification_battery(*stack) for _ in range(3))
    report_bytes = tuple(item[0].to_canonical_bytes() for item in outputs)
    receipt_bytes = tuple(item[1].to_canonical_bytes() for item in outputs)
    assert len(set(report_bytes)) == 1
    assert len(set(receipt_bytes)) == 1


def test_broken_lineage_rejection() -> None:
    geometry, morphology, topology, correspondence = _make_stack()
    broken_topology = replace(
        topology,
        path=replace(topology.path, input_transition_hash="0" * 64),
    )
    with pytest.raises(ValueError, match="broken lineage"):
        run_signal_abstraction_certification_battery(geometry, morphology, broken_topology, correspondence)


def test_corrupted_metric_rejection() -> None:
    geometry, morphology, topology, correspondence = _make_stack()
    corrupted = replace(geometry, geometry_integrity_score=1.100000000000)
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        run_signal_abstraction_certification_battery(corrupted, morphology, topology, correspondence)


def test_canonical_export_stability() -> None:
    stack = _make_stack()
    report, _ = run_signal_abstraction_certification_battery(*stack)
    assert report.to_canonical_json() == report.to_canonical_json()
    assert report.to_canonical_bytes() == report.to_canonical_bytes()
    assert export_certification_report_json(report) == report.to_canonical_json()


def test_bounded_metric_validation() -> None:
    stack = _make_stack()
    report, _ = run_signal_abstraction_certification_battery(*stack)
    result = report.result
    assert 0.0 <= result.determinism_score <= 1.0
    assert 0.0 <= result.lineage_integrity_score <= 1.0
    assert 0.0 <= result.metric_integrity_score <= 1.0
    assert 0.0 <= result.cross_layer_consistency_score <= 1.0
    assert 0.0 <= result.global_certification_score <= 1.0
    assert 0.0 <= result.certification_score <= 1.0


def test_wrapper_manual_equivalence() -> None:
    stack = _make_stack()
    report, receipt = run_signal_abstraction_certification_battery(*stack)

    manual_result = SignalAbstractionCertificationResult(
        validation_passed=report.result.validation_passed,
        certification_score=report.result.certification_score,
        failure_reasons=report.result.failure_reasons,
        report_hash="",
        receipt_hash="",
        determinism_score=report.result.determinism_score,
        lineage_integrity_score=report.result.lineage_integrity_score,
        metric_integrity_score=report.result.metric_integrity_score,
        cross_layer_consistency_score=report.result.cross_layer_consistency_score,
        global_certification_score=report.result.global_certification_score,
    )
    manual_report = build_certification_report(report.evidence, manual_result)
    assert manual_report.report_hash == report.report_hash
    assert receipt.report_hash == report.report_hash


def test_report_integrity() -> None:
    stack = _make_stack()
    report, _ = run_signal_abstraction_certification_battery(*stack)
    assert report.result.report_hash == report.report_hash
    assert report.stable_sha256() == report.report_hash
    assert report.schema_version == SCHEMA_VERSION


def test_receipt_integrity() -> None:
    stack = _make_stack()
    report, receipt = run_signal_abstraction_certification_battery(*stack)
    assert receipt.report_hash == report.report_hash
    assert receipt.stable_sha256() == receipt.receipt_hash
    assert receipt.receipt_chain[-1] != ""
    assert len(receipt.receipt_chain) == 5


def test_full_stack_certification_pass() -> None:
    stack = _make_stack()
    report, receipt = run_signal_abstraction_certification_battery(*stack)
    assert report.result.validation_passed is True
    assert report.result.failure_reasons == ()
    assert report.result.certification_score == pytest.approx(1.0)
    assert report.result.receipt_hash == receipt.receipt_hash


def test_intentional_certification_fail_path() -> None:
    geometry, morphology, topology, correspondence = _make_stack()
    evidence = SignalAbstractionEvidence(
        geometry_hash=geometry.stable_hash,
        morphology_hash=morphology.stable_hash,
        topology_hash=topology.stable_hash,
        correspondence_hash=correspondence.stable_hash,
        receipt_chain=(
            geometry.stable_hash,
            morphology.stable_hash,
            topology.stable_hash,
            correspondence.stable_hash,
            "f" * 64,
        ),
    )
    base_result = SignalAbstractionCertificationResult(
        validation_passed=False,
        certification_score=0.500000000000,
        failure_reasons=("manual.intentional_failure",),
        report_hash="",
        receipt_hash="",
        determinism_score=1.000000000000,
        lineage_integrity_score=0.500000000000,
        metric_integrity_score=1.000000000000,
        cross_layer_consistency_score=0.500000000000,
        global_certification_score=0.500000000000,
    )
    report = build_certification_report(evidence, base_result)
    assert report.result.validation_passed is False
    assert report.result.failure_reasons == ("manual.intentional_failure",)
    summary = build_ascii_certification_summary(report)
    assert "Validation Passed:        False" in summary
