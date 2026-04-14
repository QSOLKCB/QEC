from __future__ import annotations

import dataclasses

import pytest

from qec.orchestration.autonomous_research_orchestration_kernel import (
    OrchestrationStep,
    ResearchTask,
    build_autonomous_research_plan,
)
from qec.orchestration.deterministic_experiment_scheduling_kernel import (
    SchedulingLane,
    build_deterministic_experiment_schedule,
)
from qec.orchestration.deterministic_research_audit_kernel import (
    ResearchAuditFinding,
    run_deterministic_research_audit,
    traverse_research_audit_report,
    validate_research_audit_report,
)
from qec.orchestration.replay_safe_benchmark_pipeline_kernel import (
    BenchmarkResult,
    BenchmarkStage,
    build_replay_safe_benchmark_pipeline,
)
from qec.orchestration.research_trace_lineage_kernel import build_research_trace_lineage


def _sample_artifacts():
    tasks = (
        ResearchTask(task_id="task_proof", task_kind="proof", source_ref="node:0", priority=0, task_epoch=0),
        ResearchTask(task_id="task_validation", task_kind="validation", source_ref="node:1", priority=1, task_epoch=1),
    )
    steps = (
        OrchestrationStep(step_id="exp_proof", task_id="task_proof", execution_order=0, dependency_refs=(), step_epoch=0),
        OrchestrationStep(
            step_id="exp_validation",
            task_id="task_validation",
            execution_order=1,
            dependency_refs=("exp_proof",),
            step_epoch=1,
        ),
    )
    plan = build_autonomous_research_plan("plan_v137_17_4", tasks, steps)

    lanes = (
        SchedulingLane(lane_id="lane_proof_0", lane_kind="proof", capacity=1, lane_epoch=0),
        SchedulingLane(lane_id="lane_validation_0", lane_kind="validation", capacity=1, lane_epoch=0),
    )
    schedule = build_deterministic_experiment_schedule("schedule_v137_17_4", plan, lanes)

    stages = (
        BenchmarkStage(stage_id="stage_prepare", stage_kind="prepare", input_ref="exp_proof", stage_order=0, stage_epoch=0),
        BenchmarkStage(stage_id="stage_verify", stage_kind="verify", input_ref="exp_validation", stage_order=0, stage_epoch=1),
    )
    results = (
        BenchmarkResult(
            result_id="result_metric",
            stage_id="stage_prepare",
            experiment_id="exp_proof",
            result_kind="metric",
            result_hash="0" * 64,
            result_epoch=0,
        ),
        BenchmarkResult(
            result_id="result_verification",
            stage_id="stage_verify",
            experiment_id="exp_validation",
            result_kind="verification",
            result_hash="1" * 64,
            result_epoch=1,
        ),
    )
    pipeline = build_replay_safe_benchmark_pipeline("pipeline_v137_17_4", schedule, stages, results)
    lineage = build_research_trace_lineage("lineage_v137_17_4", plan, schedule, pipeline)
    return plan, schedule, pipeline, lineage


def test_repeated_run_byte_identity() -> None:
    plan, schedule, pipeline, lineage = _sample_artifacts()
    a = run_deterministic_research_audit("audit_repeat_bytes", plan, schedule, pipeline, lineage)
    b = run_deterministic_research_audit("audit_repeat_bytes", plan, schedule, pipeline, lineage)
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_repeated_run_audit_hash_identity() -> None:
    plan, schedule, pipeline, lineage = _sample_artifacts()
    a = run_deterministic_research_audit("audit_repeat_hash", plan, schedule, pipeline, lineage)
    b = run_deterministic_research_audit("audit_repeat_hash", plan, schedule, pipeline, lineage)
    assert a.audit_hash == b.audit_hash


def test_repeated_run_flow_snapshot_identity() -> None:
    plan, schedule, pipeline, lineage = _sample_artifacts()
    a = run_deterministic_research_audit("audit_repeat_snapshot", plan, schedule, pipeline, lineage)
    b = run_deterministic_research_audit("audit_repeat_snapshot", plan, schedule, pipeline, lineage)
    assert a.flow_snapshot.to_canonical_bytes() == b.flow_snapshot.to_canonical_bytes()


def test_invalid_severity_rejection() -> None:
    plan, schedule, pipeline, lineage = _sample_artifacts()
    report = run_deterministic_research_audit("audit_invalid_severity", plan, schedule, pipeline, lineage)
    bad_finding = dataclasses.replace(report.findings[0] if report.findings else ResearchAuditFinding("finding::x", "ordering_violation", "audit:x", "info", 0), severity="fatal")
    tampered = dataclasses.replace(report, findings=(bad_finding,) + report.findings[1:])
    with pytest.raises(ValueError, match="invalid severity"):
        validate_research_audit_report(tampered)


def test_invalid_finding_kind_rejection() -> None:
    plan, schedule, pipeline, lineage = _sample_artifacts()
    report = run_deterministic_research_audit("audit_invalid_kind", plan, schedule, pipeline, lineage)
    bad_finding = dataclasses.replace(report.findings[0] if report.findings else ResearchAuditFinding("finding::x", "ordering_violation", "audit:x", "info", 0), finding_kind="unknown_kind")
    tampered = dataclasses.replace(report, findings=(bad_finding,) + report.findings[1:])
    with pytest.raises(ValueError, match="invalid finding kind"):
        validate_research_audit_report(tampered)


def test_tampered_audit_hash_rejection() -> None:
    plan, schedule, pipeline, lineage = _sample_artifacts()
    report = run_deterministic_research_audit("audit_tampered", plan, schedule, pipeline, lineage)
    tampered = dataclasses.replace(report, audit_hash="f" * 64)
    validation = validate_research_audit_report(tampered)
    assert validation.is_valid is False
    assert "audit_hash_mismatch" in validation.violations


def test_flow_discontinuity_detection() -> None:
    plan, schedule, pipeline, lineage = _sample_artifacts()
    truncated = dataclasses.replace(lineage, edges=())
    report = run_deterministic_research_audit("audit_flow_gap", plan, schedule, pipeline, truncated)
    assert any(f.finding_kind == "flow_discontinuity" for f in report.findings)


def test_deterministic_full_traversal() -> None:
    plan, schedule, pipeline, lineage = _sample_artifacts()
    report = run_deterministic_research_audit("audit_full_traversal", plan, schedule, pipeline, lineage)
    a = traverse_research_audit_report(report, "full")
    b = traverse_research_audit_report(report, "full")
    assert a.ordered_findings_trace == b.ordered_findings_trace
    assert a.ordered_flow_trace == b.ordered_flow_trace
    assert a.traversal_hash == b.traversal_hash


def test_deterministic_flow_traversal() -> None:
    plan, schedule, pipeline, lineage = _sample_artifacts()
    report = run_deterministic_research_audit("audit_flow_traversal", plan, schedule, pipeline, lineage)
    a = traverse_research_audit_report(report, "flow")
    b = traverse_research_audit_report(report, "flow")
    assert a.ordered_findings_trace == b.ordered_findings_trace
    assert a.ordered_flow_trace == b.ordered_flow_trace
    assert a.traversal_hash == b.traversal_hash


def test_deterministic_replay_traversal() -> None:
    plan, schedule, pipeline, lineage = _sample_artifacts()
    report = run_deterministic_research_audit("audit_replay_traversal", plan, schedule, pipeline, lineage)
    a = traverse_research_audit_report(report, "replay")
    b = traverse_research_audit_report(report, "replay")
    assert a.ordered_findings_trace == b.ordered_findings_trace
    assert a.ordered_flow_trace == b.ordered_flow_trace
    assert a.traversal_hash == b.traversal_hash


def test_canonical_export_stability() -> None:
    plan, schedule, pipeline, lineage = _sample_artifacts()
    report = run_deterministic_research_audit("audit_export", plan, schedule, pipeline, lineage)
    assert report.to_canonical_json() == report.to_canonical_json()
    assert report.to_canonical_bytes() == report.to_canonical_bytes()


def test_valid_artifacts_no_continuity_findings() -> None:
    plan, schedule, pipeline, lineage = _sample_artifacts()
    report = run_deterministic_research_audit("audit_happy_path", plan, schedule, pipeline, lineage)
    continuity_kinds = {"continuity_gap", "flow_discontinuity", "missing_root"}
    continuity_findings = [f for f in report.findings if f.finding_kind in continuity_kinds]
    assert continuity_findings == [], (
        f"Expected no continuity/flow findings for valid artifacts, got: {continuity_findings}"
    )
