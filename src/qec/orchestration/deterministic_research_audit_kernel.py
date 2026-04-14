"""v137.17.4 — Deterministic Research Audit Kernel.

Deterministic replay + flow-integrity auditing over plan, schedule, pipeline,
and lineage artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from qec.orchestration.autonomous_research_orchestration_kernel import (
    AutonomousResearchPlan,
    build_autonomous_research_plan,
    validate_autonomous_research_plan,
)
from qec.orchestration.deterministic_experiment_scheduling_kernel import (
    DeterministicExperimentSchedule,
    ScheduledExperiment,
    SchedulingLane,
    build_deterministic_experiment_schedule,
    validate_deterministic_experiment_schedule,
)
from qec.orchestration.replay_safe_benchmark_pipeline_kernel import (
    ReplaySafeBenchmarkPipeline,
    build_replay_safe_benchmark_pipeline,
    validate_replay_safe_benchmark_pipeline,
)
from qec.orchestration.research_trace_lineage_kernel import (
    ResearchTraceLineage,
    build_research_trace_lineage,
    normalize_research_trace_input,
    validate_research_trace_lineage,
)


VALID_AUDIT_FINDING_KINDS: Tuple[str, ...] = (
    "artifact_hash_drift",
    "continuity_gap",
    "flow_discontinuity",
    "hash_mismatch",
    "missing_root",
    "ordering_violation",
    "scheduler_lane_drift",
    "traversal_drift",
    "validation_mismatch",
)

VALID_AUDIT_SEVERITIES: Tuple[str, ...] = (
    "info",
    "warning",
    "error",
    "critical",
)

VALID_AUDIT_TRAVERSAL_MODES: Tuple[str, ...] = (
    "full",
    "critical",
    "continuity",
    "replay",
    "flow",
)

PlanLike = Union[AutonomousResearchPlan, Mapping[str, Any]]
ScheduleLike = Union[DeterministicExperimentSchedule, Mapping[str, Any]]
PipelineLike = Union[ReplaySafeBenchmarkPipeline, Mapping[str, Any]]
LineageLike = Union[ResearchTraceLineage, Mapping[str, Any]]
FindingLike = Union["ResearchAuditFinding", Mapping[str, Any]]
FlowSnapshotLike = Union["ResearchAuditFlowSnapshot", Mapping[str, Any]]
ReportLike = Union["ResearchAuditReport", Mapping[str, Any]]


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _canonical_bytes(data: Any) -> bytes:
    return _canonical_json(data).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _require_non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer, not bool")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a non-negative integer")
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _require_valid_hash(value: Any, field_name: str) -> str:
    normalized = _require_non_empty_string(value, field_name)
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{field_name} must be a 64-char lowercase SHA-256 hex")
    return normalized


def _severity_rank(severity: str) -> int:
    return {"info": 0, "warning": 1, "error": 2, "critical": 3}.get(severity, 99)


@dataclass(frozen=True)
class ResearchAuditFinding:
    finding_id: str
    finding_kind: str
    artifact_ref: str
    severity: str
    finding_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "finding_kind": self.finding_kind,
            "artifact_ref": self.artifact_ref,
            "severity": self.severity,
            "finding_epoch": self.finding_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ResearchAuditFlowSnapshot:
    snapshot_id: str
    plan_flow_hash: str
    schedule_flow_hash: str
    pipeline_flow_hash: str
    lineage_flow_hash: str
    snapshot_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "plan_flow_hash": self.plan_flow_hash,
            "schedule_flow_hash": self.schedule_flow_hash,
            "pipeline_flow_hash": self.pipeline_flow_hash,
            "lineage_flow_hash": self.lineage_flow_hash,
            "snapshot_epoch": self.snapshot_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ResearchAuditReport:
    audit_id: str
    plan_id: str
    schedule_id: str
    pipeline_id: str
    lineage_id: str
    findings: Tuple[ResearchAuditFinding, ...]
    flow_snapshot: ResearchAuditFlowSnapshot
    audit_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "plan_id": self.plan_id,
            "schedule_id": self.schedule_id,
            "pipeline_id": self.pipeline_id,
            "lineage_id": self.lineage_id,
            "findings": [finding.to_dict() for finding in self.findings],
            "flow_snapshot": self.flow_snapshot.to_dict(),
            "audit_hash": self.audit_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ResearchAuditReceipt:
    receipt_id: str
    audit_id: str
    audit_hash: str
    traversal_mode: str
    ordered_findings_trace: Tuple[str, ...]
    ordered_flow_trace: Tuple[str, ...]
    traversal_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "audit_id": self.audit_id,
            "audit_hash": self.audit_hash,
            "traversal_mode": self.traversal_mode,
            "ordered_findings_trace": list(self.ordered_findings_trace),
            "ordered_flow_trace": list(self.ordered_flow_trace),
            "traversal_hash": self.traversal_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ResearchAuditValidationReport:
    audit_id: str
    is_valid: bool
    finding_validity_ok: bool
    severity_validity_ok: bool
    ordering_validity_ok: bool
    audit_hash_validity_ok: bool
    flow_snapshot_validity_ok: bool
    violations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "is_valid": self.is_valid,
            "finding_validity_ok": self.finding_validity_ok,
            "severity_validity_ok": self.severity_validity_ok,
            "ordering_validity_ok": self.ordering_validity_ok,
            "audit_hash_validity_ok": self.audit_hash_validity_ok,
            "flow_snapshot_validity_ok": self.flow_snapshot_validity_ok,
            "violations": list(self.violations),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


def _finding_sort_key(finding: ResearchAuditFinding) -> Tuple[int, int, str, str, str]:
    return (
        finding.finding_epoch,
        _severity_rank(finding.severity),
        finding.finding_kind,
        finding.artifact_ref,
        finding.finding_id,
    )


def _finding_id(finding_kind: str, artifact_ref: str, severity: str, finding_epoch: int) -> str:
    payload = f"{finding_kind}|{artifact_ref}|{severity}|{finding_epoch}".encode("utf-8")
    return f"finding::{_sha256_hex(payload)[:16]}"


def _normalize_plan(plan: PlanLike) -> AutonomousResearchPlan:
    if isinstance(plan, AutonomousResearchPlan):
        return plan
    if not isinstance(plan, Mapping):
        raise ValueError("plan must be mapping or AutonomousResearchPlan")
    return build_autonomous_research_plan(
        plan_id=_require_non_empty_string(plan.get("plan_id", ""), "plan_id"),
        tasks=plan.get("tasks", ()),
        steps=plan.get("steps", ()),
    )


def _normalize_schedule(schedule: ScheduleLike) -> DeterministicExperimentSchedule:
    if isinstance(schedule, DeterministicExperimentSchedule):
        return schedule
    if not isinstance(schedule, Mapping):
        raise ValueError("schedule must be mapping or DeterministicExperimentSchedule")
    if "plan" in schedule:
        return build_deterministic_experiment_schedule(
            schedule_id=_require_non_empty_string(schedule.get("schedule_id", ""), "schedule_id"),
            plan=_normalize_plan(schedule.get("plan", {})),
            lanes=schedule.get("lanes", ()),
        )
    # Serialized schedule form: {schedule_id, lanes, scheduled_experiments, schedule_hash}
    normalized = DeterministicExperimentSchedule(
        schedule_id=_require_non_empty_string(schedule.get("schedule_id", ""), "schedule_id"),
        lanes=tuple(
            lane if isinstance(lane, SchedulingLane) else SchedulingLane(
                lane_id=_require_non_empty_string(lane.get("lane_id", ""), "lane_id"),
                lane_kind=_require_non_empty_string(lane.get("lane_kind", ""), "lane_kind"),
                capacity=_require_non_negative_int(lane.get("capacity", 0), "capacity"),
                lane_epoch=_require_non_negative_int(lane.get("lane_epoch", 0), "lane_epoch"),
            )
            for lane in schedule.get("lanes", ())
        ),
        scheduled_experiments=tuple(
            exp if isinstance(exp, ScheduledExperiment) else ScheduledExperiment(
                experiment_id=_require_non_empty_string(exp.get("experiment_id", ""), "experiment_id"),
                task_id=_require_non_empty_string(exp.get("task_id", ""), "task_id"),
                lane_id=_require_non_empty_string(exp.get("lane_id", ""), "lane_id"),
                execution_slot=_require_non_negative_int(exp.get("execution_slot", 0), "execution_slot"),
                priority=_require_non_negative_int(exp.get("priority", 0), "priority"),
                schedule_epoch=_require_non_negative_int(exp.get("schedule_epoch", 0), "schedule_epoch"),
            )
            for exp in schedule.get("scheduled_experiments", ())
        ),
        schedule_hash=_require_non_empty_string(schedule.get("schedule_hash", ""), "schedule_hash"),
    )
    validate_deterministic_experiment_schedule(normalized)
    return normalized


def _normalize_pipeline(pipeline: PipelineLike, schedule: DeterministicExperimentSchedule) -> ReplaySafeBenchmarkPipeline:
    if isinstance(pipeline, ReplaySafeBenchmarkPipeline):
        return pipeline
    if not isinstance(pipeline, Mapping):
        raise ValueError("pipeline must be mapping or ReplaySafeBenchmarkPipeline")
    normalized = build_replay_safe_benchmark_pipeline(
        pipeline_id=_require_non_empty_string(pipeline.get("pipeline_id", ""), "pipeline_id"),
        schedule=schedule,
        stages=pipeline.get("stages", ()),
        results=pipeline.get("results", ()),
    )
    # Validate that any serialized integrity fields in the mapping match the rebuilt pipeline.
    for field, expected in (
        ("schedule_id", normalized.schedule_id),
        ("schedule_hash", normalized.schedule_hash),
        ("pipeline_hash", normalized.pipeline_hash),
    ):
        serialized = pipeline.get(field)
        if serialized is not None and serialized != expected:
            raise ValueError(f"pipeline {field} mismatch")
    return normalized


def _normalize_lineage(
    lineage: LineageLike,
    plan: AutonomousResearchPlan,
    schedule: DeterministicExperimentSchedule,
    pipeline: ReplaySafeBenchmarkPipeline,
) -> ResearchTraceLineage:
    if isinstance(lineage, ResearchTraceLineage):
        return lineage
    if not isinstance(lineage, Mapping):
        raise ValueError("lineage must be mapping or ResearchTraceLineage")
    if "lineage_hash" in lineage:
        # Serialized lineage form: {lineage_id, nodes, edges, lineage_hash}
        lineage_id = _require_non_empty_string(lineage.get("lineage_id", ""), "lineage_id")
        lineage_hash = _require_valid_hash(lineage.get("lineage_hash", ""), "lineage_hash")
        ordered_nodes, ordered_edges = normalize_research_trace_input(
            lineage.get("nodes", ()),
            lineage.get("edges", ()),
        )
        return ResearchTraceLineage(
            lineage_id=lineage_id,
            nodes=ordered_nodes,
            edges=ordered_edges,
            lineage_hash=lineage_hash,
        )
    return build_research_trace_lineage(
        lineage_id=_require_non_empty_string(lineage.get("lineage_id", ""), "lineage_id"),
        plan=plan,
        schedule=schedule,
        pipeline=pipeline,
    )


def _artifact_epoch_max(plan: AutonomousResearchPlan, schedule: DeterministicExperimentSchedule, pipeline: ReplaySafeBenchmarkPipeline, lineage: ResearchTraceLineage) -> int:
    plan_epoch = max([task.task_epoch for task in plan.tasks] + [step.step_epoch for step in plan.steps] + [0])
    schedule_epoch = max([lane.lane_epoch for lane in schedule.lanes] + [exp.schedule_epoch for exp in schedule.scheduled_experiments] + [0])
    pipeline_epoch = max([stage.stage_epoch for stage in pipeline.stages] + [result.result_epoch for result in pipeline.results] + [0])
    lineage_epoch = max([node.trace_epoch for node in lineage.nodes] + [edge.trace_epoch for edge in lineage.edges] + [0])
    return max(plan_epoch, schedule_epoch, pipeline_epoch, lineage_epoch)


def _flow_hash(artifact_kind: str, artifact_id: str, artifact_hash: str) -> str:
    payload = {"artifact_kind": artifact_kind, "artifact_id": artifact_id, "artifact_hash": artifact_hash}
    return _sha256_hex(_canonical_bytes(payload))


def _build_flow_snapshot(
    audit_id: str,
    plan: AutonomousResearchPlan,
    schedule: DeterministicExperimentSchedule,
    pipeline: ReplaySafeBenchmarkPipeline,
    lineage: ResearchTraceLineage,
) -> ResearchAuditFlowSnapshot:
    snapshot_epoch = _artifact_epoch_max(plan, schedule, pipeline, lineage)
    plan_flow_hash = _flow_hash("plan", plan.plan_id, plan.plan_hash)
    schedule_flow_hash = _flow_hash("schedule", schedule.schedule_id, schedule.schedule_hash)
    pipeline_flow_hash = _flow_hash("pipeline", pipeline.pipeline_id, pipeline.pipeline_hash)
    lineage_flow_hash = _flow_hash("lineage", lineage.lineage_id, lineage.lineage_hash)
    snapshot_id = f"snapshot::{_sha256_hex(_canonical_bytes({'audit_id': audit_id, 'snapshot_epoch': snapshot_epoch, 'plan_flow_hash': plan_flow_hash, 'schedule_flow_hash': schedule_flow_hash, 'pipeline_flow_hash': pipeline_flow_hash, 'lineage_flow_hash': lineage_flow_hash}))[:16]}"
    return ResearchAuditFlowSnapshot(
        snapshot_id=snapshot_id,
        plan_flow_hash=plan_flow_hash,
        schedule_flow_hash=schedule_flow_hash,
        pipeline_flow_hash=pipeline_flow_hash,
        lineage_flow_hash=lineage_flow_hash,
        snapshot_epoch=snapshot_epoch,
    )


def _normalize_finding(raw: FindingLike) -> ResearchAuditFinding:
    if isinstance(raw, ResearchAuditFinding):
        finding = ResearchAuditFinding(
            finding_id=_require_non_empty_string(raw.finding_id, "finding_id"),
            finding_kind=_require_non_empty_string(raw.finding_kind, "finding_kind"),
            artifact_ref=_require_non_empty_string(raw.artifact_ref, "artifact_ref"),
            severity=_require_non_empty_string(raw.severity, "severity"),
            finding_epoch=_require_non_negative_int(raw.finding_epoch, "finding_epoch"),
        )
    else:
        if not isinstance(raw, Mapping):
            raise ValueError("finding must be mapping or ResearchAuditFinding")
        finding = ResearchAuditFinding(
            finding_id=_require_non_empty_string(raw.get("finding_id", ""), "finding_id"),
            finding_kind=_require_non_empty_string(raw.get("finding_kind", ""), "finding_kind"),
            artifact_ref=_require_non_empty_string(raw.get("artifact_ref", ""), "artifact_ref"),
            severity=_require_non_empty_string(raw.get("severity", ""), "severity"),
            finding_epoch=_require_non_negative_int(raw.get("finding_epoch", 0), "finding_epoch"),
        )
    if finding.finding_kind not in VALID_AUDIT_FINDING_KINDS:
        raise ValueError(f"invalid finding kind: {finding.finding_kind}")
    if finding.severity not in VALID_AUDIT_SEVERITIES:
        raise ValueError(f"invalid severity: {finding.severity}")
    return finding


def _normalize_flow_snapshot(raw: FlowSnapshotLike) -> ResearchAuditFlowSnapshot:
    if isinstance(raw, ResearchAuditFlowSnapshot):
        snapshot = ResearchAuditFlowSnapshot(
            snapshot_id=_require_non_empty_string(raw.snapshot_id, "snapshot_id"),
            plan_flow_hash=_require_valid_hash(raw.plan_flow_hash, "plan_flow_hash"),
            schedule_flow_hash=_require_valid_hash(raw.schedule_flow_hash, "schedule_flow_hash"),
            pipeline_flow_hash=_require_valid_hash(raw.pipeline_flow_hash, "pipeline_flow_hash"),
            lineage_flow_hash=_require_valid_hash(raw.lineage_flow_hash, "lineage_flow_hash"),
            snapshot_epoch=_require_non_negative_int(raw.snapshot_epoch, "snapshot_epoch"),
        )
    else:
        if not isinstance(raw, Mapping):
            raise ValueError("flow_snapshot must be mapping or ResearchAuditFlowSnapshot")
        snapshot = ResearchAuditFlowSnapshot(
            snapshot_id=_require_non_empty_string(raw.get("snapshot_id", ""), "snapshot_id"),
            plan_flow_hash=_require_valid_hash(raw.get("plan_flow_hash", ""), "plan_flow_hash"),
            schedule_flow_hash=_require_valid_hash(raw.get("schedule_flow_hash", ""), "schedule_flow_hash"),
            pipeline_flow_hash=_require_valid_hash(raw.get("pipeline_flow_hash", ""), "pipeline_flow_hash"),
            lineage_flow_hash=_require_valid_hash(raw.get("lineage_flow_hash", ""), "lineage_flow_hash"),
            snapshot_epoch=_require_non_negative_int(raw.get("snapshot_epoch", 0), "snapshot_epoch"),
        )
    return snapshot


def _compute_audit_hash(audit_id: str, plan_id: str, schedule_id: str, pipeline_id: str, lineage_id: str, findings: Tuple[ResearchAuditFinding, ...], flow_snapshot: ResearchAuditFlowSnapshot) -> str:
    payload = {
        "audit_id": audit_id,
        "plan_id": plan_id,
        "schedule_id": schedule_id,
        "pipeline_id": pipeline_id,
        "lineage_id": lineage_id,
        "findings": [finding.to_dict() for finding in findings],
        "flow_snapshot": flow_snapshot.to_dict(),
    }
    return _sha256_hex(_canonical_bytes(payload))


def _normalize_report(report: ReportLike) -> ResearchAuditReport:
    if isinstance(report, ResearchAuditReport):
        findings = tuple(_normalize_finding(raw) for raw in report.findings)
        flow_snapshot = _normalize_flow_snapshot(report.flow_snapshot)
        audit_hash = _require_valid_hash(report.audit_hash, "audit_hash")
        return ResearchAuditReport(
            audit_id=_require_non_empty_string(report.audit_id, "audit_id"),
            plan_id=_require_non_empty_string(report.plan_id, "plan_id"),
            schedule_id=_require_non_empty_string(report.schedule_id, "schedule_id"),
            pipeline_id=_require_non_empty_string(report.pipeline_id, "pipeline_id"),
            lineage_id=_require_non_empty_string(report.lineage_id, "lineage_id"),
            findings=findings,
            flow_snapshot=flow_snapshot,
            audit_hash=audit_hash,
        )
    if not isinstance(report, Mapping):
        raise ValueError("report must be mapping or ResearchAuditReport")
    findings = tuple(_normalize_finding(raw) for raw in report.get("findings", ()))
    return ResearchAuditReport(
        audit_id=_require_non_empty_string(report.get("audit_id", ""), "audit_id"),
        plan_id=_require_non_empty_string(report.get("plan_id", ""), "plan_id"),
        schedule_id=_require_non_empty_string(report.get("schedule_id", ""), "schedule_id"),
        pipeline_id=_require_non_empty_string(report.get("pipeline_id", ""), "pipeline_id"),
        lineage_id=_require_non_empty_string(report.get("lineage_id", ""), "lineage_id"),
        findings=findings,
        flow_snapshot=_normalize_flow_snapshot(report.get("flow_snapshot", {})),
        audit_hash=_require_valid_hash(report.get("audit_hash", ""), "audit_hash"),
    )


def run_deterministic_research_audit(
    audit_id: str,
    plan: PlanLike,
    schedule: ScheduleLike,
    pipeline: PipelineLike,
    lineage: LineageLike,
) -> ResearchAuditReport:
    normalized_audit_id = _require_non_empty_string(audit_id, "audit_id")
    normalized_plan = _normalize_plan(plan)
    normalized_schedule = _normalize_schedule(schedule)
    normalized_pipeline = _normalize_pipeline(pipeline, normalized_schedule)
    normalized_lineage = _normalize_lineage(lineage, normalized_plan, normalized_schedule, normalized_pipeline)

    findings: List[ResearchAuditFinding] = []

    for artifact_name, validation_report in (
        ("plan", validate_autonomous_research_plan(normalized_plan)),
        ("schedule", validate_deterministic_experiment_schedule(normalized_schedule)),
        ("pipeline", validate_replay_safe_benchmark_pipeline(normalized_pipeline)),
        ("lineage", validate_research_trace_lineage(normalized_lineage)),
    ):
        if not validation_report.is_valid:
            findings.append(
                ResearchAuditFinding(
                    finding_id=_finding_id("validation_mismatch", f"{artifact_name}:{getattr(validation_report, artifact_name + '_id', getattr(validation_report, 'lineage_id', 'unknown'))}", "error", 0),
                    finding_kind="validation_mismatch",
                    artifact_ref=f"{artifact_name}:{getattr(validation_report, artifact_name + '_id', getattr(validation_report, 'lineage_id', 'unknown'))}",
                    severity="error",
                    finding_epoch=0,
                )
            )

    if normalized_pipeline.schedule_hash != normalized_schedule.schedule_hash:
        findings.append(
            ResearchAuditFinding(
                finding_id=_finding_id("artifact_hash_drift", f"schedule:{normalized_schedule.schedule_id}", "critical", 1),
                finding_kind="artifact_hash_drift",
                artifact_ref=f"schedule:{normalized_schedule.schedule_id}",
                severity="critical",
                finding_epoch=1,
            )
        )

    if normalized_pipeline.schedule_id != normalized_schedule.schedule_id:
        findings.append(
            ResearchAuditFinding(
                finding_id=_finding_id("validation_mismatch", f"schedule:{normalized_schedule.schedule_id}", "error", 1),
                finding_kind="validation_mismatch",
                artifact_ref=f"schedule:{normalized_schedule.schedule_id}",
                severity="error",
                finding_epoch=1,
            )
        )

    lineage_by_ref = {node.source_ref: node for node in normalized_lineage.nodes}
    # build_research_trace_lineage creates nodes only for plan and schedule as root-level
    # artifacts; pipeline stages/results get their own nodes but no pipeline-level root node
    # is created. Checking for a pipeline root would always produce a false continuity_gap.
    expected_pairs = (
        ("plan", normalized_plan.plan_id, normalized_plan.plan_hash),
        ("schedule", normalized_schedule.schedule_id, normalized_schedule.schedule_hash),
    )
    for artifact_kind, artifact_id, artifact_hash in expected_pairs:
        ref = f"{artifact_kind}:{artifact_id}"
        node = lineage_by_ref.get(ref)
        if node is None:
            kind = "missing_root" if artifact_kind == "plan" else "continuity_gap"
            findings.append(
                ResearchAuditFinding(
                    finding_id=_finding_id(kind, ref, "critical" if kind == "missing_root" else "error", 2),
                    finding_kind=kind,
                    artifact_ref=ref,
                    severity="critical" if kind == "missing_root" else "error",
                    finding_epoch=2,
                )
            )
        elif node.trace_hash != artifact_hash:
            findings.append(
                ResearchAuditFinding(
                    finding_id=_finding_id("hash_mismatch", ref, "critical", 3),
                    finding_kind="hash_mismatch",
                    artifact_ref=ref,
                    severity="critical",
                    finding_epoch=3,
                )
            )

    edge_pairs = {(edge.source_trace_id, edge.target_trace_id) for edge in normalized_lineage.edges}
    plan_node = lineage_by_ref.get(f"plan:{normalized_plan.plan_id}")
    schedule_node = lineage_by_ref.get(f"schedule:{normalized_schedule.schedule_id}")
    pipeline_node = lineage_by_ref.get(f"pipeline:{normalized_pipeline.pipeline_id}")
    if plan_node and schedule_node and (schedule_node.trace_id, plan_node.trace_id) not in edge_pairs:
        findings.append(
            ResearchAuditFinding(
                finding_id=_finding_id("flow_discontinuity", f"schedule:{normalized_schedule.schedule_id}", "error", 4),
                finding_kind="flow_discontinuity",
                artifact_ref=f"schedule:{normalized_schedule.schedule_id}",
                severity="error",
                finding_epoch=4,
            )
        )
    if schedule_node and pipeline_node and (pipeline_node.trace_id, schedule_node.trace_id) not in edge_pairs:
        findings.append(
            ResearchAuditFinding(
                finding_id=_finding_id("flow_discontinuity", f"pipeline:{normalized_pipeline.pipeline_id}", "error", 4),
                finding_kind="flow_discontinuity",
                artifact_ref=f"pipeline:{normalized_pipeline.pipeline_id}",
                severity="error",
                finding_epoch=4,
            )
        )

    task_kind_by_id = {task.task_id: task.task_kind for task in normalized_plan.tasks}
    lane_kind_by_id = {lane.lane_id: lane.lane_kind for lane in normalized_schedule.lanes}
    for scheduled in normalized_schedule.scheduled_experiments:
        task_kind = task_kind_by_id.get(scheduled.task_id)
        lane_kind = lane_kind_by_id.get(scheduled.lane_id)
        if task_kind is not None and lane_kind is not None and task_kind != lane_kind:
            findings.append(
                ResearchAuditFinding(
                    finding_id=_finding_id("scheduler_lane_drift", f"experiment:{scheduled.experiment_id}", "error", 5),
                    finding_kind="scheduler_lane_drift",
                    artifact_ref=f"experiment:{scheduled.experiment_id}",
                    severity="error",
                    finding_epoch=5,
                )
            )

    flow_snapshot = _build_flow_snapshot(
        normalized_audit_id,
        normalized_plan,
        normalized_schedule,
        normalized_pipeline,
        normalized_lineage,
    )
    ordered_findings = tuple(sorted(findings, key=_finding_sort_key))
    audit_hash = _compute_audit_hash(
        normalized_audit_id,
        normalized_plan.plan_id,
        normalized_schedule.schedule_id,
        normalized_pipeline.pipeline_id,
        normalized_lineage.lineage_id,
        ordered_findings,
        flow_snapshot,
    )
    return ResearchAuditReport(
        audit_id=normalized_audit_id,
        plan_id=normalized_plan.plan_id,
        schedule_id=normalized_schedule.schedule_id,
        pipeline_id=normalized_pipeline.pipeline_id,
        lineage_id=normalized_lineage.lineage_id,
        findings=ordered_findings,
        flow_snapshot=flow_snapshot,
        audit_hash=audit_hash,
    )


def validate_research_audit_report(report: ReportLike) -> ResearchAuditValidationReport:
    normalized_report = _normalize_report(report)
    violations: List[str] = []

    ordering_validity_ok = normalized_report.findings == tuple(sorted(normalized_report.findings, key=_finding_sort_key))
    if not ordering_validity_ok:
        violations.append("finding_ordering_violation")

    flow_snapshot_validity_ok = True
    snapshot = normalized_report.flow_snapshot
    for field_name, field_value in (
        ("plan_flow_hash", snapshot.plan_flow_hash),
        ("schedule_flow_hash", snapshot.schedule_flow_hash),
        ("pipeline_flow_hash", snapshot.pipeline_flow_hash),
        ("lineage_flow_hash", snapshot.lineage_flow_hash),
    ):
        if len(field_value) != 64 or any(ch not in "0123456789abcdef" for ch in field_value):
            flow_snapshot_validity_ok = False
            violations.append(f"invalid_{field_name}")

    expected_audit_hash = _compute_audit_hash(
        normalized_report.audit_id,
        normalized_report.plan_id,
        normalized_report.schedule_id,
        normalized_report.pipeline_id,
        normalized_report.lineage_id,
        normalized_report.findings,
        normalized_report.flow_snapshot,
    )
    audit_hash_validity_ok = expected_audit_hash == normalized_report.audit_hash
    if not audit_hash_validity_ok:
        violations.append("audit_hash_mismatch")

    is_valid = ordering_validity_ok and audit_hash_validity_ok and flow_snapshot_validity_ok
    return ResearchAuditValidationReport(
        audit_id=normalized_report.audit_id,
        is_valid=is_valid,
        finding_validity_ok=True,
        severity_validity_ok=True,
        ordering_validity_ok=ordering_validity_ok,
        audit_hash_validity_ok=audit_hash_validity_ok,
        flow_snapshot_validity_ok=flow_snapshot_validity_ok,
        violations=tuple(violations),
    )


def traverse_research_audit_report(report: ReportLike, traversal_mode: str) -> ResearchAuditReceipt:
    normalized_report = _normalize_report(report)
    normalized_mode = _require_non_empty_string(traversal_mode, "traversal_mode")
    if normalized_mode not in VALID_AUDIT_TRAVERSAL_MODES:
        raise ValueError(f"unsupported traversal mode: {normalized_mode}")

    findings = tuple(sorted(normalized_report.findings, key=_finding_sort_key))
    if normalized_mode == "critical":
        findings = tuple(finding for finding in findings if finding.severity in {"error", "critical"})
    elif normalized_mode == "continuity":
        findings = tuple(
            finding
            for finding in findings
            if finding.finding_kind in {"continuity_gap", "flow_discontinuity", "missing_root", "ordering_violation"}
        )
    elif normalized_mode == "replay":
        findings = tuple(
            finding
            for finding in findings
            if finding.finding_kind in {"artifact_hash_drift", "hash_mismatch", "traversal_drift", "validation_mismatch"}
        )
    elif normalized_mode == "flow":
        findings = tuple(
            finding
            for finding in findings
            if finding.finding_kind in {"flow_discontinuity", "scheduler_lane_drift", "ordering_violation"}
        )

    ordered_findings_trace = tuple(finding.finding_id for finding in findings)
    flow_trace_all = (
        normalized_report.flow_snapshot.snapshot_id,
        normalized_report.flow_snapshot.plan_flow_hash,
        normalized_report.flow_snapshot.schedule_flow_hash,
        normalized_report.flow_snapshot.pipeline_flow_hash,
        normalized_report.flow_snapshot.lineage_flow_hash,
    )
    ordered_flow_trace = flow_trace_all if normalized_mode != "flow" else flow_trace_all[1:]

    traversal_payload = {
        "audit_id": normalized_report.audit_id,
        "audit_hash": normalized_report.audit_hash,
        "traversal_mode": normalized_mode,
        "ordered_findings_trace": list(ordered_findings_trace),
        "ordered_flow_trace": list(ordered_flow_trace),
    }
    traversal_hash = _sha256_hex(_canonical_bytes(traversal_payload))
    receipt_id = f"receipt::{_sha256_hex(_canonical_bytes({'audit_id': normalized_report.audit_id, 'traversal_mode': normalized_mode, 'traversal_hash': traversal_hash}))[:16]}"
    return ResearchAuditReceipt(
        receipt_id=receipt_id,
        audit_id=normalized_report.audit_id,
        audit_hash=normalized_report.audit_hash,
        traversal_mode=normalized_mode,
        ordered_findings_trace=ordered_findings_trace,
        ordered_flow_trace=ordered_flow_trace,
        traversal_hash=traversal_hash,
    )
