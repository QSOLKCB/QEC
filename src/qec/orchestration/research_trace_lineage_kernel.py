"""v137.17.3 — Research Trace Lineage Kernel.

Deterministic bounded lineage construction, validation, and traversal over
orchestration plans, schedules, benchmark pipeline stages/results, and audit /
verification artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from qec.orchestration.autonomous_research_orchestration_kernel import AutonomousResearchPlan
from qec.orchestration.deterministic_experiment_scheduling_kernel import DeterministicExperimentSchedule
from qec.orchestration.replay_safe_benchmark_pipeline_kernel import ReplaySafeBenchmarkPipeline


VALID_RESEARCH_TRACE_KINDS: Tuple[str, ...] = (
    "plan",
    "schedule",
    "stage",
    "result",
    "audit",
    "verification",
    "artifact",
)

VALID_RESEARCH_TRACE_RELATIONS: Tuple[str, ...] = (
    "derives_from",
    "scheduled_by",
    "produced_by",
    "verified_by",
    "audited_by",
    "replays",
)

VALID_RESEARCH_TRACE_TRAVERSAL_MODES: Tuple[str, ...] = (
    "lineage",
    "verification",
    "audit",
    "replay",
)

ResearchTraceNodeLike = Union["ResearchTraceNode", Mapping[str, Any]]
ResearchTraceEdgeLike = Union["ResearchTraceEdge", Mapping[str, Any]]
LineageLike = Union["ResearchTraceLineage", Mapping[str, Any]]


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
    if len(normalized) != 64:
        raise ValueError(f"{field_name} must be a 64-char SHA-256 hex")
    if any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{field_name} must be lowercase hex")
    return normalized


def _require_valid_source_ref(value: Any, field_name: str) -> str:
    normalized = _require_non_empty_string(value, field_name)
    if any(ch.isspace() for ch in normalized) or ":" not in normalized:
        raise ValueError(f"{field_name} must be a colon-qualified reference")
    return normalized


def _require_valid_edge_weight(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative finite float")
    try:
        weight = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_name} must be a non-negative finite float")
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"{field_name} must be non-negative and finite")
    return weight


@dataclass(frozen=True)
class ResearchTraceNode:
    trace_id: str
    trace_kind: str
    source_ref: str
    trace_hash: str
    trace_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "trace_kind": self.trace_kind,
            "source_ref": self.source_ref,
            "trace_hash": self.trace_hash,
            "trace_epoch": self.trace_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ResearchTraceEdge:
    edge_id: str
    source_trace_id: str
    target_trace_id: str
    relation_kind: str
    edge_weight: float
    trace_epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_trace_id": self.source_trace_id,
            "target_trace_id": self.target_trace_id,
            "relation_kind": self.relation_kind,
            "edge_weight": self.edge_weight,
            "trace_epoch": self.trace_epoch,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ResearchTraceLineage:
    lineage_id: str
    nodes: Tuple[ResearchTraceNode, ...]
    edges: Tuple[ResearchTraceEdge, ...]
    lineage_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lineage_id": self.lineage_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "lineage_hash": self.lineage_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ResearchTraceValidationReport:
    lineage_id: str
    is_valid: bool
    uniqueness_ok: bool
    node_validity_ok: bool
    edge_validity_ok: bool
    hash_validity_ok: bool
    reference_validity_ok: bool
    lineage_continuity_ok: bool
    weight_validity_ok: bool
    violations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lineage_id": self.lineage_id,
            "is_valid": self.is_valid,
            "uniqueness_ok": self.uniqueness_ok,
            "node_validity_ok": self.node_validity_ok,
            "edge_validity_ok": self.edge_validity_ok,
            "hash_validity_ok": self.hash_validity_ok,
            "reference_validity_ok": self.reference_validity_ok,
            "lineage_continuity_ok": self.lineage_continuity_ok,
            "weight_validity_ok": self.weight_validity_ok,
            "violations": list(self.violations),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


@dataclass(frozen=True)
class ResearchTraceExecutionReceipt:
    receipt_id: str
    lineage_id: str
    lineage_hash: str
    traversal_mode: str
    ordered_node_trace: Tuple[str, ...]
    ordered_edge_trace: Tuple[str, ...]
    traversal_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "lineage_id": self.lineage_id,
            "lineage_hash": self.lineage_hash,
            "traversal_mode": self.traversal_mode,
            "ordered_node_trace": list(self.ordered_node_trace),
            "ordered_edge_trace": list(self.ordered_edge_trace),
            "traversal_hash": self.traversal_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def as_hash_payload(self) -> bytes:
        return self.to_canonical_bytes()


def _node_sort_key(node: ResearchTraceNode) -> Tuple[int, str, str]:
    return (node.trace_epoch, node.trace_kind, node.trace_id)


def _edge_sort_key(edge: ResearchTraceEdge) -> Tuple[int, str, str]:
    return (edge.trace_epoch, edge.relation_kind, edge.edge_id)


def _normalize_node(raw: ResearchTraceNodeLike) -> ResearchTraceNode:
    if isinstance(raw, ResearchTraceNode):
        node = ResearchTraceNode(
            trace_id=_require_non_empty_string(raw.trace_id, "trace_id"),
            trace_kind=_require_non_empty_string(raw.trace_kind, "trace_kind"),
            source_ref=_require_valid_source_ref(raw.source_ref, "source_ref"),
            trace_hash=_require_valid_hash(raw.trace_hash, "trace_hash"),
            trace_epoch=_require_non_negative_int(raw.trace_epoch, "trace_epoch"),
        )
    else:
        if not isinstance(raw, Mapping):
            raise ValueError("node must be mapping or ResearchTraceNode")
        node = ResearchTraceNode(
            trace_id=_require_non_empty_string(raw.get("trace_id", ""), "trace_id"),
            trace_kind=_require_non_empty_string(raw.get("trace_kind", ""), "trace_kind"),
            source_ref=_require_valid_source_ref(raw.get("source_ref", ""), "source_ref"),
            trace_hash=_require_valid_hash(raw.get("trace_hash", ""), "trace_hash"),
            trace_epoch=_require_non_negative_int(raw.get("trace_epoch", 0), "trace_epoch"),
        )
    if node.trace_kind not in VALID_RESEARCH_TRACE_KINDS:
        raise ValueError(f"invalid trace kind: {node.trace_kind}")
    return node


def _normalize_edge(raw: ResearchTraceEdgeLike) -> ResearchTraceEdge:
    if isinstance(raw, ResearchTraceEdge):
        edge = ResearchTraceEdge(
            edge_id=_require_non_empty_string(raw.edge_id, "edge_id"),
            source_trace_id=_require_non_empty_string(raw.source_trace_id, "source_trace_id"),
            target_trace_id=_require_non_empty_string(raw.target_trace_id, "target_trace_id"),
            relation_kind=_require_non_empty_string(raw.relation_kind, "relation_kind"),
            edge_weight=_require_valid_edge_weight(raw.edge_weight, "edge_weight"),
            trace_epoch=_require_non_negative_int(raw.trace_epoch, "trace_epoch"),
        )
    else:
        if not isinstance(raw, Mapping):
            raise ValueError("edge must be mapping or ResearchTraceEdge")
        edge = ResearchTraceEdge(
            edge_id=_require_non_empty_string(raw.get("edge_id", ""), "edge_id"),
            source_trace_id=_require_non_empty_string(raw.get("source_trace_id", ""), "source_trace_id"),
            target_trace_id=_require_non_empty_string(raw.get("target_trace_id", ""), "target_trace_id"),
            relation_kind=_require_non_empty_string(raw.get("relation_kind", ""), "relation_kind"),
            edge_weight=_require_valid_edge_weight(raw.get("edge_weight", 0.0), "edge_weight"),
            trace_epoch=_require_non_negative_int(raw.get("trace_epoch", 0), "trace_epoch"),
        )
    if edge.relation_kind not in VALID_RESEARCH_TRACE_RELATIONS:
        raise ValueError(f"invalid relation kind: {edge.relation_kind}")
    return edge


def normalize_research_trace_input(
    nodes: Sequence[ResearchTraceNodeLike],
    edges: Sequence[ResearchTraceEdgeLike],
) -> Tuple[Tuple[ResearchTraceNode, ...], Tuple[ResearchTraceEdge, ...]]:
    normalized_nodes = tuple(_normalize_node(node) for node in nodes)
    normalized_edges = tuple(_normalize_edge(edge) for edge in edges)

    node_ids = tuple(node.trace_id for node in normalized_nodes)
    edge_ids = tuple(edge.edge_id for edge in normalized_edges)
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("duplicate trace IDs")
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("duplicate edge IDs")

    known_node_ids = set(node_ids)
    for edge in normalized_edges:
        if edge.source_trace_id not in known_node_ids or edge.target_trace_id not in known_node_ids:
            raise ValueError("invalid edge references")

    ordered_nodes = tuple(sorted(normalized_nodes, key=_node_sort_key))
    ordered_edges = tuple(sorted(normalized_edges, key=_edge_sort_key))
    return ordered_nodes, ordered_edges


def _compute_lineage_hash(lineage_id: str, nodes: Tuple[ResearchTraceNode, ...], edges: Tuple[ResearchTraceEdge, ...]) -> str:
    payload = {
        "lineage_id": lineage_id,
        "nodes": [node.to_dict() for node in nodes],
        "edges": [edge.to_dict() for edge in edges],
    }
    return _sha256_hex(_canonical_bytes(payload))


def _edge_id(relation_kind: str, source_trace_id: str, target_trace_id: str, index: int) -> str:
    return f"edge::{relation_kind}::{source_trace_id}::{target_trace_id}::{index}"


def build_research_trace_lineage(
    lineage_id: str,
    plan: AutonomousResearchPlan,
    schedule: DeterministicExperimentSchedule,
    pipeline: ReplaySafeBenchmarkPipeline,
    audit_refs: Sequence[str] = (),
    verification_refs: Sequence[str] = (),
    artifact_refs: Sequence[str] = (),
) -> ResearchTraceLineage:
    lineage_id = _require_non_empty_string(lineage_id, "lineage_id")

    nodes: List[ResearchTraceNode] = [
        ResearchTraceNode(
            trace_id=plan.plan_id,
            trace_kind="plan",
            source_ref=f"plan:{plan.plan_id}",
            trace_hash=plan.plan_hash,
            trace_epoch=0,
        ),
        ResearchTraceNode(
            trace_id=schedule.schedule_id,
            trace_kind="schedule",
            source_ref=f"schedule:{schedule.schedule_id}",
            trace_hash=schedule.schedule_hash,
            trace_epoch=0,
        ),
    ]

    for stage in pipeline.stages:
        nodes.append(
            ResearchTraceNode(
                trace_id=stage.stage_id,
                trace_kind="stage",
                source_ref=f"stage:{stage.input_ref}",
                trace_hash=_sha256_hex(stage.as_hash_payload()),
                trace_epoch=stage.stage_epoch,
            )
        )
    for result in pipeline.results:
        nodes.append(
            ResearchTraceNode(
                trace_id=result.result_id,
                trace_kind="result",
                source_ref=f"result:{result.experiment_id}",
                trace_hash=result.result_hash,
                trace_epoch=result.result_epoch,
            )
        )

    for idx, audit_ref in enumerate(sorted(set(_require_valid_source_ref(ref, "audit_ref") for ref in audit_refs))):
        nodes.append(
            ResearchTraceNode(
                trace_id=f"audit::{idx}",
                trace_kind="audit",
                source_ref=audit_ref,
                trace_hash=_sha256_hex(_canonical_bytes({"audit_ref": audit_ref, "i": idx})),
                trace_epoch=max(1, len(pipeline.stages)),
            )
        )
    for idx, verification_ref in enumerate(sorted(set(_require_valid_source_ref(ref, "verification_ref") for ref in verification_refs))):
        nodes.append(
            ResearchTraceNode(
                trace_id=f"verification::{idx}",
                trace_kind="verification",
                source_ref=verification_ref,
                trace_hash=_sha256_hex(_canonical_bytes({"verification_ref": verification_ref, "i": idx})),
                trace_epoch=max(1, len(pipeline.results)),
            )
        )
    for idx, artifact_ref in enumerate(sorted(set(_require_valid_source_ref(ref, "artifact_ref") for ref in artifact_refs))):
        nodes.append(
            ResearchTraceNode(
                trace_id=f"artifact::{idx}",
                trace_kind="artifact",
                source_ref=artifact_ref,
                trace_hash=_sha256_hex(_canonical_bytes({"artifact_ref": artifact_ref, "i": idx})),
                trace_epoch=max(1, len(pipeline.results)),
            )
        )

    edges: List[ResearchTraceEdge] = []
    edges.append(
        ResearchTraceEdge(
            edge_id=_edge_id("derives_from", schedule.schedule_id, plan.plan_id, 0),
            source_trace_id=schedule.schedule_id,
            target_trace_id=plan.plan_id,
            relation_kind="derives_from",
            edge_weight=1.0,
            trace_epoch=0,
        )
    )

    for idx, stage in enumerate(pipeline.stages):
        edges.append(
            ResearchTraceEdge(
                edge_id=_edge_id("scheduled_by", stage.stage_id, schedule.schedule_id, idx),
                source_trace_id=stage.stage_id,
                target_trace_id=schedule.schedule_id,
                relation_kind="scheduled_by",
                edge_weight=1.0,
                trace_epoch=stage.stage_epoch,
            )
        )

    stage_by_id = {stage.stage_id: stage for stage in pipeline.stages}
    for idx, result in enumerate(pipeline.results):
        stage = stage_by_id[result.stage_id]
        edges.append(
            ResearchTraceEdge(
                edge_id=_edge_id("produced_by", result.result_id, stage.stage_id, idx),
                source_trace_id=result.result_id,
                target_trace_id=stage.stage_id,
                relation_kind="produced_by",
                edge_weight=1.0,
                trace_epoch=result.result_epoch,
            )
        )

    audit_nodes = [node for node in nodes if node.trace_kind == "audit"]
    verification_nodes = [node for node in nodes if node.trace_kind == "verification"]
    artifact_nodes = [node for node in nodes if node.trace_kind == "artifact"]
    result_nodes = [node for node in nodes if node.trace_kind == "result"]

    for idx, audit_node in enumerate(audit_nodes):
        target = result_nodes[idx % len(result_nodes)] if result_nodes else nodes[0]
        edges.append(
            ResearchTraceEdge(
                edge_id=_edge_id("audited_by", target.trace_id, audit_node.trace_id, idx),
                source_trace_id=target.trace_id,
                target_trace_id=audit_node.trace_id,
                relation_kind="audited_by",
                edge_weight=1.0,
                trace_epoch=max(target.trace_epoch, audit_node.trace_epoch),
            )
        )

    for idx, verification_node in enumerate(verification_nodes):
        target = result_nodes[idx % len(result_nodes)] if result_nodes else nodes[0]
        edges.append(
            ResearchTraceEdge(
                edge_id=_edge_id("verified_by", target.trace_id, verification_node.trace_id, idx),
                source_trace_id=target.trace_id,
                target_trace_id=verification_node.trace_id,
                relation_kind="verified_by",
                edge_weight=1.0,
                trace_epoch=max(target.trace_epoch, verification_node.trace_epoch),
            )
        )

    for idx, artifact_node in enumerate(artifact_nodes):
        target = result_nodes[idx % len(result_nodes)] if result_nodes else nodes[0]
        edges.append(
            ResearchTraceEdge(
                edge_id=_edge_id("replays", artifact_node.trace_id, target.trace_id, idx),
                source_trace_id=artifact_node.trace_id,
                target_trace_id=target.trace_id,
                relation_kind="replays",
                edge_weight=1.0,
                trace_epoch=max(target.trace_epoch, artifact_node.trace_epoch),
            )
        )

    ordered_nodes, ordered_edges = normalize_research_trace_input(nodes, edges)
    lineage_hash = _compute_lineage_hash(lineage_id, ordered_nodes, ordered_edges)
    return ResearchTraceLineage(lineage_id=lineage_id, nodes=ordered_nodes, edges=ordered_edges, lineage_hash=lineage_hash)


def validate_research_trace_lineage(lineage: LineageLike) -> ResearchTraceValidationReport:
    if isinstance(lineage, ResearchTraceLineage):
        candidate = lineage
    else:
        if not isinstance(lineage, Mapping):
            raise ValueError("lineage must be mapping or ResearchTraceLineage")
        candidate = ResearchTraceLineage(
            lineage_id=_require_non_empty_string(lineage.get("lineage_id", ""), "lineage_id"),
            nodes=tuple(_normalize_node(node) for node in lineage.get("nodes", ())),
            edges=tuple(_normalize_edge(edge) for edge in lineage.get("edges", ())),
            lineage_hash=_require_valid_hash(lineage.get("lineage_hash", ""), "lineage_hash"),
        )

    violations: List[str] = []
    uniqueness_ok = True
    node_validity_ok = True
    edge_validity_ok = True
    hash_validity_ok = True
    reference_validity_ok = True
    lineage_continuity_ok = True
    weight_validity_ok = True

    node_ids = tuple(node.trace_id for node in candidate.nodes)
    edge_ids = tuple(edge.edge_id for edge in candidate.edges)
    if len(node_ids) != len(set(node_ids)):
        uniqueness_ok = False
        violations.append("duplicate_trace_ids")
    if len(edge_ids) != len(set(edge_ids)):
        uniqueness_ok = False
        violations.append("duplicate_edge_ids")

    if tuple(sorted(candidate.nodes, key=_node_sort_key)) != candidate.nodes:
        node_validity_ok = False
        violations.append("node_ordering_invalid")

    if tuple(sorted(candidate.edges, key=_edge_sort_key)) != candidate.edges:
        edge_validity_ok = False
        violations.append("edge_ordering_invalid")

    known_nodes = set(node_ids)
    referenced_sources = set()
    referenced_targets = set()
    for edge in candidate.edges:
        if edge.source_trace_id not in known_nodes or edge.target_trace_id not in known_nodes:
            reference_validity_ok = False
            violations.append("invalid_edge_reference")
            break
        referenced_sources.add(edge.source_trace_id)
        referenced_targets.add(edge.target_trace_id)
        if edge.edge_weight < 0.0 or not math.isfinite(edge.edge_weight):
            weight_validity_ok = False
            violations.append("invalid_edge_weight")
            break

    if candidate.nodes:
        roots = [node.trace_id for node in candidate.nodes if node.trace_kind == "plan"]
        if not roots:
            lineage_continuity_ok = False
            violations.append("missing_plan_root")
        reachable = referenced_sources | referenced_targets
        if any(node.trace_id not in reachable and node.trace_kind != "plan" for node in candidate.nodes):
            lineage_continuity_ok = False
            violations.append("disconnected_lineage")

    expected_hash = _compute_lineage_hash(candidate.lineage_id, candidate.nodes, candidate.edges)
    if candidate.lineage_hash != expected_hash:
        hash_validity_ok = False
        violations.append("lineage_hash_mismatch")

    is_valid = all(
        (
            uniqueness_ok,
            node_validity_ok,
            edge_validity_ok,
            hash_validity_ok,
            reference_validity_ok,
            lineage_continuity_ok,
            weight_validity_ok,
        )
    )
    return ResearchTraceValidationReport(
        lineage_id=candidate.lineage_id,
        is_valid=is_valid,
        uniqueness_ok=uniqueness_ok,
        node_validity_ok=node_validity_ok,
        edge_validity_ok=edge_validity_ok,
        hash_validity_ok=hash_validity_ok,
        reference_validity_ok=reference_validity_ok,
        lineage_continuity_ok=lineage_continuity_ok,
        weight_validity_ok=weight_validity_ok,
        violations=tuple(violations),
    )


def traverse_research_trace_lineage(
    lineage: ResearchTraceLineage,
    traversal_mode: str,
) -> ResearchTraceExecutionReceipt:
    traversal_mode = _require_non_empty_string(traversal_mode, "traversal_mode")
    if traversal_mode not in VALID_RESEARCH_TRACE_TRAVERSAL_MODES:
        raise ValueError(f"unsupported traversal mode: {traversal_mode}")

    if traversal_mode == "lineage":
        nodes = lineage.nodes
        edges = lineage.edges
    elif traversal_mode == "verification":
        allowed_kinds = {"plan", "schedule", "stage", "result", "verification"}
        allowed_relations = {"derives_from", "scheduled_by", "produced_by", "verified_by", "replays"}
        nodes = tuple(node for node in lineage.nodes if node.trace_kind in allowed_kinds)
        node_ids = {node.trace_id for node in nodes}
        edges = tuple(edge for edge in lineage.edges if edge.relation_kind in allowed_relations and edge.source_trace_id in node_ids and edge.target_trace_id in node_ids)
    elif traversal_mode == "audit":
        allowed_kinds = {"plan", "schedule", "stage", "result", "audit"}
        allowed_relations = {"derives_from", "scheduled_by", "produced_by", "audited_by", "replays"}
        nodes = tuple(node for node in lineage.nodes if node.trace_kind in allowed_kinds)
        node_ids = {node.trace_id for node in nodes}
        edges = tuple(edge for edge in lineage.edges if edge.relation_kind in allowed_relations and edge.source_trace_id in node_ids and edge.target_trace_id in node_ids)
    else:
        allowed_kinds = {"plan", "schedule", "stage", "result", "audit", "verification", "artifact"}
        allowed_relations = {"derives_from", "scheduled_by", "produced_by", "verified_by", "audited_by", "replays"}
        nodes = tuple(node for node in lineage.nodes if node.trace_kind in allowed_kinds)
        node_ids = {node.trace_id for node in nodes}
        edges = tuple(edge for edge in lineage.edges if edge.relation_kind in allowed_relations and edge.source_trace_id in node_ids and edge.target_trace_id in node_ids)

    ordered_nodes = tuple(sorted(nodes, key=_node_sort_key))
    ordered_edges = tuple(sorted(edges, key=_edge_sort_key))

    ordered_node_trace = tuple(node.trace_id for node in ordered_nodes)
    ordered_edge_trace = tuple(edge.edge_id for edge in ordered_edges)
    receipt_id = f"receipt::{lineage.lineage_id}::{traversal_mode}"

    traversal_payload = {
        "receipt_id": receipt_id,
        "lineage_id": lineage.lineage_id,
        "lineage_hash": lineage.lineage_hash,
        "traversal_mode": traversal_mode,
        "ordered_node_trace": list(ordered_node_trace),
        "ordered_edge_trace": list(ordered_edge_trace),
    }
    traversal_hash = _sha256_hex(_canonical_bytes(traversal_payload))

    return ResearchTraceExecutionReceipt(
        receipt_id=receipt_id,
        lineage_id=lineage.lineage_id,
        lineage_hash=lineage.lineage_hash,
        traversal_mode=traversal_mode,
        ordered_node_trace=ordered_node_trace,
        ordered_edge_trace=ordered_edge_trace,
        traversal_hash=traversal_hash,
    )
