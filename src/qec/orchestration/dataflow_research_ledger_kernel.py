"""v137.17.5 — Dataflow Research Ledger Kernel.

Deterministic replay-safe dataflow ledger construction, validation, continuity
summary, and bounded traversal over orchestration artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from qec.orchestration.autonomous_research_orchestration_kernel import AutonomousResearchPlan
from qec.orchestration.deterministic_experiment_scheduling_kernel import DeterministicExperimentSchedule
from qec.orchestration.replay_safe_benchmark_pipeline_kernel import ReplaySafeBenchmarkPipeline
from qec.orchestration.research_trace_lineage_kernel import ResearchTraceLineage
from qec.orchestration.deterministic_research_audit_kernel import ResearchAuditReport


CANONICAL_DATAFLOW_STAGES: Tuple[str, ...] = (
    "plan",
    "schedule",
    "pipeline",
    "lineage",
    "audit",
)

VALID_DATAFLOW_TRAVERSAL_MODES: Tuple[str, ...] = (
    "full",
    "continuity",
    "critical",
    "receipt",
)


@dataclass(frozen=True)
class DataflowLedgerNode:
    stage_name: str
    stage_ordinal: int
    artifact_id: str
    artifact_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "stage_ordinal": self.stage_ordinal,
            "artifact_id": self.artifact_id,
            "artifact_hash": self.artifact_hash,
        }


@dataclass(frozen=True)
class DataflowLedgerEdge:
    edge_id: str
    source_stage: str
    target_stage: str
    source_stage_ordinal: int
    target_stage_ordinal: int
    upstream_hash: str
    downstream_upstream_hash: str
    continuity_ok: bool
    continuity_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "source_stage": self.source_stage,
            "target_stage": self.target_stage,
            "source_stage_ordinal": self.source_stage_ordinal,
            "target_stage_ordinal": self.target_stage_ordinal,
            "upstream_hash": self.upstream_hash,
            "downstream_upstream_hash": self.downstream_upstream_hash,
            "continuity_ok": self.continuity_ok,
            "continuity_reason": self.continuity_reason,
        }


@dataclass(frozen=True)
class DataflowLedgerEntry:
    stage_name: str
    stage_ordinal: int
    artifact_in_id: str
    artifact_in_hash: str
    artifact_out_id: str
    artifact_out_hash: str
    predecessor_stage: str
    upstream_hash_link: str
    continuity_ok: bool
    validation_flags: Tuple[str, ...]
    traversal_index: int
    continuity_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "stage_ordinal": self.stage_ordinal,
            "artifact_in_id": self.artifact_in_id,
            "artifact_in_hash": self.artifact_in_hash,
            "artifact_out_id": self.artifact_out_id,
            "artifact_out_hash": self.artifact_out_hash,
            "predecessor_stage": self.predecessor_stage,
            "upstream_hash_link": self.upstream_hash_link,
            "continuity_ok": self.continuity_ok,
            "validation_flags": list(self.validation_flags),
            "traversal_index": self.traversal_index,
            "continuity_reason": self.continuity_reason,
        }


@dataclass(frozen=True)
class DataflowContinuitySummary:
    total_stages: int
    linked_stages: int
    broken_links: int
    terminal_stage: str
    continuity_ok: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_stages": self.total_stages,
            "linked_stages": self.linked_stages,
            "broken_links": self.broken_links,
            "terminal_stage": self.terminal_stage,
            "continuity_ok": self.continuity_ok,
        }


@dataclass(frozen=True)
class DataflowResearchLedger:
    ledger_id: str
    entries: Tuple[DataflowLedgerEntry, ...]
    nodes: Tuple[DataflowLedgerNode, ...]
    edges: Tuple[DataflowLedgerEdge, ...]
    continuity_summary: DataflowContinuitySummary
    ledger_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ledger_id": self.ledger_id,
            "entries": [entry.to_dict() for entry in self.entries],
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "continuity_summary": self.continuity_summary.to_dict(),
            "ledger_hash": self.ledger_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return self.ledger_hash


@dataclass(frozen=True)
class DataflowLedgerValidationReport:
    ledger_id: str
    is_valid: bool
    violations: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ledger_id": self.ledger_id,
            "is_valid": self.is_valid,
            "violations": list(self.violations),
        }


@dataclass(frozen=True)
class DataflowLedgerTraversalReceipt:
    receipt_id: str
    ledger_id: str
    ledger_hash: str
    traversal_mode: str
    ordered_stage_trace: Tuple[str, ...]
    ordered_edge_trace: Tuple[str, ...]
    traversal_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "ledger_id": self.ledger_id,
            "ledger_hash": self.ledger_hash,
            "traversal_mode": self.traversal_mode,
            "ordered_stage_trace": list(self.ordered_stage_trace),
            "ordered_edge_trace": list(self.ordered_edge_trace),
            "traversal_hash": self.traversal_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


LedgerLike = Union[DataflowResearchLedger, Mapping[str, Any]]


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


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


def _require_optional_hash(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    normalized = str(value).strip()
    if not normalized:
        return ""
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{field_name} must be empty or a 64-char lowercase SHA-256 hex")
    return normalized


def _require_valid_hash(value: Any, field_name: str) -> str:
    normalized = _require_non_empty_string(value, field_name)
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{field_name} must be a 64-char lowercase SHA-256 hex")
    return normalized


def _entry_sort_key(entry: DataflowLedgerEntry) -> Tuple[int, str, int]:
    canonical_index = CANONICAL_DATAFLOW_STAGES.index(entry.stage_name)
    return (entry.stage_ordinal, entry.stage_name, canonical_index)


def _edge_sort_key(edge: DataflowLedgerEdge) -> Tuple[int, int, str]:
    return (edge.source_stage_ordinal, edge.target_stage_ordinal, edge.edge_id)


def _artifact_id_hash_for_stage(stage_name: str, artifact: Any) -> Tuple[str, str]:
    if stage_name == "plan" and isinstance(artifact, AutonomousResearchPlan):
        return artifact.plan_id, artifact.plan_hash
    if stage_name == "schedule" and isinstance(artifact, DeterministicExperimentSchedule):
        return artifact.schedule_id, artifact.schedule_hash
    if stage_name == "pipeline" and isinstance(artifact, ReplaySafeBenchmarkPipeline):
        return artifact.pipeline_id, artifact.pipeline_hash
    if stage_name == "lineage" and isinstance(artifact, ResearchTraceLineage):
        return artifact.lineage_id, artifact.lineage_hash
    if stage_name == "audit" and isinstance(artifact, ResearchAuditReport):
        return artifact.audit_id, artifact.audit_hash
    if isinstance(artifact, Mapping):
        key_id = f"{stage_name}_id"
        key_hash = f"{stage_name}_hash"
        if stage_name == "plan":
            key_id, key_hash = "plan_id", "plan_hash"
        elif stage_name == "schedule":
            key_id, key_hash = "schedule_id", "schedule_hash"
        elif stage_name == "pipeline":
            key_id, key_hash = "pipeline_id", "pipeline_hash"
        elif stage_name == "lineage":
            key_id, key_hash = "lineage_id", "lineage_hash"
        elif stage_name == "audit":
            key_id, key_hash = "audit_id", "audit_hash"
        return (
            _require_non_empty_string(artifact.get(key_id, ""), key_id),
            _require_valid_hash(artifact.get(key_hash, ""), key_hash),
        )
    raise ValueError(f"unsupported stage artifact for {stage_name}")


def _normalize_entry(raw: Any) -> DataflowLedgerEntry:
    if isinstance(raw, DataflowLedgerEntry):
        entry = raw
    elif isinstance(raw, Mapping):
        flags = tuple(sorted(str(v).strip() for v in raw.get("validation_flags", ()) if str(v).strip()))
        entry = DataflowLedgerEntry(
            stage_name=_require_non_empty_string(raw.get("stage_name", ""), "stage_name"),
            stage_ordinal=_require_non_negative_int(raw.get("stage_ordinal", 0), "stage_ordinal"),
            artifact_in_id=str(raw.get("artifact_in_id", "")).strip(),
            artifact_in_hash=_require_optional_hash(raw.get("artifact_in_hash", ""), "artifact_in_hash"),
            artifact_out_id=_require_non_empty_string(raw.get("artifact_out_id", ""), "artifact_out_id"),
            artifact_out_hash=_require_valid_hash(raw.get("artifact_out_hash", ""), "artifact_out_hash"),
            predecessor_stage=str(raw.get("predecessor_stage", "")).strip(),
            upstream_hash_link=_require_optional_hash(raw.get("upstream_hash_link", ""), "upstream_hash_link"),
            continuity_ok=bool(raw.get("continuity_ok", False)),
            validation_flags=flags,
            traversal_index=_require_non_negative_int(raw.get("traversal_index", 0), "traversal_index"),
            continuity_reason=str(raw.get("continuity_reason", "")).strip(),
        )
    else:
        raise ValueError("entry must be mapping or DataflowLedgerEntry")

    if entry.stage_name not in CANONICAL_DATAFLOW_STAGES:
        raise ValueError(f"invalid stage_name: {entry.stage_name}")
    return entry


def _normalize_ledger(ledger: LedgerLike) -> DataflowResearchLedger:
    if isinstance(ledger, DataflowResearchLedger):
        entries: Tuple[DataflowLedgerEntry, ...] = ledger.entries
        ledger_id = ledger.ledger_id
        provided_hash = ledger.ledger_hash
    elif isinstance(ledger, Mapping):
        entries = tuple(_normalize_entry(raw) for raw in ledger.get("entries", ()))
        ledger_id = _require_non_empty_string(ledger.get("ledger_id", ""), "ledger_id")
        provided_hash = str(ledger.get("ledger_hash", "")).strip()
    else:
        raise ValueError("ledger must be mapping or DataflowResearchLedger")

    normalized_entries = tuple(sorted(entries, key=_entry_sort_key))
    rebuilt = _build_ledger_from_entries(ledger_id=ledger_id, entries=normalized_entries)
    if provided_hash and rebuilt.ledger_hash != provided_hash:
        raise ValueError("ledger_hash mismatch")
    return rebuilt


def _compute_ledger_hash(ledger_id: str, entries: Tuple[DataflowLedgerEntry, ...], nodes: Tuple[DataflowLedgerNode, ...], edges: Tuple[DataflowLedgerEdge, ...], continuity_summary: DataflowContinuitySummary) -> str:
    payload = {
        "ledger_id": ledger_id,
        "entries": [entry.to_dict() for entry in entries],
        "nodes": [node.to_dict() for node in nodes],
        "edges": [edge.to_dict() for edge in edges],
        "continuity_summary": continuity_summary.to_dict(),
    }
    return _sha256_hex(_canonical_json(payload).encode("utf-8"))


def _build_ledger_from_entries(ledger_id: str, entries: Tuple[DataflowLedgerEntry, ...]) -> DataflowResearchLedger:
    ordered_entries = tuple(sorted(entries, key=_entry_sort_key))
    nodes = tuple(
        DataflowLedgerNode(
            stage_name=entry.stage_name,
            stage_ordinal=entry.stage_ordinal,
            artifact_id=entry.artifact_out_id,
            artifact_hash=entry.artifact_out_hash,
        )
        for entry in ordered_entries
    )
    edges_list: List[DataflowLedgerEdge] = []
    for idx in range(1, len(ordered_entries)):
        prev_entry = ordered_entries[idx - 1]
        entry = ordered_entries[idx]
        edge_payload = {
            "source_stage": prev_entry.stage_name,
            "target_stage": entry.stage_name,
            "source_stage_ordinal": prev_entry.stage_ordinal,
            "target_stage_ordinal": entry.stage_ordinal,
            "upstream_hash": prev_entry.artifact_out_hash,
            "downstream_upstream_hash": entry.upstream_hash_link,
        }
        edges_list.append(
            DataflowLedgerEdge(
                edge_id=f"edge::{_sha256_hex(_canonical_json(edge_payload).encode('utf-8'))[:16]}",
                source_stage=prev_entry.stage_name,
                target_stage=entry.stage_name,
                source_stage_ordinal=prev_entry.stage_ordinal,
                target_stage_ordinal=entry.stage_ordinal,
                upstream_hash=prev_entry.artifact_out_hash,
                downstream_upstream_hash=entry.upstream_hash_link,
                continuity_ok=entry.continuity_ok and entry.upstream_hash_link == prev_entry.artifact_out_hash,
                continuity_reason=entry.continuity_reason,
            )
        )

    edges = tuple(sorted(edges_list, key=_edge_sort_key))
    continuity_summary = compute_dataflow_continuity(ordered_entries)
    ledger_hash = _compute_ledger_hash(ledger_id, ordered_entries, nodes, edges, continuity_summary)
    return DataflowResearchLedger(
        ledger_id=ledger_id,
        entries=ordered_entries,
        nodes=nodes,
        edges=edges,
        continuity_summary=continuity_summary,
        ledger_hash=ledger_hash,
    )


def build_dataflow_research_ledger(
    ledger_id: str,
    plan: Union[AutonomousResearchPlan, Mapping[str, Any]],
    schedule: Union[DeterministicExperimentSchedule, Mapping[str, Any]],
    pipeline: Union[ReplaySafeBenchmarkPipeline, Mapping[str, Any]],
    lineage: Union[ResearchTraceLineage, Mapping[str, Any]],
    audit: Union[ResearchAuditReport, Mapping[str, Any]],
) -> DataflowResearchLedger:
    normalized_ledger_id = _require_non_empty_string(ledger_id, "ledger_id")
    stage_artifacts = {
        "plan": plan,
        "schedule": schedule,
        "pipeline": pipeline,
        "lineage": lineage,
        "audit": audit,
    }

    entries: List[DataflowLedgerEntry] = []
    prev_stage = ""
    prev_id = ""
    prev_hash = ""
    for ordinal, stage_name in enumerate(CANONICAL_DATAFLOW_STAGES):
        artifact_id, artifact_hash = _artifact_id_hash_for_stage(stage_name, stage_artifacts[stage_name])
        continuity_ok = (ordinal == 0) or bool(prev_hash)
        reason = "root" if ordinal == 0 else "linked" if prev_hash else "missing_upstream"
        validation_flags: Tuple[str, ...] = ("canonical_stage",)
        entry = DataflowLedgerEntry(
            stage_name=stage_name,
            stage_ordinal=ordinal,
            artifact_in_id=prev_id,
            artifact_in_hash=prev_hash,
            artifact_out_id=artifact_id,
            artifact_out_hash=artifact_hash,
            predecessor_stage=prev_stage,
            upstream_hash_link=prev_hash,
            continuity_ok=continuity_ok,
            validation_flags=validation_flags,
            traversal_index=ordinal,
            continuity_reason=reason,
        )
        entries.append(entry)
        prev_stage = stage_name
        prev_id = artifact_id
        prev_hash = artifact_hash

    ledger = _build_ledger_from_entries(normalized_ledger_id, tuple(entries))
    return ledger


def compute_dataflow_continuity(entries: Sequence[DataflowLedgerEntry]) -> DataflowContinuitySummary:
    ordered_entries = tuple(sorted(entries, key=_entry_sort_key))
    linked_stages = max(len(ordered_entries) - 1, 0)
    broken_links = sum(
        1 for entry in ordered_entries[1:] if not (entry.upstream_hash_link and entry.continuity_ok)
    )
    terminal_stage = ordered_entries[-1].stage_name if ordered_entries else ""
    return DataflowContinuitySummary(
        total_stages=len(ordered_entries),
        linked_stages=linked_stages,
        broken_links=broken_links,
        terminal_stage=terminal_stage,
        continuity_ok=broken_links == 0,
    )


def normalize_dataflow_research_ledger(ledger: LedgerLike) -> DataflowResearchLedger:
    """Return canonical normalized form of a ledger, rebuilt entirely from entries."""
    return _normalize_ledger(ledger)


def validate_dataflow_research_ledger(ledger: LedgerLike) -> DataflowLedgerValidationReport:
    normalized = _normalize_ledger(ledger)

    violations: List[str] = []
    entries = normalized.entries
    if not entries:
        violations.append("empty_ledger_entries")
        return DataflowLedgerValidationReport(
            ledger_id=normalized.ledger_id,
            is_valid=False,
            violations=tuple(violations),
        )

    expected_ordinals = tuple(range(len(entries)))
    ordinals = tuple(entry.stage_ordinal for entry in entries)
    if len(set(ordinals)) != len(ordinals):
        violations.append("duplicate_stage_ordinals")

    if not violations and ordinals != expected_ordinals:
        violations.append("impossible_stage_regression")

    expected_stage_names = CANONICAL_DATAFLOW_STAGES[: len(entries)]
    stage_names = tuple(entry.stage_name for entry in entries)
    if not violations and stage_names != expected_stage_names:
        violations.append("unstable_or_unsorted_traversal_inputs")

    traversal_indices = tuple(entry.traversal_index for entry in entries)
    if not violations and traversal_indices != expected_ordinals:
        violations.append("unstable_or_unsorted_traversal_inputs")

    if not violations:
        for idx in range(1, len(entries)):
            prev_entry = entries[idx - 1]
            entry = entries[idx]
            if entry.continuity_ok and not entry.upstream_hash_link:
                violations.append("missing_required_upstream_link")
                break
            if entry.upstream_hash_link != prev_entry.artifact_out_hash:
                violations.append("hash_drift_between_linked_stages")
                break
            if entry.predecessor_stage != prev_entry.stage_name:
                violations.append("malformed_receipt_chain_structure")
                break
            if entry.artifact_in_id != prev_entry.artifact_out_id:
                violations.append("artifact_identity_mismatch_between_linked_stages")
                break
            # artifact_in_hash is the per-entry receipt of what entered this stage;
            # upstream_hash_link is the chaining mechanism.  Both are set from the
            # same predecessor hash in the builder, but may be tampered independently,
            # so we validate them as separate fields.
            if entry.artifact_in_hash != prev_entry.artifact_out_hash:
                violations.append("artifact_hash_mismatch_between_linked_stages")
                break

    expected_summary = compute_dataflow_continuity(entries)
    if expected_summary.to_dict() != normalized.continuity_summary.to_dict():
        violations.append("continuity_summary_mismatch")

    expected_hash = _compute_ledger_hash(
        normalized.ledger_id,
        normalized.entries,
        normalized.nodes,
        normalized.edges,
        normalized.continuity_summary,
    )
    if expected_hash != normalized.ledger_hash:
        violations.append("ledger_hash_mismatch")

    return DataflowLedgerValidationReport(
        ledger_id=normalized.ledger_id,
        is_valid=not violations,
        violations=tuple(violations),
    )


def traverse_dataflow_research_ledger(
    ledger: LedgerLike,
    traversal_mode: str,
) -> DataflowLedgerTraversalReceipt:
    normalized = _normalize_ledger(ledger)
    mode = _require_non_empty_string(traversal_mode, "traversal_mode")
    if mode not in VALID_DATAFLOW_TRAVERSAL_MODES:
        raise ValueError(f"invalid traversal mode: {mode}")

    entries = tuple(sorted(normalized.entries, key=_entry_sort_key))
    edges = tuple(sorted(normalized.edges, key=_edge_sort_key))

    if mode == "full":
        selected_entries = entries
    elif mode == "continuity":
        selected_entries = tuple(entry for entry in entries if entry.stage_ordinal > 0)
    elif mode == "critical":
        selected_entries = tuple(entry for entry in entries if (not entry.continuity_ok) or entry.continuity_reason != "linked")
    else:  # receipt
        selected_entries = entries

    selected_stage_trace = tuple(
        f"{entry.stage_ordinal}:{entry.stage_name}:{entry.artifact_out_id}" for entry in selected_entries
    )
    selected_stage_set = {entry.stage_name for entry in selected_entries}
    selected_edge_trace = tuple(
        f"{edge.source_stage_ordinal}->{edge.target_stage_ordinal}:{edge.edge_id}"
        for edge in edges
        if edge.source_stage in selected_stage_set and edge.target_stage in selected_stage_set
    )
    payload = {
        "ledger_id": normalized.ledger_id,
        "ledger_hash": normalized.ledger_hash,
        "traversal_mode": mode,
        "ordered_stage_trace": list(selected_stage_trace),
        "ordered_edge_trace": list(selected_edge_trace),
    }
    traversal_hash = _sha256_hex(_canonical_json(payload).encode("utf-8"))
    receipt_id = f"receipt::{traversal_hash[:16]}"
    return DataflowLedgerTraversalReceipt(
        receipt_id=receipt_id,
        ledger_id=normalized.ledger_id,
        ledger_hash=normalized.ledger_hash,
        traversal_mode=mode,
        ordered_stage_trace=selected_stage_trace,
        ordered_edge_trace=selected_edge_trace,
        traversal_hash=traversal_hash,
    )
