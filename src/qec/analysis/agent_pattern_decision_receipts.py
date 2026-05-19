from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from qec.analysis.agent_observation_trace_receipts import (
    AgentObservationTraceReceipt,
    validate_agent_observation_trace_receipt,
)
from qec.analysis.crawler_boundary_receipts import CrawlerBoundaryReceipt, validate_crawler_boundary_receipt
from qec.analysis.skill_library_manifest import SkillLibraryManifest, validate_skill_library_manifest
from qec.analysis.tool_dispatch_telemetry_receipts import (
    ToolDispatchTelemetryReceipt,
    validate_tool_dispatch_telemetry_receipt,
)

_SCHEMA_VERSION = "AGENT_PATTERN_DECISION_RECEIPT_V1"

_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 1024

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_PATTERN_IDENTITY_TYPES = {
    "DECLARED_REVIEW_PATTERN",
    "DECLARED_TOOL_ROUTING_PATTERN",
    "DECLARED_OBSERVATION_PATTERN",
    "DECLARED_AUDIT_PATTERN",
    "DECLARED_CONTEXT_PATTERN",
    "DECLARED_CUSTOM_PATTERN",
}
_ALLOWED_SELECTION_MODES = {
    "PATTERN_DECLARED_BEFORE_EXECUTION",
    "PATTERN_DECLARED_FOR_AUDIT_ONLY",
    "PATTERN_DECLARED_FOR_REPLAY_ONLY",
    "PATTERN_DECLARED_CONTEXT_ONLY",
    "DECLARED_CUSTOM_SELECTION",
}
_ALLOWED_DECISION_BOUNDARY_MODES = {
    "DECISION_BOUNDARY_AUDIT_ONLY",
    "DECISION_BOUNDARY_REPLAY_ONLY",
    "DECISION_BOUNDARY_CONTEXT_ONLY",
    "DECISION_BOUNDARY_DECLARED_ONLY",
    "DECLARED_CUSTOM_DECISION_BOUNDARY",
}
_ALLOWED_EXECUTION_BOUNDARY_MODES = {
    "PATTERN_NOT_EXECUTED",
    "PATTERN_REPLAY_ONLY",
    "PATTERN_AUDIT_ONLY",
    "PATTERN_CONTEXT_ONLY",
    "DECLARED_CUSTOM_EXECUTION_BOUNDARY",
}
_ALLOWED_AUDIT_BOUNDARY_MODES = {
    "STRICT_PATTERN_AUDIT",
    "REPLAY_PATTERN_AUDIT",
    "CONTEXT_PATTERN_AUDIT",
    "DECLARED_CUSTOM_PATTERN_AUDIT",
}

_REPLAY_SAFE_SELECTION_MODES = {
    "PATTERN_DECLARED_BEFORE_EXECUTION",
    "PATTERN_DECLARED_FOR_AUDIT_ONLY",
    "PATTERN_DECLARED_FOR_REPLAY_ONLY",
}
_REPLAY_SAFE_DECISION_BOUNDARY_MODES = {
    "DECISION_BOUNDARY_AUDIT_ONLY",
    "DECISION_BOUNDARY_REPLAY_ONLY",
    "DECISION_BOUNDARY_DECLARED_ONLY",
}
_REPLAY_SAFE_EXECUTION_BOUNDARY_MODES = {
    "PATTERN_NOT_EXECUTED",
    "PATTERN_REPLAY_ONLY",
    "PATTERN_AUDIT_ONLY",
}
_REPLAY_SAFE_AUDIT_BOUNDARY_MODES = {"STRICT_PATTERN_AUDIT", "REPLAY_PATTERN_AUDIT"}

_FORBIDDEN_TOKENS = (
    "autonomous planning",
    "autonomous reasoning",
    "pattern selected itself",
    "tool execution succeeded",
    "runtime dispatch",
    "live crawler",
    "agent output is evidence",
    "agent output as evidence",
    "automatic reasoning correctness",
    "semantic equivalence guaranteed",
    "autonomous evaluation",
    "hidden runtime execution",
    "hidden tool execution",
    "hidden tool call",
    "hidden crawler execution",
    "autonomous network crawling",
    "hidden network semantics",
    "hidden autonomous",
    "hidden replay equivalence",
    "hidden mutable pattern",
)


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


def _to_canonical_obj(value: Any) -> Any:
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return {k: _to_canonical_obj(v) for k, v in value.__dict__.items()}
    if isinstance(value, Mapping):
        return {k: _to_canonical_obj(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_canonical_obj(v) for v in value]
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_to_canonical_obj(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _base_payload(payload: Mapping[str, Any], hash_key: str) -> dict[str, Any]:
    out = dict(payload)
    out.pop(hash_key, None)
    return out


def _validate_hash_format(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character hex digest")


def _check_text(value: str, field_name: str, max_len: int) -> None:
    if not isinstance(value, str) or not value or len(value) > max_len:
        raise ValueError(f"{field_name} must be non-empty and bounded")


def _check_no_forbidden_runtime_semantics(payload: Any) -> None:
    canonical = payload if isinstance(payload, str) else _canonical_json({"payload": payload})
    lowered = canonical.lower().replace("_", " ").replace("-", " ")
    for token in _FORBIDDEN_TOKENS:
        if token.lower().replace("_", " ").replace("-", " ") in lowered:
            raise ValueError("forbidden runtime or hidden semantics")


def _validate_agent_pattern_decision_semantics(*texts: str) -> None:
    _check_no_forbidden_runtime_semantics("\n".join(texts))


def _revalidate_exact_instance(value: Any, cls: type[Any]) -> None:
    if type(value) is not cls:
        raise _invalid_input()
    cls(**value.__dict__)


@dataclass(frozen=True)
class AgentPatternIdentity:
    pattern_name: str
    pattern_version: str
    pattern_type: str
    agent_pattern_identity_hash: str

    def __post_init__(self) -> None:
        if type(self) is not AgentPatternIdentity:
            raise _invalid_input()
        _check_text(self.pattern_name, "pattern_name", _MAX_NAME_LENGTH)
        _check_text(self.pattern_version, "pattern_version", _MAX_NAME_LENGTH)
        if self.pattern_type not in _ALLOWED_PATTERN_IDENTITY_TYPES:
            raise ValueError("invalid pattern_type")
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.agent_pattern_identity_hash, "agent_pattern_identity_hash")
        if _hash_payload(_base_payload(self.__dict__, "agent_pattern_identity_hash")) != self.agent_pattern_identity_hash:
            raise ValueError("agent_pattern_identity_hash mismatch")


@dataclass(frozen=True)
class PatternSelectionDeclaration:
    selection_mode: str
    selection_reason: str
    pattern_selection_declaration_hash: str

    def __post_init__(self) -> None:
        if type(self) is not PatternSelectionDeclaration:
            raise _invalid_input()
        if self.selection_mode not in _ALLOWED_SELECTION_MODES:
            raise ValueError("invalid selection_mode")
        _check_text(self.selection_reason, "selection_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.pattern_selection_declaration_hash, "pattern_selection_declaration_hash")
        if _hash_payload(_base_payload(self.__dict__, "pattern_selection_declaration_hash")) != self.pattern_selection_declaration_hash:
            raise ValueError("pattern_selection_declaration_hash mismatch")


@dataclass(frozen=True)
class PatternDecisionBoundary:
    decision_boundary_mode: str
    decision_boundary_reason: str
    pattern_decision_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not PatternDecisionBoundary:
            raise _invalid_input()
        if self.decision_boundary_mode not in _ALLOWED_DECISION_BOUNDARY_MODES:
            raise ValueError("invalid decision_boundary_mode")
        _check_text(self.decision_boundary_reason, "decision_boundary_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.pattern_decision_boundary_hash, "pattern_decision_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "pattern_decision_boundary_hash")) != self.pattern_decision_boundary_hash:
            raise ValueError("pattern_decision_boundary_hash mismatch")


@dataclass(frozen=True)
class PatternExecutionBoundary:
    execution_boundary_mode: str
    execution_boundary_reason: str
    pattern_execution_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not PatternExecutionBoundary:
            raise _invalid_input()
        if self.execution_boundary_mode not in _ALLOWED_EXECUTION_BOUNDARY_MODES:
            raise ValueError("invalid execution_boundary_mode")
        _check_text(self.execution_boundary_reason, "execution_boundary_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.pattern_execution_boundary_hash, "pattern_execution_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "pattern_execution_boundary_hash")) != self.pattern_execution_boundary_hash:
            raise ValueError("pattern_execution_boundary_hash mismatch")


@dataclass(frozen=True)
class PatternAuditBoundary:
    audit_boundary_mode: str
    audit_boundary_reason: str
    pattern_audit_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not PatternAuditBoundary:
            raise _invalid_input()
        if self.audit_boundary_mode not in _ALLOWED_AUDIT_BOUNDARY_MODES:
            raise ValueError("invalid audit_boundary_mode")
        _check_text(self.audit_boundary_reason, "audit_boundary_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.pattern_audit_boundary_hash, "pattern_audit_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "pattern_audit_boundary_hash")) != self.pattern_audit_boundary_hash:
            raise ValueError("pattern_audit_boundary_hash mismatch")


@dataclass(frozen=True)
class AgentPatternDecisionReceipt:
    schema_version: str
    agent_observation_trace_receipt_hash: str
    skill_library_manifest_hash: str
    tool_dispatch_telemetry_receipt_hash: str
    crawler_boundary_receipt_hash: str
    pattern_identity: AgentPatternIdentity
    selection_declaration: PatternSelectionDeclaration
    decision_boundary: PatternDecisionBoundary
    execution_boundary: PatternExecutionBoundary
    audit_boundary: PatternAuditBoundary
    replay_safe_pattern_decision: bool
    adapter_only: bool
    agent_pattern_decision_receipt_hash: str


def build_agent_pattern_identity(pattern_name: str, pattern_version: str, pattern_type: str) -> AgentPatternIdentity:
    payload = {"pattern_name": pattern_name, "pattern_version": pattern_version, "pattern_type": pattern_type}
    return AgentPatternIdentity(**payload, agent_pattern_identity_hash=_hash_payload(payload))


def build_pattern_selection_declaration(selection_mode: str, selection_reason: str) -> PatternSelectionDeclaration:
    payload = {"selection_mode": selection_mode, "selection_reason": selection_reason}
    return PatternSelectionDeclaration(**payload, pattern_selection_declaration_hash=_hash_payload(payload))


def build_pattern_decision_boundary(decision_boundary_mode: str, decision_boundary_reason: str) -> PatternDecisionBoundary:
    payload = {"decision_boundary_mode": decision_boundary_mode, "decision_boundary_reason": decision_boundary_reason}
    return PatternDecisionBoundary(**payload, pattern_decision_boundary_hash=_hash_payload(payload))


def build_pattern_execution_boundary(execution_boundary_mode: str, execution_boundary_reason: str) -> PatternExecutionBoundary:
    payload = {"execution_boundary_mode": execution_boundary_mode, "execution_boundary_reason": execution_boundary_reason}
    return PatternExecutionBoundary(**payload, pattern_execution_boundary_hash=_hash_payload(payload))


def build_pattern_audit_boundary(audit_boundary_mode: str, audit_boundary_reason: str) -> PatternAuditBoundary:
    payload = {"audit_boundary_mode": audit_boundary_mode, "audit_boundary_reason": audit_boundary_reason}
    return PatternAuditBoundary(**payload, pattern_audit_boundary_hash=_hash_payload(payload))


def _recompute_replay_safe_pattern_decision(
    agent_observation_trace_receipt: AgentObservationTraceReceipt,
    skill_library_manifest: SkillLibraryManifest,
    tool_dispatch_telemetry_receipt: ToolDispatchTelemetryReceipt,
    crawler_boundary_receipt: CrawlerBoundaryReceipt,
    pattern_identity: AgentPatternIdentity,
    selection_declaration: PatternSelectionDeclaration,
    decision_boundary: PatternDecisionBoundary,
    execution_boundary: PatternExecutionBoundary,
    audit_boundary: PatternAuditBoundary,
    adapter_only: bool,
) -> bool:
    return (
        adapter_only is True
        and agent_observation_trace_receipt.replay_safe_observation_trace is True
        and skill_library_manifest.replay_safe_skill_library is True
        and tool_dispatch_telemetry_receipt.replay_safe_dispatch is True
        and crawler_boundary_receipt.replay_safe_crawler is True
        and pattern_identity.pattern_type == agent_observation_trace_receipt.pattern_declaration.pattern_type
        and pattern_identity.pattern_type != "DECLARED_CUSTOM_PATTERN"
        and selection_declaration.selection_mode in _REPLAY_SAFE_SELECTION_MODES
        and decision_boundary.decision_boundary_mode in _REPLAY_SAFE_DECISION_BOUNDARY_MODES
        and execution_boundary.execution_boundary_mode in _REPLAY_SAFE_EXECUTION_BOUNDARY_MODES
        and audit_boundary.audit_boundary_mode in _REPLAY_SAFE_AUDIT_BOUNDARY_MODES
    )


def build_agent_pattern_decision_receipt(
    agent_observation_trace_receipt: AgentObservationTraceReceipt,
    skill_library_manifest: SkillLibraryManifest,
    tool_dispatch_telemetry_receipt: ToolDispatchTelemetryReceipt,
    crawler_boundary_receipt: CrawlerBoundaryReceipt,
    pattern_identity: AgentPatternIdentity,
    selection_declaration: PatternSelectionDeclaration,
    decision_boundary: PatternDecisionBoundary,
    execution_boundary: PatternExecutionBoundary,
    audit_boundary: PatternAuditBoundary,
    adapter_only: bool,
) -> AgentPatternDecisionReceipt:
    replay_safe_pattern_decision = _recompute_replay_safe_pattern_decision(
        agent_observation_trace_receipt,
        skill_library_manifest,
        tool_dispatch_telemetry_receipt,
        crawler_boundary_receipt,
        pattern_identity,
        selection_declaration,
        decision_boundary,
        execution_boundary,
        audit_boundary,
        adapter_only,
    )
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "agent_observation_trace_receipt_hash": agent_observation_trace_receipt.agent_observation_trace_receipt_hash,
        "skill_library_manifest_hash": skill_library_manifest.skill_library_manifest_hash,
        "tool_dispatch_telemetry_receipt_hash": tool_dispatch_telemetry_receipt.tool_dispatch_telemetry_receipt_hash,
        "crawler_boundary_receipt_hash": crawler_boundary_receipt.crawler_boundary_receipt_hash,
        "pattern_identity": pattern_identity,
        "selection_declaration": selection_declaration,
        "decision_boundary": decision_boundary,
        "execution_boundary": execution_boundary,
        "audit_boundary": audit_boundary,
        "replay_safe_pattern_decision": replay_safe_pattern_decision,
        "adapter_only": adapter_only,
    }
    return AgentPatternDecisionReceipt(**payload, agent_pattern_decision_receipt_hash=_hash_payload(payload))


def validate_agent_pattern_identity(value: AgentPatternIdentity) -> AgentPatternIdentity:
    _revalidate_exact_instance(value, AgentPatternIdentity)
    return value


def validate_pattern_selection_declaration(value: PatternSelectionDeclaration) -> PatternSelectionDeclaration:
    _revalidate_exact_instance(value, PatternSelectionDeclaration)
    return value


def validate_pattern_decision_boundary(value: PatternDecisionBoundary) -> PatternDecisionBoundary:
    _revalidate_exact_instance(value, PatternDecisionBoundary)
    return value


def validate_pattern_execution_boundary(value: PatternExecutionBoundary) -> PatternExecutionBoundary:
    _revalidate_exact_instance(value, PatternExecutionBoundary)
    return value


def validate_pattern_audit_boundary(value: PatternAuditBoundary) -> PatternAuditBoundary:
    _revalidate_exact_instance(value, PatternAuditBoundary)
    return value


def validate_agent_pattern_decision_receipt(
    receipt: AgentPatternDecisionReceipt,
    agent_observation_trace_receipt: AgentObservationTraceReceipt,
    skill_library_manifest: SkillLibraryManifest,
    tool_dispatch_telemetry_receipt: ToolDispatchTelemetryReceipt,
    crawler_boundary_receipt: CrawlerBoundaryReceipt,
    **upstream_kwargs: Any,
) -> AgentPatternDecisionReceipt:
    _revalidate_exact_instance(receipt, AgentPatternDecisionReceipt)
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("invalid schema_version")
    if isinstance(receipt.adapter_only, bool) is False:
        raise ValueError("adapter_only must be bool")
    if isinstance(receipt.replay_safe_pattern_decision, bool) is False:
        raise ValueError("replay_safe_pattern_decision must be bool")

    validate_agent_pattern_identity(receipt.pattern_identity)
    validate_pattern_selection_declaration(receipt.selection_declaration)
    validate_pattern_decision_boundary(receipt.decision_boundary)
    validate_pattern_execution_boundary(receipt.execution_boundary)
    validate_pattern_audit_boundary(receipt.audit_boundary)

    validate_agent_observation_trace_receipt(agent_observation_trace_receipt, **upstream_kwargs)
    validate_skill_library_manifest(skill_library_manifest, agent_observation_trace_receipt=agent_observation_trace_receipt, **upstream_kwargs)
    validate_tool_dispatch_telemetry_receipt(
        tool_dispatch_telemetry_receipt,
        skill_library_manifest,
        agent_observation_trace_receipt=agent_observation_trace_receipt,
        **upstream_kwargs,
    )
    validate_crawler_boundary_receipt(
        crawler_boundary_receipt,
        tool_dispatch_telemetry_receipt,
        skill_library_manifest,
        agent_observation_trace_receipt=agent_observation_trace_receipt,
        **upstream_kwargs,
    )

    if receipt.agent_observation_trace_receipt_hash != agent_observation_trace_receipt.agent_observation_trace_receipt_hash:
        raise ValueError("agent_observation_trace_receipt_hash mismatch")
    if receipt.skill_library_manifest_hash != skill_library_manifest.skill_library_manifest_hash:
        raise ValueError("skill_library_manifest_hash mismatch")
    if receipt.tool_dispatch_telemetry_receipt_hash != tool_dispatch_telemetry_receipt.tool_dispatch_telemetry_receipt_hash:
        raise ValueError("tool_dispatch_telemetry_receipt_hash mismatch")
    if receipt.crawler_boundary_receipt_hash != crawler_boundary_receipt.crawler_boundary_receipt_hash:
        raise ValueError("crawler_boundary_receipt_hash mismatch")

    _validate_agent_pattern_decision_semantics(
        receipt.selection_declaration.selection_reason,
        receipt.decision_boundary.decision_boundary_reason,
        receipt.execution_boundary.execution_boundary_reason,
        receipt.audit_boundary.audit_boundary_reason,
    )
    _check_no_forbidden_runtime_semantics(receipt.__dict__)

    recomputed = _recompute_replay_safe_pattern_decision(
        agent_observation_trace_receipt,
        skill_library_manifest,
        tool_dispatch_telemetry_receipt,
        crawler_boundary_receipt,
        receipt.pattern_identity,
        receipt.selection_declaration,
        receipt.decision_boundary,
        receipt.execution_boundary,
        receipt.audit_boundary,
        receipt.adapter_only,
    )
    if receipt.replay_safe_pattern_decision != recomputed:
        raise ValueError("replay_safe_pattern_decision must be recomputed")

    _validate_hash_format(receipt.agent_pattern_decision_receipt_hash, "agent_pattern_decision_receipt_hash")
    if _hash_payload(_base_payload(receipt.__dict__, "agent_pattern_decision_receipt_hash")) != receipt.agent_pattern_decision_receipt_hash:
        raise ValueError("agent_pattern_decision_receipt_hash mismatch")
    return receipt
