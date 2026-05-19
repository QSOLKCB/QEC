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
from qec.analysis.skill_library_manifest import SkillLibraryManifest, validate_skill_library_manifest

_SCHEMA_VERSION = "TOOL_DISPATCH_TELEMETRY_RECEIPT_V1"

_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 1024

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_TOOL_TYPES = {
    "DECLARED_ANALYSIS_TOOL",
    "DECLARED_AUDIT_TOOL",
    "DECLARED_CONTEXT_TOOL",
    "DECLARED_REPLAY_TOOL",
    "DECLARED_OBSERVATION_TOOL",
    "DECLARED_CUSTOM_TOOL",
}
_ALLOWED_INPUT_MODES = {
    "HASH_BOUND_INPUT",
    "CONTEXT_ONLY_INPUT",
    "DECLARED_REPLAY_INPUT",
    "DECLARED_AUDIT_INPUT",
    "DECLARED_CUSTOM_INPUT",
}
_ALLOWED_OUTPUT_MODES = {
    "HASH_BOUND_OUTPUT",
    "CONTEXT_ONLY_OUTPUT",
    "DECLARED_REPLAY_OUTPUT",
    "DECLARED_AUDIT_OUTPUT",
    "DECLARED_CUSTOM_OUTPUT",
}
_ALLOWED_EXECUTION_MODES = {
    "TOOL_DECLARED_NOT_EXECUTED",
    "TOOL_DECLARED_REPLAY_ONLY",
    "TOOL_DECLARED_AUDIT_ONLY",
    "TOOL_DECLARED_CONTEXT_ONLY",
    "DECLARED_CUSTOM_EXECUTION",
}
_ALLOWED_AUDIT_MODES = {
    "STRICT_AUDIT_TRAIL",
    "DECLARED_REPLAY_AUDIT",
    "DECLARED_CONTEXT_AUDIT",
    "DECLARED_CUSTOM_AUDIT",
}

_REPLAY_SAFE_EXECUTION_MODES = {
    "TOOL_DECLARED_REPLAY_ONLY",
    "TOOL_DECLARED_AUDIT_ONLY",
    "TOOL_DECLARED_CONTEXT_ONLY",
}
_REPLAY_SAFE_INPUT_MODES = {"HASH_BOUND_INPUT", "DECLARED_REPLAY_INPUT", "DECLARED_AUDIT_INPUT"}
_REPLAY_SAFE_OUTPUT_MODES = {"HASH_BOUND_OUTPUT", "DECLARED_REPLAY_OUTPUT", "DECLARED_AUDIT_OUTPUT"}
_REPLAY_SAFE_AUDIT_MODES = {"STRICT_AUDIT_TRAIL", "DECLARED_REPLAY_AUDIT"}

_FORBIDDEN_TOKENS = (
    "tool execution succeeded",
    "runtime dispatch",
    "live crawler",
    "autonomous planning",
    "agent output is evidence",
    "automatic reasoning correctness",
    "semantic equivalence guaranteed",
    "benchmark superiority established",
    "autonomous evaluation",
    "hidden tool execution",
    "hidden network semantics",
    "hidden crawler semantics",
    "hidden autonomous",
    "hidden replay equivalence",
    "hidden mutable dispatch",
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


def _validate_tool_dispatch_semantics(*texts: str) -> None:
    _check_no_forbidden_runtime_semantics("\n".join(texts))


def _revalidate_exact_instance(value: Any, cls: type[Any]) -> None:
    if type(value) is not cls:
        raise _invalid_input()
    cls(**value.__dict__)


@dataclass(frozen=True)
class ToolDispatchIdentity:
    tool_name: str
    tool_version: str
    tool_type: str
    tool_dispatch_identity_hash: str

    def __post_init__(self) -> None:
        if type(self) is not ToolDispatchIdentity:
            raise _invalid_input()
        _check_text(self.tool_name, "tool_name", _MAX_NAME_LENGTH)
        _check_text(self.tool_version, "tool_version", _MAX_NAME_LENGTH)
        if self.tool_type not in _ALLOWED_TOOL_TYPES:
            raise ValueError("invalid tool_type")
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.tool_dispatch_identity_hash, "tool_dispatch_identity_hash")
        if _hash_payload(_base_payload(self.__dict__, "tool_dispatch_identity_hash")) != self.tool_dispatch_identity_hash:
            raise ValueError("tool_dispatch_identity_hash mismatch")


@dataclass(frozen=True)
class ToolDispatchInputBoundary:
    input_mode: str
    input_hash: str
    input_reason: str
    tool_dispatch_input_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not ToolDispatchInputBoundary:
            raise _invalid_input()
        if self.input_mode not in _ALLOWED_INPUT_MODES:
            raise ValueError("invalid input_mode")
        _validate_hash_format(self.input_hash, "input_hash")
        _check_text(self.input_reason, "input_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.tool_dispatch_input_boundary_hash, "tool_dispatch_input_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "tool_dispatch_input_boundary_hash")) != self.tool_dispatch_input_boundary_hash:
            raise ValueError("tool_dispatch_input_boundary_hash mismatch")


@dataclass(frozen=True)
class ToolDispatchOutputBoundary:
    output_mode: str
    output_hash: str
    output_reason: str
    tool_dispatch_output_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not ToolDispatchOutputBoundary:
            raise _invalid_input()
        if self.output_mode not in _ALLOWED_OUTPUT_MODES:
            raise ValueError("invalid output_mode")
        _validate_hash_format(self.output_hash, "output_hash")
        _check_text(self.output_reason, "output_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.tool_dispatch_output_boundary_hash, "tool_dispatch_output_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "tool_dispatch_output_boundary_hash")) != self.tool_dispatch_output_boundary_hash:
            raise ValueError("tool_dispatch_output_boundary_hash mismatch")


@dataclass(frozen=True)
class ToolDispatchExecutionBoundary:
    execution_mode: str
    declared_execution_time_ms: int
    execution_reason: str
    tool_dispatch_execution_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not ToolDispatchExecutionBoundary:
            raise _invalid_input()
        if self.execution_mode not in _ALLOWED_EXECUTION_MODES:
            raise ValueError("invalid execution_mode")
        if isinstance(self.declared_execution_time_ms, bool) or not isinstance(self.declared_execution_time_ms, int):
            raise ValueError("declared_execution_time_ms must be int")
        if self.declared_execution_time_ms < 0:
            raise ValueError("declared_execution_time_ms out of bounds")
        _check_text(self.execution_reason, "execution_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.tool_dispatch_execution_boundary_hash, "tool_dispatch_execution_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "tool_dispatch_execution_boundary_hash")) != self.tool_dispatch_execution_boundary_hash:
            raise ValueError("tool_dispatch_execution_boundary_hash mismatch")


@dataclass(frozen=True)
class ToolDispatchAuditBoundary:
    audit_mode: str
    audit_reason: str
    tool_dispatch_audit_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not ToolDispatchAuditBoundary:
            raise _invalid_input()
        if self.audit_mode not in _ALLOWED_AUDIT_MODES:
            raise ValueError("invalid audit_mode")
        _check_text(self.audit_reason, "audit_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.tool_dispatch_audit_boundary_hash, "tool_dispatch_audit_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "tool_dispatch_audit_boundary_hash")) != self.tool_dispatch_audit_boundary_hash:
            raise ValueError("tool_dispatch_audit_boundary_hash mismatch")


@dataclass(frozen=True)
class ToolDispatchTelemetryReceipt:
    schema_version: str
    skill_library_manifest_hash: str
    agent_observation_trace_receipt_hash: str
    dispatch_identity: ToolDispatchIdentity
    input_boundary: ToolDispatchInputBoundary
    output_boundary: ToolDispatchOutputBoundary
    execution_boundary: ToolDispatchExecutionBoundary
    audit_boundary: ToolDispatchAuditBoundary
    replay_safe_dispatch: bool
    adapter_only: bool
    tool_dispatch_telemetry_receipt_hash: str


def build_tool_dispatch_identity(tool_name: str, tool_version: str, tool_type: str) -> ToolDispatchIdentity:
    payload = {"tool_name": tool_name, "tool_version": tool_version, "tool_type": tool_type}
    return ToolDispatchIdentity(**payload, tool_dispatch_identity_hash=_hash_payload(payload))


def build_tool_dispatch_input_boundary(input_mode: str, input_hash: str, input_reason: str) -> ToolDispatchInputBoundary:
    payload = {"input_mode": input_mode, "input_hash": input_hash, "input_reason": input_reason}
    return ToolDispatchInputBoundary(**payload, tool_dispatch_input_boundary_hash=_hash_payload(payload))


def build_tool_dispatch_output_boundary(output_mode: str, output_hash: str, output_reason: str) -> ToolDispatchOutputBoundary:
    payload = {"output_mode": output_mode, "output_hash": output_hash, "output_reason": output_reason}
    return ToolDispatchOutputBoundary(**payload, tool_dispatch_output_boundary_hash=_hash_payload(payload))


def build_tool_dispatch_execution_boundary(execution_mode: str, declared_execution_time_ms: int, execution_reason: str) -> ToolDispatchExecutionBoundary:
    payload = {
        "execution_mode": execution_mode,
        "declared_execution_time_ms": declared_execution_time_ms,
        "execution_reason": execution_reason,
    }
    return ToolDispatchExecutionBoundary(**payload, tool_dispatch_execution_boundary_hash=_hash_payload(payload))


def build_tool_dispatch_audit_boundary(audit_mode: str, audit_reason: str) -> ToolDispatchAuditBoundary:
    payload = {"audit_mode": audit_mode, "audit_reason": audit_reason}
    return ToolDispatchAuditBoundary(**payload, tool_dispatch_audit_boundary_hash=_hash_payload(payload))


def _recompute_replay_safe_dispatch(
    skill_library_manifest: SkillLibraryManifest,
    agent_observation_trace_receipt: AgentObservationTraceReceipt,
    dispatch_identity: ToolDispatchIdentity,
    input_boundary: ToolDispatchInputBoundary,
    output_boundary: ToolDispatchOutputBoundary,
    execution_boundary: ToolDispatchExecutionBoundary,
    audit_boundary: ToolDispatchAuditBoundary,
    adapter_only: bool,
) -> bool:
    return (
        adapter_only is True
        and dispatch_identity.tool_type != "DECLARED_CUSTOM_TOOL"
        and skill_library_manifest.replay_safe_skill_library is True
        and agent_observation_trace_receipt.replay_safe_observation_trace is True
        and execution_boundary.execution_mode in _REPLAY_SAFE_EXECUTION_MODES
        and input_boundary.input_mode in _REPLAY_SAFE_INPUT_MODES
        and output_boundary.output_mode in _REPLAY_SAFE_OUTPUT_MODES
        and audit_boundary.audit_mode in _REPLAY_SAFE_AUDIT_MODES
        and execution_boundary.declared_execution_time_ms >= 0
    )


def build_tool_dispatch_telemetry_receipt(
    skill_library_manifest: SkillLibraryManifest,
    agent_observation_trace_receipt: AgentObservationTraceReceipt,
    dispatch_identity: ToolDispatchIdentity,
    input_boundary: ToolDispatchInputBoundary,
    output_boundary: ToolDispatchOutputBoundary,
    execution_boundary: ToolDispatchExecutionBoundary,
    audit_boundary: ToolDispatchAuditBoundary,
    adapter_only: bool,
) -> ToolDispatchTelemetryReceipt:
    replay_safe_dispatch = _recompute_replay_safe_dispatch(
        skill_library_manifest,
        agent_observation_trace_receipt,
        dispatch_identity,
        input_boundary,
        output_boundary,
        execution_boundary,
        audit_boundary,
        adapter_only,
    )
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "skill_library_manifest_hash": skill_library_manifest.skill_library_manifest_hash,
        "agent_observation_trace_receipt_hash": agent_observation_trace_receipt.agent_observation_trace_receipt_hash,
        "dispatch_identity": dispatch_identity,
        "input_boundary": input_boundary,
        "output_boundary": output_boundary,
        "execution_boundary": execution_boundary,
        "audit_boundary": audit_boundary,
        "replay_safe_dispatch": replay_safe_dispatch,
        "adapter_only": adapter_only,
    }
    return ToolDispatchTelemetryReceipt(**payload, tool_dispatch_telemetry_receipt_hash=_hash_payload(payload))


def validate_tool_dispatch_identity(value: ToolDispatchIdentity) -> ToolDispatchIdentity:
    _revalidate_exact_instance(value, ToolDispatchIdentity)
    return value


def validate_tool_dispatch_input_boundary(value: ToolDispatchInputBoundary) -> ToolDispatchInputBoundary:
    _revalidate_exact_instance(value, ToolDispatchInputBoundary)
    return value


def validate_tool_dispatch_output_boundary(value: ToolDispatchOutputBoundary) -> ToolDispatchOutputBoundary:
    _revalidate_exact_instance(value, ToolDispatchOutputBoundary)
    return value


def validate_tool_dispatch_execution_boundary(value: ToolDispatchExecutionBoundary) -> ToolDispatchExecutionBoundary:
    _revalidate_exact_instance(value, ToolDispatchExecutionBoundary)
    return value


def validate_tool_dispatch_audit_boundary(value: ToolDispatchAuditBoundary) -> ToolDispatchAuditBoundary:
    _revalidate_exact_instance(value, ToolDispatchAuditBoundary)
    return value


def validate_tool_dispatch_telemetry_receipt(
    receipt: ToolDispatchTelemetryReceipt,
    skill_library_manifest: SkillLibraryManifest,
    agent_observation_trace_receipt: AgentObservationTraceReceipt,
    **upstream_kwargs: Any,
) -> ToolDispatchTelemetryReceipt:
    _revalidate_exact_instance(receipt, ToolDispatchTelemetryReceipt)
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("invalid schema_version")
    if isinstance(receipt.adapter_only, bool) is False:
        raise ValueError("adapter_only must be bool")
    if isinstance(receipt.replay_safe_dispatch, bool) is False:
        raise ValueError("replay_safe_dispatch must be bool")

    validate_tool_dispatch_identity(receipt.dispatch_identity)
    validate_tool_dispatch_input_boundary(receipt.input_boundary)
    validate_tool_dispatch_output_boundary(receipt.output_boundary)
    validate_tool_dispatch_execution_boundary(receipt.execution_boundary)
    validate_tool_dispatch_audit_boundary(receipt.audit_boundary)

    validate_skill_library_manifest(skill_library_manifest, agent_observation_trace_receipt, **upstream_kwargs)
    validate_agent_observation_trace_receipt(agent_observation_trace_receipt, **upstream_kwargs)

    if receipt.skill_library_manifest_hash != skill_library_manifest.skill_library_manifest_hash:
        raise ValueError("skill_library_manifest_hash mismatch")
    if receipt.agent_observation_trace_receipt_hash != agent_observation_trace_receipt.agent_observation_trace_receipt_hash:
        raise ValueError("agent_observation_trace_receipt_hash mismatch")
    for observed_call in agent_observation_trace_receipt.tool_call_observations:
        if (
            observed_call.tool_name == receipt.dispatch_identity.tool_name
            and observed_call.tool_input_hash == receipt.input_boundary.input_hash
            and observed_call.tool_output_hash == receipt.output_boundary.output_hash
        ):
            break
    else:
        raise ValueError("dispatch telemetry receipt is not bound to an observed tool call")

    _validate_tool_dispatch_semantics(
        receipt.input_boundary.input_reason,
        receipt.output_boundary.output_reason,
        receipt.execution_boundary.execution_reason,
        receipt.audit_boundary.audit_reason,
    )

    recomputed = _recompute_replay_safe_dispatch(
        skill_library_manifest,
        agent_observation_trace_receipt,
        receipt.dispatch_identity,
        receipt.input_boundary,
        receipt.output_boundary,
        receipt.execution_boundary,
        receipt.audit_boundary,
        receipt.adapter_only,
    )
    if receipt.replay_safe_dispatch != recomputed:
        raise ValueError("replay_safe_dispatch must be recomputed")

    _validate_hash_format(receipt.tool_dispatch_telemetry_receipt_hash, "tool_dispatch_telemetry_receipt_hash")
    if _hash_payload(_base_payload(receipt.__dict__, "tool_dispatch_telemetry_receipt_hash")) != receipt.tool_dispatch_telemetry_receipt_hash:
        raise ValueError("tool_dispatch_telemetry_receipt_hash mismatch")
    return receipt
