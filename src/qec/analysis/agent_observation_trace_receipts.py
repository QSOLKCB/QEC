from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from qec.analysis.inference_backend_manifest import InferenceBackendManifest, validate_inference_backend_manifest
from qec.analysis.kv_cache_policy_receipts import KVCachePolicyReceipt, validate_kv_cache_policy_receipt

_SCHEMA_VERSION = "AGENT_OBSERVATION_TRACE_RECEIPT_V1"

_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 1024
_MAX_SEQUENCE_LENGTH = 4096

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_AGENT_TYPES = {
    "DECLARED_ASSISTANT_AGENT",
    "DECLARED_REVIEW_AGENT",
    "DECLARED_TOOL_ROUTER",
    "DECLARED_OBSERVATION_AGENT",
    "DECLARED_RESEARCH_AGENT",
    "DECLARED_CUSTOM_AGENT",
}
_ALLOWED_PATTERN_TYPES = {
    "SEQUENTIAL_TOOL_PATTERN",
    "REVIEW_ONLY_PATTERN",
    "DECLARED_TOOL_ROUTING_PATTERN",
    "DECLARED_OBSERVATION_PATTERN",
    "DECLARED_CONTEXT_PATTERN",
    "DECLARED_CUSTOM_PATTERN",
}
_ALLOWED_TOOL_OBSERVATION_MODES = {
    "TOOL_DECLARED_NOT_EXECUTED",
    "TOOL_DECLARED_CONTEXT_ONLY",
    "TOOL_DECLARED_REPLAY_ONLY",
    "TOOL_DECLARED_AUDIT_ONLY",
    "DECLARED_CUSTOM_TOOL_MODE",
}
_ALLOWED_DECISION_MODES = {
    "DECLARED_PATTERN_DECISION",
    "DECLARED_CONTEXT_SELECTION",
    "DECLARED_ROUTING_SELECTION",
    "DECLARED_AUDIT_SELECTION",
    "DECLARED_CUSTOM_DECISION",
}
_ALLOWED_OBSERVATION_SEQUENCE_MODES = {
    "STRICT_ORDERED_SEQUENCE",
    "DECLARED_CONTEXTUAL_SEQUENCE",
    "DECLARED_REPLAY_SEQUENCE",
    "DECLARED_AUDIT_SEQUENCE",
    "DECLARED_CUSTOM_SEQUENCE",
}
_FORBIDDEN_TOKENS = (
    "autonomous reasoning",
    "agent proved",
    "tool execution succeeded",
    "hidden tool call",
    "runtime dispatch",
    "live crawler",
    "automatic reasoning correctness",
    "semantic equivalence guaranteed",
    "autonomous evaluation",
    "hidden tool execution",
    "hidden network semantics",
    "hidden autonomous planning",
    "hidden replay equivalence",
    "hidden observation mutation",
    "agent output as evidence",
)


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


def _to_canonical_obj(value: Any) -> Any:
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return {k: _to_canonical_obj(v) for k, v in value.__dict__.items()}
    if isinstance(value, Mapping):
        return {k: _to_canonical_obj(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_canonical_obj(v) for v in value]
    if isinstance(value, list):
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


def _validate_agent_observation_semantics(*reasons: str) -> None:
    _check_no_forbidden_runtime_semantics("\n".join(reasons))


def _validate_sequence_ordering(decisions: tuple["IntermediateDecisionObservation", ...]) -> None:
    seen: set[int] = set()
    expected = 0
    for decision in decisions:
        if decision.decision_index in seen:
            raise ValueError("duplicate decision_index")
        if decision.decision_index != expected:
            raise ValueError("out-of-order decision_index")
        seen.add(decision.decision_index)
        expected += 1


@dataclass(frozen=True)
class AgentIdentity:
    agent_name: str
    agent_version: str
    agent_type: str
    agent_identity_hash: str

    def __post_init__(self) -> None:
        if type(self) is not AgentIdentity:
            raise _invalid_input()
        _check_text(self.agent_name, "agent_name", _MAX_NAME_LENGTH)
        _check_text(self.agent_version, "agent_version", _MAX_NAME_LENGTH)
        if self.agent_type not in _ALLOWED_AGENT_TYPES:
            raise ValueError("invalid agent_type")
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.agent_identity_hash, "agent_identity_hash")
        if _hash_payload(_base_payload(self.__dict__, "agent_identity_hash")) != self.agent_identity_hash:
            raise ValueError("agent_identity_hash mismatch")


@dataclass(frozen=True)
class AgentPatternDeclaration:
    pattern_type: str
    pattern_reason: str
    agent_pattern_declaration_hash: str

    def __post_init__(self) -> None:
        if type(self) is not AgentPatternDeclaration:
            raise _invalid_input()
        if self.pattern_type not in _ALLOWED_PATTERN_TYPES:
            raise ValueError("invalid pattern_type")
        _check_text(self.pattern_reason, "pattern_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.agent_pattern_declaration_hash, "agent_pattern_declaration_hash")
        if _hash_payload(_base_payload(self.__dict__, "agent_pattern_declaration_hash")) != self.agent_pattern_declaration_hash:
            raise ValueError("agent_pattern_declaration_hash mismatch")


@dataclass(frozen=True)
class ToolCallObservation:
    tool_name: str
    tool_mode: str
    tool_input_hash: str
    tool_output_hash: str
    tool_reason: str
    tool_call_observation_hash: str

    def __post_init__(self) -> None:
        if type(self) is not ToolCallObservation:
            raise _invalid_input()
        _check_text(self.tool_name, "tool_name", _MAX_NAME_LENGTH)
        if self.tool_mode not in _ALLOWED_TOOL_OBSERVATION_MODES:
            raise ValueError("invalid tool_mode")
        _validate_hash_format(self.tool_input_hash, "tool_input_hash")
        _validate_hash_format(self.tool_output_hash, "tool_output_hash")
        _check_text(self.tool_reason, "tool_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.tool_call_observation_hash, "tool_call_observation_hash")
        if _hash_payload(_base_payload(self.__dict__, "tool_call_observation_hash")) != self.tool_call_observation_hash:
            raise ValueError("tool_call_observation_hash mismatch")


@dataclass(frozen=True)
class IntermediateDecisionObservation:
    decision_index: int
    decision_mode: str
    decision_reason: str
    intermediate_decision_observation_hash: str

    def __post_init__(self) -> None:
        if type(self) is not IntermediateDecisionObservation:
            raise _invalid_input()
        if isinstance(self.decision_index, bool) or not isinstance(self.decision_index, int) or self.decision_index < 0:
            raise ValueError("decision_index must be non-negative int")
        if self.decision_mode not in _ALLOWED_DECISION_MODES:
            raise ValueError("invalid decision_mode")
        _check_text(self.decision_reason, "decision_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.intermediate_decision_observation_hash, "intermediate_decision_observation_hash")
        if _hash_payload(_base_payload(self.__dict__, "intermediate_decision_observation_hash")) != self.intermediate_decision_observation_hash:
            raise ValueError("intermediate_decision_observation_hash mismatch")


@dataclass(frozen=True)
class ObservationSequenceBoundary:
    sequence_mode: str
    declared_step_count: int
    sequence_reason: str
    observation_sequence_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not ObservationSequenceBoundary:
            raise _invalid_input()
        if self.sequence_mode not in _ALLOWED_OBSERVATION_SEQUENCE_MODES:
            raise ValueError("invalid sequence_mode")
        if isinstance(self.declared_step_count, bool) or not isinstance(self.declared_step_count, int):
            raise ValueError("declared_step_count must be int")
        if self.declared_step_count < 0 or self.declared_step_count > _MAX_SEQUENCE_LENGTH:
            raise ValueError("declared_step_count out of bounds")
        _check_text(self.sequence_reason, "sequence_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.observation_sequence_boundary_hash, "observation_sequence_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "observation_sequence_boundary_hash")) != self.observation_sequence_boundary_hash:
            raise ValueError("observation_sequence_boundary_hash mismatch")


@dataclass(frozen=True)
class AgentObservationTraceReceipt:
    schema_version: str
    inference_backend_manifest_hash: str
    kv_cache_policy_receipt_hash: str
    agent_identity: AgentIdentity
    pattern_declaration: AgentPatternDeclaration
    tool_call_observations: tuple[ToolCallObservation, ...]
    intermediate_decision_observations: tuple[IntermediateDecisionObservation, ...]
    observation_sequence_boundary: ObservationSequenceBoundary
    replay_safe_observation_trace: bool
    adapter_only: bool
    agent_observation_trace_receipt_hash: str


def build_agent_observation_trace_receipt(
    inference_backend_manifest: InferenceBackendManifest,
    kv_cache_policy_receipt: KVCachePolicyReceipt,
    agent_identity: AgentIdentity,
    pattern_declaration: AgentPatternDeclaration,
    tool_call_observations: tuple[ToolCallObservation, ...],
    intermediate_decision_observations: tuple[IntermediateDecisionObservation, ...],
    observation_sequence_boundary: ObservationSequenceBoundary,
    adapter_only: bool,
) -> AgentObservationTraceReceipt:
    observed_count = len(tool_call_observations) + len(intermediate_decision_observations)
    replay_safe = (
        adapter_only is True
        and observation_sequence_boundary.sequence_mode == "STRICT_ORDERED_SEQUENCE"
        and observation_sequence_boundary.declared_step_count == observed_count
    )
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "inference_backend_manifest_hash": inference_backend_manifest.inference_backend_manifest_hash,
        "kv_cache_policy_receipt_hash": kv_cache_policy_receipt.kv_cache_policy_receipt_hash,
        "agent_identity": agent_identity,
        "pattern_declaration": pattern_declaration,
        "tool_call_observations": tool_call_observations,
        "intermediate_decision_observations": intermediate_decision_observations,
        "observation_sequence_boundary": observation_sequence_boundary,
        "replay_safe_observation_trace": replay_safe,
        "adapter_only": adapter_only,
    }
    return AgentObservationTraceReceipt(**payload, agent_observation_trace_receipt_hash=_hash_payload(payload))


def validate_agent_observation_trace_receipt(
    receipt: AgentObservationTraceReceipt,
    inference_backend_manifest: InferenceBackendManifest,
    kv_cache_policy_receipt: KVCachePolicyReceipt,
    **kv_kwargs: Any,
) -> AgentObservationTraceReceipt:
    if type(receipt) is not AgentObservationTraceReceipt:
        raise _invalid_input()
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("invalid schema_version")
    if type(receipt.adapter_only) is not bool or type(receipt.replay_safe_observation_trace) is not bool:
        raise ValueError("adapter_only and replay_safe_observation_trace must be bool")
    if type(receipt.tool_call_observations) is not tuple or type(receipt.intermediate_decision_observations) is not tuple:
        raise ValueError("observations must be tuples")

    validate_inference_backend_manifest(inference_backend_manifest)
    validate_kv_cache_policy_receipt(kv_cache_policy_receipt, inference_backend_manifest, **kv_kwargs)

    if receipt.inference_backend_manifest_hash != inference_backend_manifest.inference_backend_manifest_hash:
        raise ValueError("inference_backend_manifest_hash mismatch")
    if receipt.kv_cache_policy_receipt_hash != kv_cache_policy_receipt.kv_cache_policy_receipt_hash:
        raise ValueError("kv_cache_policy_receipt_hash mismatch")

    _validate_hash_format(receipt.inference_backend_manifest_hash, "inference_backend_manifest_hash")
    _validate_hash_format(receipt.kv_cache_policy_receipt_hash, "kv_cache_policy_receipt_hash")
    _validate_hash_format(receipt.agent_observation_trace_receipt_hash, "agent_observation_trace_receipt_hash")

    if type(receipt.agent_identity) is not AgentIdentity or type(receipt.pattern_declaration) is not AgentPatternDeclaration:
        raise _invalid_input()
    if type(receipt.observation_sequence_boundary) is not ObservationSequenceBoundary:
        raise _invalid_input()

    for item in receipt.tool_call_observations:
        if type(item) is not ToolCallObservation:
            raise _invalid_input()
    for item in receipt.intermediate_decision_observations:
        if type(item) is not IntermediateDecisionObservation:
            raise _invalid_input()

    _validate_sequence_ordering(receipt.intermediate_decision_observations)
    _validate_agent_observation_semantics(
        receipt.pattern_declaration.pattern_reason,
        receipt.observation_sequence_boundary.sequence_reason,
        *[t.tool_reason for t in receipt.tool_call_observations],
        *[d.decision_reason for d in receipt.intermediate_decision_observations],
    )
    _check_no_forbidden_runtime_semantics(receipt.__dict__)

    observed_count = len(receipt.tool_call_observations) + len(receipt.intermediate_decision_observations)
    if receipt.observation_sequence_boundary.declared_step_count != observed_count:
        raise ValueError("declared_step_count mismatch")

    recomputed_replay = (
        receipt.adapter_only is True
        and receipt.observation_sequence_boundary.sequence_mode == "STRICT_ORDERED_SEQUENCE"
        and receipt.observation_sequence_boundary.declared_step_count == observed_count
    )
    if receipt.replay_safe_observation_trace is not recomputed_replay:
        raise ValueError("replay_safe_observation_trace must be recomputed")

    recomputed_hash = _hash_payload(_base_payload(receipt.__dict__, "agent_observation_trace_receipt_hash"))
    if recomputed_hash != receipt.agent_observation_trace_receipt_hash:
        raise ValueError("agent_observation_trace_receipt_hash mismatch")

    return receipt
