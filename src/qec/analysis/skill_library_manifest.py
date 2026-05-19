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

_SCHEMA_VERSION = "SKILL_LIBRARY_MANIFEST_V1"

_MAX_NAME_LENGTH = 128
_MAX_REASON_LENGTH = 1024
_MAX_SKILLS = 4096

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_SKILL_LIBRARY_TYPES = {
    "DECLARED_AGENT_SKILL_LIBRARY",
    "DECLARED_REVIEW_SKILL_LIBRARY",
    "DECLARED_TOOL_SKILL_LIBRARY",
    "DECLARED_OBSERVATION_SKILL_LIBRARY",
    "DECLARED_RESEARCH_SKILL_LIBRARY",
    "DECLARED_CUSTOM_SKILL_LIBRARY",
}
_ALLOWED_SKILL_MODES = {
    "SKILL_DECLARED_NOT_EXECUTED",
    "SKILL_DECLARED_CONTEXT_ONLY",
    "SKILL_DECLARED_REPLAY_ONLY",
    "SKILL_DECLARED_AUDIT_ONLY",
    "DECLARED_CUSTOM_SKILL_MODE",
}
_ALLOWED_VERSION_MODES = {"FIXED_VERSION", "HASH_BOUND_VERSION", "DECLARED_SEMVER_VERSION", "DECLARED_CUSTOM_VERSION"}
_ALLOWED_CAPABILITY_MODES = {
    "CAPABILITY_CONTEXT_ONLY",
    "CAPABILITY_AUDIT_ONLY",
    "CAPABILITY_REPLAY_ONLY",
    "CAPABILITY_DECLARED_NOT_EXECUTED",
    "DECLARED_CUSTOM_CAPABILITY",
}
_ALLOWED_DEPENDENCY_MODES = {
    "NO_DEPENDENCIES",
    "DECLARED_STATIC_DEPENDENCIES",
    "DECLARED_HASH_BOUND_DEPENDENCIES",
    "DECLARED_CONTEXT_DEPENDENCIES",
    "DECLARED_CUSTOM_DEPENDENCIES",
}

_FORBIDDEN_TOKENS = (
    "skill executed",
    "tool execution succeeded",
    "hidden tool call",
    "runtime dispatch",
    "live crawler",
    "autonomous planning",
    "agent output is evidence",
    "agent output as evidence",
    "automatic reasoning correctness",
    "semantic equivalence guaranteed",
    "autonomous evaluation",
    "hidden skill execution",
    "hidden tool execution",
    "hidden network semantics",
    "hidden autonomous planning",
    "hidden replay equivalence",
    "hidden mutable skill registry",
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


def _validate_skill_library_semantics(*texts: str) -> None:
    _check_no_forbidden_runtime_semantics("\n".join(texts))


def _validate_skill_indices(entries: tuple["SkillDefinitionEntry", ...]) -> None:
    seen: set[int] = set()
    for expected, entry in enumerate(entries):
        if entry.skill_index in seen:
            raise ValueError("duplicate skill_index")
        if entry.skill_index != expected:
            raise ValueError("out-of-order skill_index")
        seen.add(entry.skill_index)


def _skill_entries_replay_safe(entries: tuple["SkillDefinitionEntry", ...]) -> bool:
    try:
        _validate_skill_indices(entries)
    except ValueError:
        return False
    return all(e.skill_mode != "DECLARED_CUSTOM_SKILL_MODE" for e in entries)


def _revalidate_exact_instance(value: Any, cls: type[Any]) -> None:
    if type(value) is not cls:
        raise _invalid_input()
    cls(**value.__dict__)


@dataclass(frozen=True)
class SkillLibraryIdentity:
    library_name: str
    library_version: str
    library_type: str
    skill_library_identity_hash: str

    def __post_init__(self) -> None:
        if type(self) is not SkillLibraryIdentity:
            raise _invalid_input()
        _check_text(self.library_name, "library_name", _MAX_NAME_LENGTH)
        _check_text(self.library_version, "library_version", _MAX_NAME_LENGTH)
        if self.library_type not in _ALLOWED_SKILL_LIBRARY_TYPES:
            raise ValueError("invalid library_type")
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.skill_library_identity_hash, "skill_library_identity_hash")
        if _hash_payload(_base_payload(self.__dict__, "skill_library_identity_hash")) != self.skill_library_identity_hash:
            raise ValueError("skill_library_identity_hash mismatch")


@dataclass(frozen=True)
class SkillDefinitionEntry:
    skill_index: int
    skill_name: str
    skill_mode: str
    skill_description: str
    skill_definition_entry_hash: str

    def __post_init__(self) -> None:
        if type(self) is not SkillDefinitionEntry:
            raise _invalid_input()
        if isinstance(self.skill_index, bool) or not isinstance(self.skill_index, int) or self.skill_index < 0:
            raise ValueError("skill_index must be non-negative int")
        _check_text(self.skill_name, "skill_name", _MAX_NAME_LENGTH)
        if self.skill_mode not in _ALLOWED_SKILL_MODES:
            raise ValueError("invalid skill_mode")
        _check_text(self.skill_description, "skill_description", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.skill_definition_entry_hash, "skill_definition_entry_hash")
        if _hash_payload(_base_payload(self.__dict__, "skill_definition_entry_hash")) != self.skill_definition_entry_hash:
            raise ValueError("skill_definition_entry_hash mismatch")


@dataclass(frozen=True)
class SkillVersionDeclaration:
    version_mode: str
    version_reason: str
    skill_version_declaration_hash: str

    def __post_init__(self) -> None:
        if type(self) is not SkillVersionDeclaration:
            raise _invalid_input()
        if self.version_mode not in _ALLOWED_VERSION_MODES:
            raise ValueError("invalid version_mode")
        _check_text(self.version_reason, "version_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.skill_version_declaration_hash, "skill_version_declaration_hash")
        if _hash_payload(_base_payload(self.__dict__, "skill_version_declaration_hash")) != self.skill_version_declaration_hash:
            raise ValueError("skill_version_declaration_hash mismatch")


@dataclass(frozen=True)
class SkillCapabilityBoundary:
    capability_mode: str
    capability_reason: str
    skill_capability_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not SkillCapabilityBoundary:
            raise _invalid_input()
        if self.capability_mode not in _ALLOWED_CAPABILITY_MODES:
            raise ValueError("invalid capability_mode")
        _check_text(self.capability_reason, "capability_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.skill_capability_boundary_hash, "skill_capability_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "skill_capability_boundary_hash")) != self.skill_capability_boundary_hash:
            raise ValueError("skill_capability_boundary_hash mismatch")


@dataclass(frozen=True)
class SkillDependencyBoundary:
    dependency_mode: str
    dependency_reason: str
    skill_dependency_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not SkillDependencyBoundary:
            raise _invalid_input()
        if self.dependency_mode not in _ALLOWED_DEPENDENCY_MODES:
            raise ValueError("invalid dependency_mode")
        _check_text(self.dependency_reason, "dependency_reason", _MAX_REASON_LENGTH)
        _check_no_forbidden_runtime_semantics(self.__dict__)
        _validate_hash_format(self.skill_dependency_boundary_hash, "skill_dependency_boundary_hash")
        if _hash_payload(_base_payload(self.__dict__, "skill_dependency_boundary_hash")) != self.skill_dependency_boundary_hash:
            raise ValueError("skill_dependency_boundary_hash mismatch")


@dataclass(frozen=True)
class SkillLibraryManifest:
    schema_version: str
    agent_observation_trace_receipt_hash: str
    skill_library_identity: SkillLibraryIdentity
    skill_entries: tuple[SkillDefinitionEntry, ...]
    version_declaration: SkillVersionDeclaration
    capability_boundary: SkillCapabilityBoundary
    dependency_boundary: SkillDependencyBoundary
    skill_count: int
    replay_safe_skill_library: bool
    adapter_only: bool
    skill_library_manifest_hash: str


def build_skill_library_identity(library_name: str, library_version: str, library_type: str) -> SkillLibraryIdentity:
    payload = {"library_name": library_name, "library_version": library_version, "library_type": library_type}
    return SkillLibraryIdentity(**payload, skill_library_identity_hash=_hash_payload(payload))


def build_skill_definition_entry(skill_index: int, skill_name: str, skill_mode: str, skill_description: str) -> SkillDefinitionEntry:
    payload = {
        "skill_index": skill_index,
        "skill_name": skill_name,
        "skill_mode": skill_mode,
        "skill_description": skill_description,
    }
    return SkillDefinitionEntry(**payload, skill_definition_entry_hash=_hash_payload(payload))


def build_skill_version_declaration(version_mode: str, version_reason: str) -> SkillVersionDeclaration:
    payload = {"version_mode": version_mode, "version_reason": version_reason}
    return SkillVersionDeclaration(**payload, skill_version_declaration_hash=_hash_payload(payload))


def build_skill_capability_boundary(capability_mode: str, capability_reason: str) -> SkillCapabilityBoundary:
    payload = {"capability_mode": capability_mode, "capability_reason": capability_reason}
    return SkillCapabilityBoundary(**payload, skill_capability_boundary_hash=_hash_payload(payload))


def build_skill_dependency_boundary(dependency_mode: str, dependency_reason: str) -> SkillDependencyBoundary:
    payload = {"dependency_mode": dependency_mode, "dependency_reason": dependency_reason}
    return SkillDependencyBoundary(**payload, skill_dependency_boundary_hash=_hash_payload(payload))


def build_skill_library_manifest(
    agent_observation_trace_receipt: AgentObservationTraceReceipt,
    skill_library_identity: SkillLibraryIdentity,
    skill_entries: tuple[SkillDefinitionEntry, ...],
    version_declaration: SkillVersionDeclaration,
    capability_boundary: SkillCapabilityBoundary,
    dependency_boundary: SkillDependencyBoundary,
    adapter_only: bool,
) -> SkillLibraryManifest:
    skill_count = len(skill_entries)
    replay_safe = (
        adapter_only is True
        and agent_observation_trace_receipt.replay_safe_observation_trace is True
        and skill_library_identity.library_type != "DECLARED_CUSTOM_SKILL_LIBRARY"
        and version_declaration.version_mode in {"FIXED_VERSION", "HASH_BOUND_VERSION", "DECLARED_SEMVER_VERSION"}
        and capability_boundary.capability_mode in {"CAPABILITY_CONTEXT_ONLY", "CAPABILITY_AUDIT_ONLY", "CAPABILITY_REPLAY_ONLY", "CAPABILITY_DECLARED_NOT_EXECUTED"}
        and dependency_boundary.dependency_mode in {"NO_DEPENDENCIES", "DECLARED_STATIC_DEPENDENCIES", "DECLARED_HASH_BOUND_DEPENDENCIES", "DECLARED_CONTEXT_DEPENDENCIES"}
        and type(skill_entries) is tuple
        and _skill_entries_replay_safe(skill_entries)
    )
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "agent_observation_trace_receipt_hash": agent_observation_trace_receipt.agent_observation_trace_receipt_hash,
        "skill_library_identity": skill_library_identity,
        "skill_entries": skill_entries,
        "version_declaration": version_declaration,
        "capability_boundary": capability_boundary,
        "dependency_boundary": dependency_boundary,
        "skill_count": skill_count,
        "replay_safe_skill_library": replay_safe,
        "adapter_only": adapter_only,
    }
    return SkillLibraryManifest(**payload, skill_library_manifest_hash=_hash_payload(payload))


def validate_skill_library_identity(identity: SkillLibraryIdentity) -> SkillLibraryIdentity:
    _revalidate_exact_instance(identity, SkillLibraryIdentity)
    return identity


def validate_skill_definition_entry(entry: SkillDefinitionEntry) -> SkillDefinitionEntry:
    _revalidate_exact_instance(entry, SkillDefinitionEntry)
    return entry


def validate_skill_version_declaration(decl: SkillVersionDeclaration) -> SkillVersionDeclaration:
    _revalidate_exact_instance(decl, SkillVersionDeclaration)
    return decl


def validate_skill_capability_boundary(boundary: SkillCapabilityBoundary) -> SkillCapabilityBoundary:
    _revalidate_exact_instance(boundary, SkillCapabilityBoundary)
    return boundary


def validate_skill_dependency_boundary(boundary: SkillDependencyBoundary) -> SkillDependencyBoundary:
    _revalidate_exact_instance(boundary, SkillDependencyBoundary)
    return boundary


def validate_skill_library_manifest(
    manifest: SkillLibraryManifest,
    agent_observation_trace_receipt: AgentObservationTraceReceipt,
    **trace_kwargs: Any,
) -> SkillLibraryManifest:
    if type(manifest) is not SkillLibraryManifest:
        raise _invalid_input()
    if manifest.schema_version != _SCHEMA_VERSION:
        raise ValueError("invalid schema_version")
    if type(manifest.adapter_only) is not bool or type(manifest.replay_safe_skill_library) is not bool:
        raise ValueError("adapter_only and replay_safe_skill_library must be bool")
    if isinstance(manifest.skill_count, bool) or not isinstance(manifest.skill_count, int):
        raise ValueError("skill_count must be int")
    if type(manifest.skill_entries) is not tuple:
        raise ValueError("skill_entries must be tuple")
    if manifest.skill_count < 0 or manifest.skill_count > _MAX_SKILLS:
        raise ValueError("skill_count out of bounds")

    validate_agent_observation_trace_receipt(agent_observation_trace_receipt, **trace_kwargs)
    if manifest.agent_observation_trace_receipt_hash != agent_observation_trace_receipt.agent_observation_trace_receipt_hash:
        raise ValueError("agent_observation_trace_receipt_hash mismatch")

    _validate_hash_format(manifest.agent_observation_trace_receipt_hash, "agent_observation_trace_receipt_hash")
    _validate_hash_format(manifest.skill_library_manifest_hash, "skill_library_manifest_hash")

    validate_skill_library_identity(manifest.skill_library_identity)
    validate_skill_version_declaration(manifest.version_declaration)
    validate_skill_capability_boundary(manifest.capability_boundary)
    validate_skill_dependency_boundary(manifest.dependency_boundary)
    for e in manifest.skill_entries:
        validate_skill_definition_entry(e)

    _validate_skill_indices(manifest.skill_entries)
    recomputed_count = len(manifest.skill_entries)
    if manifest.skill_count != recomputed_count:
        raise ValueError("skill_count mismatch")

    _validate_skill_library_semantics(
        manifest.version_declaration.version_reason,
        manifest.capability_boundary.capability_reason,
        manifest.dependency_boundary.dependency_reason,
        *[e.skill_description for e in manifest.skill_entries],
    )
    _check_no_forbidden_runtime_semantics(manifest.__dict__)

    recomputed_replay = (
        manifest.adapter_only is True
        and agent_observation_trace_receipt.replay_safe_observation_trace is True
        and manifest.skill_library_identity.library_type != "DECLARED_CUSTOM_SKILL_LIBRARY"
        and recomputed_count == manifest.skill_count
        and type(manifest.skill_entries) is tuple
        and _skill_entries_replay_safe(manifest.skill_entries)
        and manifest.version_declaration.version_mode in {"FIXED_VERSION", "HASH_BOUND_VERSION", "DECLARED_SEMVER_VERSION"}
        and manifest.capability_boundary.capability_mode in {"CAPABILITY_CONTEXT_ONLY", "CAPABILITY_AUDIT_ONLY", "CAPABILITY_REPLAY_ONLY", "CAPABILITY_DECLARED_NOT_EXECUTED"}
        and manifest.dependency_boundary.dependency_mode in {"NO_DEPENDENCIES", "DECLARED_STATIC_DEPENDENCIES", "DECLARED_HASH_BOUND_DEPENDENCIES", "DECLARED_CONTEXT_DEPENDENCIES"}
    )
    if manifest.replay_safe_skill_library is not recomputed_replay:
        raise ValueError("replay_safe_skill_library must be recomputed")

    if _hash_payload(_base_payload(manifest.__dict__, "skill_library_manifest_hash")) != manifest.skill_library_manifest_hash:
        raise ValueError("skill_library_manifest_hash mismatch")
    return manifest
