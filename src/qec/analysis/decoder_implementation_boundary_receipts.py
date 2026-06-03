from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, ClassVar, Mapping, Sequence

IMPLEMENTATION_BOUNDARY_RELEASE = "v166.5"
RECEIPT_KIND = "DecoderImplementationBoundaryReceipt"
PREVIOUS_RELEASE_TAG = "v166.4"
PREVIOUS_RELEASE_URL = "https://github.com/QSOLKCB/QEC/releases/tag/v166.4"

_IMPLEMENTATION_STATUS = "BOUNDARY_DECLARED_NOT_ENABLED"
_IMPLEMENTATION_MODE = "DECLARED_IMPLEMENTATION_BOUNDARY_ONLY"
_ARTIFACT_MODE = "DECLARED_HASH_BOUND_ARTIFACT"
_SOURCE_BOUNDARY_MODE = "DECLARED_IMPLEMENTATION_SOURCE_HASH_BOUND_NO_RUNTIME"
_RUNTIME_BOUNDARY_MODE = "NO_RUNTIME_AUTHORITY_FROM_IMPLEMENTATION_BOUNDARY"
_CONFIG_MODE = "DECLARED_STATIC_CONFIGURATION_BOUNDARY"
_BUILD_BOUNDARY_MODE = "DECLARED_BUILD_BOUNDARY_NO_EXECUTION"
_EQUIVALENCE_MODE = "EXACT_FAST_PATH_OUTPUT_MATCH"
_EQUIVALENCE_SCOPE = "DECLARED_CORPUS_ONLY"
_PRECISION_POLICY = "DECLARED_EXACT_NO_HIDDEN_PRECISION_DRIFT"
_APPROXIMATION_POLICY = "NO_UNDECLARED_APPROXIMATION"
_AUDIT_MODE = "STATIC_BOUNDARY_AUDIT_DECLARED"
_ROLLBACK_GATE_MODE = "FUTURE_ROLLBACK_RECEIPT_REQUIRED"
_AUTHORITY_MODE = "NO_AUTHORITY_FROM_IMPLEMENTATION_BOUNDARY"
_REQUIRED_ROLLBACK_KIND = "DecoderRollbackReceipt"
_REQUIRED_ROLLBACK_RELEASE = "v166.7"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_TEXT_LENGTH = 512
_FORBIDDEN_DECODER_ROOT = "src/qec/decoder/"

IMPLEMENTATION_KINDS = frozenset({
    "DECLARED_DECODER_IMPLEMENTATION_BOUNDARY",
    "FAST_PATH_IMPLEMENTATION_BOUNDARY",
    "ADAPTER_IMPLEMENTATION_BOUNDARY",
    "SPARSE_REPRESENTATION_IMPLEMENTATION_BOUNDARY",
    "MEMORY_LAYOUT_IMPLEMENTATION_BOUNDARY",
    "GRAPH_CONSTRUCTION_IMPLEMENTATION_BOUNDARY",
    "PRECISION_PRESERVING_IMPLEMENTATION_BOUNDARY",
})
ARTIFACT_ROLES = frozenset({
    "IMPLEMENTATION_BOUNDARY_DECLARATION",
    "IMPLEMENTATION_CONFIG_DECLARATION",
    "IMPLEMENTATION_ADAPTER_DECLARATION",
    "IMPLEMENTATION_SOURCE_HASH_DECLARATION",
    "IMPLEMENTATION_TEST_FIXTURE_DECLARATION",
    "IMPLEMENTATION_BUILD_MANIFEST_DECLARATION",
})
ARTIFACT_LANGUAGES = frozenset({
    "PYTHON_DECLARATION", "JSON_DECLARATION", "YAML_DECLARATION", "TEXT_DECLARATION",
    "TOML_DECLARATION", "RUST_DECLARATION", "C_DECLARATION", "CPP_DECLARATION", "UNKNOWN_DECLARATION",
})
ALLOWED_SOURCE_ROOTS = frozenset({
    "implementation_boundaries/",
    "external/decoder_implementation_boundaries/",
    "src/qec/analysis/decoder_implementation_boundaries/",
})

_FORBIDDEN_DECLARATION_TOKENS = (
    "silent decoder replacement", "candidate replaces baseline", "decoder replaced because faster",
    "speed proves correctness", "benchmark proves correctness", "benchmark marketing", "runtime promotion",
    "candidate decoder promoted", "candidate decoder authority", "probabilistic decoder authority",
    "probabilistic decoder promotion", "ml decoder authority", "hardware authority", "qec advantage proven",
    "mutation of canonical decoder", "deleting rollback path", "rollback bypass", "hidden precision drift",
    "undeclared approximation policy", "output accepted as universal canonical truth", "global correctness proven",
    "replay equivalence implies promotion", "replay equivalence implies speedup", "optimization implies correctness",
    "optimization grants execution authority", "contract permits implementation", "fast path accepted",
    "fast path implemented", "fast path runtime enabled", "fast path proves speedup", "benchmark proves fast path",
    "implementation permission granted", "implementation enabled", "implementation proves correctness",
    "implementation proves speedup", "implementation replaces baseline", "runtime implementation authority",
    "build proves correctness", "config grants runtime authority",
)
_SEMANTIC_GUARD_EXACT_ALLOWLIST = {
    _PRECISION_POLICY, _APPROXIMATION_POLICY, "implementation_boundary_safe", "implementation_boundary_release",
    "rollback_receipt_required_before_promotion", "future_benchmark_ladder_required", "future_promotion_receipt_required",
}

class DecoderImplementationBoundaryErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_HASH = "INVALID_HASH"
    HASH_MISMATCH = "HASH_MISMATCH"
    INVALID_DECODER_IMPLEMENTATION_BOUNDARY = "INVALID_DECODER_IMPLEMENTATION_BOUNDARY"

class DecoderImplementationBoundaryError(ValueError):
    def __init__(self, code: DecoderImplementationBoundaryErrorCode, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code.value}:{detail}")

def _error(code: DecoderImplementationBoundaryErrorCode, detail: str) -> DecoderImplementationBoundaryError:
    return DecoderImplementationBoundaryError(code, detail)
def _invalid_input(detail: str = "GENERIC") -> DecoderImplementationBoundaryError:
    return _error(DecoderImplementationBoundaryErrorCode.INVALID_INPUT, detail)
def _invalid_hash(detail: str = "FORMAT") -> DecoderImplementationBoundaryError:
    return _error(DecoderImplementationBoundaryErrorCode.INVALID_HASH, detail)
def _hash_mismatch(detail: str) -> DecoderImplementationBoundaryError:
    return _error(DecoderImplementationBoundaryErrorCode.HASH_MISMATCH, detail)
def _invalid_boundary(detail: str) -> DecoderImplementationBoundaryError:
    return _error(DecoderImplementationBoundaryErrorCode.INVALID_DECODER_IMPLEMENTATION_BOUNDARY, detail)

def _normalize_semantics_text(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"[\n\r\t]", " ", lowered)
    lowered = re.sub(r"\\[nrt]", " ", lowered)
    lowered = lowered.replace("_", " ").replace("-", " ")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())

def _check_forbidden_declaration_semantics(value: Any, field_name: str = "text") -> None:
    if not isinstance(value, str) or value in _SEMANTIC_GUARD_EXACT_ALLOWLIST:
        return
    normalized = _normalize_semantics_text(value)
    for token in _FORBIDDEN_DECLARATION_TOKENS:
        nt = _normalize_semantics_text(token)
        if nt in normalized:
            raise _invalid_input(f"{field_name}:FORBIDDEN_DECLARATION:{nt.replace(' ', '_')}")

def _to_canonical_obj(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _dataclass_payload(value, exclude_hash_field=None)
    if isinstance(value, Mapping):
        return {str(k): _to_canonical_obj(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_canonical_obj(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    raise _invalid_input("NON_JSON_VALUE")

def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_to_canonical_obj(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
def _hash_payload(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))
def _dataclass_payload(obj: Any, *, exclude_hash_field: str | None) -> dict[str, Any]:
    if not is_dataclass(obj) or isinstance(obj, type):
        raise _invalid_input("DATACLASS")
    payload: dict[str, Any] = {}
    for field in fields(obj):
        if field.name == exclude_hash_field:
            continue
        payload[field.name] = _to_canonical_obj(getattr(obj, field.name))
    return payload

def _payload_without(obj: Any, name: str) -> dict[str, Any]:
    return _dataclass_payload(obj, exclude_hash_field=name)
def _validate_hash_format(value: str, field_name: str = "sha256") -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise _invalid_hash(f"{field_name}:FORMAT")
def _assert_hash_matches(obj: Any, field_name: str, payload_fn: Any) -> None:
    expected = getattr(obj, field_name)
    _validate_hash_format(expected, field_name)
    if _hash_payload(payload_fn(obj)) != expected:
        raise _hash_mismatch(field_name)
def _revalidate_exact_instance(value: Any, cls: type[Any]) -> None:
    if type(value) is not cls or not is_dataclass(value):
        raise _invalid_input(f"{cls.__name__}:EXACT_DATACLASS")
    if tuple(f.name for f in fields(value)) != tuple(f.name for f in fields(cls)):
        raise _invalid_input(f"{cls.__name__}:EXACT_DATACLASS")
    value.__post_init__()
def _require_exact_bool(value: Any, field_name: str) -> None:
    if type(value) is not bool:
        raise _invalid_input(f"{field_name}:BOOL")
def _require_exact_int(value: Any, field_name: str) -> None:
    if type(value) is not int:
        raise _invalid_input(f"{field_name}:INT")
def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT_LENGTH:
        raise _invalid_input(f"{field_name}:TEXT")
    _check_forbidden_declaration_semantics(value, field_name)
def _require_flags(obj: Any, expected: Mapping[str, bool], detail: str, *, boundary: bool = False) -> None:
    for name, expected_value in expected.items():
        value = getattr(obj, name)
        _require_exact_bool(value, name)
        if value is not expected_value:
            if boundary:
                raise _invalid_boundary(detail)
            raise _invalid_input(detail)
def _validate_posix_relative_path(path: str, field_name: str, *, require_trailing_slash: bool = False) -> None:
    if not isinstance(path, str) or not path:
        raise _invalid_input(f"{field_name}:TEXT")
    _check_forbidden_declaration_semantics(path, field_name)
    if "\\" in path: raise _invalid_input(f"{field_name}:BACKSLASH")
    if path.startswith("/"): raise _invalid_input(f"{field_name}:ABSOLUTE")
    if "//" in path: raise _invalid_input(f"{field_name}:DOUBLE_SLASH")
    if require_trailing_slash and not path.endswith("/"): raise _invalid_input(f"{field_name}:TRAILING_SLASH")
    parts = path.split("/")
    components = parts[:-1] if require_trailing_slash and parts[-1] == "" else parts
    if any(part in {"", ".", ".."} for part in components): raise _invalid_input(f"{field_name}:NON_CANONICAL_COMPONENT")
    canonical = PurePosixPath(path.rstrip("/")).as_posix() + ("/" if path.endswith("/") else "")
    if canonical != path: raise _invalid_input(f"{field_name}:NON_CANONICAL")
def _validate_artifact_path(path: str) -> None:
    _validate_posix_relative_path(path, "artifact_path")
    if path == _FORBIDDEN_DECODER_ROOT.rstrip("/") or path.startswith(_FORBIDDEN_DECODER_ROOT):
        raise _invalid_input("artifact_path:DECODER_ROOT_FORBIDDEN")
def _validate_source_root(root: str) -> None:
    _validate_posix_relative_path(root, "implementation_source_root", require_trailing_slash=True)
    if root == _FORBIDDEN_DECODER_ROOT or root.startswith(_FORBIDDEN_DECODER_ROOT):
        raise _invalid_input("implementation_source_root:DECODER_ROOT_FORBIDDEN")
    if root not in ALLOWED_SOURCE_ROOTS:
        raise _invalid_input("implementation_source_root:UNAPPROVED_DECLARATIVE_ROOT")
def _sorted_artifacts(artifacts: Sequence[Any]) -> tuple["DecoderImplementationArtifact", ...]:
    if not isinstance(artifacts, tuple) or not artifacts:
        raise _invalid_input("implementation_artifacts:TUPLE")
    for artifact in artifacts:
        _revalidate_exact_instance(artifact, DecoderImplementationArtifact)
    ordered = tuple(sorted(artifacts, key=lambda a: (a.artifact_path, a.artifact_id, a.artifact_sha256, a.decoder_implementation_artifact_hash)))
    if len({a.artifact_id for a in ordered}) != len(ordered): raise _invalid_input("implementation_artifacts:DUPLICATE_ID")
    if len({a.artifact_path for a in ordered}) != len(ordered): raise _invalid_input("implementation_artifacts:DUPLICATE_PATH")
    return ordered
def _source_tree_hash_payload(mode: str, artifacts: tuple["DecoderImplementationArtifact", ...]) -> dict[str, Any]:
    return {"source_boundary_mode": mode, "implementation_artifacts": tuple({"artifact_path": a.artifact_path, "artifact_sha256": a.artifact_sha256, "artifact_role": a.artifact_role, "artifact_schema_hash": a.artifact_schema_hash} for a in artifacts)}
def _compute_source_tree_hash(mode: str, artifacts: tuple["DecoderImplementationArtifact", ...]) -> str:
    return _hash_payload(_source_tree_hash_payload(mode, artifacts))
_RUNTIME_DENIED_FLAGS = ("baseline_decoder_import_allowed", "candidate_decoder_import_allowed", "fast_path_import_allowed", "implementation_import_allowed", "runtime_decoder_execution_allowed", "candidate_runtime_execution_allowed", "fast_path_runtime_execution_allowed", "implementation_runtime_execution_allowed", "replay_execution_allowed", "optimization_execution_allowed", "benchmark_execution_allowed", "network_allowed", "heavy_backend_import_allowed", "hardware_sdk_allowed", "filesystem_mutation_allowed")
_CONFIG_DENIED_FLAGS = ("mutable_runtime_config_allowed", "environment_variable_dependency_allowed", "wall_clock_dependency_allowed", "randomness_dependency_allowed", "filesystem_probe_dependency_allowed", "network_dependency_allowed", "hardware_probe_dependency_allowed")
_BUILD_DENIED_FLAGS = ("build_execution_allowed", "dependency_install_allowed", "network_resolution_allowed", "native_extension_build_allowed", "hardware_specific_build_allowed", "unpinned_dependency_allowed", "build_cache_authority_allowed")
_AUTHORITY_DENIED_FLAGS = ("runtime_authority_allowed", "implementation_authority_allowed", "benchmark_authority_allowed", "hardware_authority_allowed", "ml_decoder_authority_allowed", "probabilistic_decoder_authority_allowed", "qec_advantage_claim_allowed", "global_correctness_claim_allowed", "silent_replacement_allowed", "baseline_mutation_allowed", "candidate_promotion_allowed")

def _all_flags_false(obj: Any, names: tuple[str, ...]) -> bool:
    return all(getattr(obj, name) is False for name in names)

def _upstream_safe(upstream: Any) -> bool:
    return (
        upstream.replay_equivalence_proven_for_declared_corpus is True
        and upstream.optimization_contract_safe is True
        and upstream.fast_path_equivalence_proven_for_declared_corpus is True
        and upstream.candidate_adapter_only is True
        and upstream.candidate_promoted is False
        and upstream.baseline_immutable is True
        and upstream.baseline_mutation_allowed is False
        and upstream.runtime_authority_allowed is False
    )

def _identity_safe(ident: Any) -> bool:
    return (
        ident.adapter_only is True
        and ident.boundary_only is True
        and ident.runtime_enabled is False
        and ident.importable_runtime_allowed is False
        and ident.implementation_authority_allowed is False
        and ident.promotion_allowed is False
        and ident.benchmark_claim_allowed is False
        and ident.speedup_claim_allowed is False
        and ident.hardware_authority_allowed is False
        and ident.qec_advantage_claim_allowed is False
    )

def _source_safe(source: Any) -> bool:
    return (
        source.source_boundary_mode == _SOURCE_BOUNDARY_MODE
        and source.source_files_exist_required is False
        and source.repository_walk_allowed is False
        and source.runtime_import_allowed is False
        and source.runtime_execution_allowed is False
        and source.implementation_file_creation_allowed is False
        and source.baseline_mutation_allowed is False
        and source.filesystem_mutation_allowed is False
    )

def _runtime_safe(runtime: Any) -> bool:
    return runtime.declared_boundary_only is True and _all_flags_false(runtime, _RUNTIME_DENIED_FLAGS)

def _config_safe(config: Any) -> bool:
    return (
        config.config_mode == _CONFIG_MODE
        and config.deterministic_config_ordering is True
        and _all_flags_false(config, _CONFIG_DENIED_FLAGS)
    )

def _build_safe(build: Any) -> bool:
    return build.build_boundary_mode == _BUILD_BOUNDARY_MODE and _all_flags_false(build, _BUILD_DENIED_FLAGS)

def _equivalence_safe(equiv: Any) -> bool:
    return (
        equiv.equivalence_required_before_runtime is True
        and equiv.implementation_valid_without_fast_path_equivalence is False
        and equiv.equivalence_mode == _EQUIVALENCE_MODE
        and equiv.fast_path_equivalence_scope == _EQUIVALENCE_SCOPE
    )

def _audit_safe(audit: Any) -> bool:
    return (
        audit.future_benchmark_ladder_required is True
        and audit.future_rollback_receipt_required is True
        and audit.future_promotion_receipt_required is True
        and audit.audit_complete is False
    )

def _rollback_safe(rollback: Any) -> bool:
    return (
        rollback.rollback_receipt_required_before_promotion is True
        and rollback.required_future_rollback_receipt_kind == _REQUIRED_ROLLBACK_KIND
        and rollback.required_future_rollback_release == _REQUIRED_ROLLBACK_RELEASE
        and rollback.promotion_blocked_without_rollback_receipt is True
    )

def _authority_safe(authority: Any) -> bool:
    return (
        authority.candidate_adapter_only is True
        and authority.boundary_only is True
        and _all_flags_false(authority, _AUTHORITY_DENIED_FLAGS)
    )

def _boundary_safe(upstream: Any, ident: Any, source: Any, runtime: Any, config: Any, build: Any, equiv: Any, audit: Any, rollback: Any, authority: Any) -> bool:
    return all((
        _upstream_safe(upstream),
        _identity_safe(ident),
        _source_safe(source),
        _runtime_safe(runtime),
        _config_safe(config),
        _build_safe(build),
        _equivalence_safe(equiv),
        _audit_safe(audit),
        _rollback_safe(rollback),
        _authority_safe(authority),
    ))
def _candidate_remains_adapter_only(upstream: Any, ident: Any, authority: Any) -> bool:
    return upstream.candidate_adapter_only is True and upstream.candidate_promoted is False and ident.adapter_only is True and authority.candidate_adapter_only is True and authority.candidate_promotion_allowed is False

def _hash_field_name(obj_or_cls: Any) -> str:
    cls = obj_or_cls if isinstance(obj_or_cls, type) else type(obj_or_cls)
    name = getattr(cls, "_HASH_FIELD", None)
    if not isinstance(name, str) or not name:
        raise _invalid_input(f"{cls.__name__}:HASH_FIELD")
    if not any(field.name == name for field in fields(cls)):
        raise _invalid_input(f"{cls.__name__}:HASH_FIELD")
    return name

def _build_dataclass(cls: type[Any], payload: Mapping[str, Any]) -> Any:
    hash_field = _hash_field_name(cls)
    h = _hash_payload(payload)
    return cls(**dict(payload), **{hash_field: h})

# payload helpers
def _payload(obj: Any) -> dict[str, Any]: return _payload_without(obj, _hash_field_name(obj))

@dataclass(frozen=True)
class DecoderImplementationUpstreamBinding:
    _HASH_FIELD: ClassVar[str] = "decoder_implementation_upstream_binding_hash"
    previous_release_tag: str; previous_release_url: str; implementation_boundary_release: str
    upstream_canonical_decoder_baseline_receipt_hash: str; upstream_decoder_candidate_manifest_hash: str; upstream_decoder_replay_equivalence_receipt_hash: str; upstream_decoder_optimization_contract_hash: str; upstream_decoder_fast_path_equivalence_receipt_hash: str
    candidate_declaration_hash: str; fast_path_identity_hash: str; candidate_name: str; candidate_version: str
    replay_equivalence_proven_for_declared_corpus: bool; optimization_contract_safe: bool; fast_path_equivalence_proven_for_declared_corpus: bool; candidate_adapter_only: bool; candidate_promoted: bool; baseline_immutable: bool; baseline_mutation_allowed: bool; runtime_authority_allowed: bool
    decoder_implementation_upstream_binding_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderImplementationUpstreamBinding: raise _invalid_input()
        if self.previous_release_tag != PREVIOUS_RELEASE_TAG: raise _invalid_input("previous_release_tag")
        if self.previous_release_url != PREVIOUS_RELEASE_URL: raise _invalid_input("previous_release_url")
        if self.implementation_boundary_release != IMPLEMENTATION_BOUNDARY_RELEASE: raise _invalid_input("implementation_boundary_release")
        for n in ("upstream_canonical_decoder_baseline_receipt_hash","upstream_decoder_candidate_manifest_hash","upstream_decoder_replay_equivalence_receipt_hash","upstream_decoder_optimization_contract_hash","upstream_decoder_fast_path_equivalence_receipt_hash","candidate_declaration_hash","fast_path_identity_hash"): _validate_hash_format(getattr(self,n), n)
        _require_text(self.candidate_name,"candidate_name"); _require_text(self.candidate_version,"candidate_version")
        _require_flags(self,{"replay_equivalence_proven_for_declared_corpus":True,"optimization_contract_safe":True,"fast_path_equivalence_proven_for_declared_corpus":True,"candidate_adapter_only":True,"candidate_promoted":False,"baseline_immutable":True,"baseline_mutation_allowed":False,"runtime_authority_allowed":False},"upstream_binding:UNSAFE")
        _assert_hash_matches(self,"decoder_implementation_upstream_binding_hash",_payload)

@dataclass(frozen=True)
class DecoderImplementationIdentity:
    _HASH_FIELD: ClassVar[str] = "decoder_implementation_identity_hash"
    implementation_id: str; implementation_name: str; implementation_version: str; implementation_kind: str; implementation_status: str; implementation_mode: str
    associated_candidate_declaration_hash: str; associated_fast_path_identity_hash: str; associated_fast_path_equivalence_receipt_hash: str
    adapter_only: bool; boundary_only: bool; runtime_enabled: bool; importable_runtime_allowed: bool; implementation_authority_allowed: bool; promotion_allowed: bool; benchmark_claim_allowed: bool; speedup_claim_allowed: bool; hardware_authority_allowed: bool; qec_advantage_claim_allowed: bool
    decoder_implementation_identity_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderImplementationIdentity: raise _invalid_input()
        for n in ("implementation_id","implementation_name","implementation_version"): _require_text(getattr(self,n), n)
        if self.implementation_kind not in IMPLEMENTATION_KINDS: raise _invalid_input("implementation_kind")
        if self.implementation_status != _IMPLEMENTATION_STATUS: raise _invalid_boundary("implementation_status")
        if self.implementation_mode != _IMPLEMENTATION_MODE: raise _invalid_boundary("implementation_mode")
        for n in ("associated_candidate_declaration_hash","associated_fast_path_identity_hash","associated_fast_path_equivalence_receipt_hash"): _validate_hash_format(getattr(self,n), n)
        _require_flags(self,{"adapter_only":True,"boundary_only":True,"runtime_enabled":False,"importable_runtime_allowed":False,"implementation_authority_allowed":False,"promotion_allowed":False,"benchmark_claim_allowed":False,"speedup_claim_allowed":False,"hardware_authority_allowed":False,"qec_advantage_claim_allowed":False},"implementation_identity:UNSAFE", boundary=True)
        _assert_hash_matches(self,"decoder_implementation_identity_hash",_payload)

@dataclass(frozen=True)
class DecoderImplementationArtifact:
    _HASH_FIELD: ClassVar[str] = "decoder_implementation_artifact_hash"
    artifact_id: str; artifact_path: str; artifact_sha256: str; artifact_role: str; artifact_mode: str; artifact_language: str; artifact_schema_hash: str
    executable_runtime_artifact: bool; import_allowed: bool; execution_allowed: bool; benchmark_allowed: bool
    decoder_implementation_artifact_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderImplementationArtifact: raise _invalid_input()
        _require_text(self.artifact_id,"artifact_id"); _validate_artifact_path(self.artifact_path); _validate_hash_format(self.artifact_sha256,"artifact_sha256")
        if self.artifact_role not in ARTIFACT_ROLES: raise _invalid_input("artifact_role")
        if self.artifact_mode != _ARTIFACT_MODE: raise _invalid_input("artifact_mode")
        if self.artifact_language not in ARTIFACT_LANGUAGES: raise _invalid_input("artifact_language")
        _validate_hash_format(self.artifact_schema_hash,"artifact_schema_hash")
        _require_flags(self,{"executable_runtime_artifact":False,"import_allowed":False,"execution_allowed":False,"benchmark_allowed":False},"artifact:UNSAFE")
        _assert_hash_matches(self,"decoder_implementation_artifact_hash",_payload)

@dataclass(frozen=True)
class DecoderImplementationSourceBoundary:
    _HASH_FIELD: ClassVar[str] = "decoder_implementation_source_boundary_hash"
    implementation_source_root: str; source_boundary_mode: str; implementation_artifacts: tuple[DecoderImplementationArtifact, ...]; implementation_artifact_count: int; source_tree_hash: str
    source_files_exist_required: bool; repository_walk_allowed: bool; runtime_import_allowed: bool; runtime_execution_allowed: bool; implementation_file_creation_allowed: bool; baseline_mutation_allowed: bool; filesystem_mutation_allowed: bool
    decoder_implementation_source_boundary_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderImplementationSourceBoundary: raise _invalid_input()
        _validate_source_root(self.implementation_source_root)
        if self.source_boundary_mode != _SOURCE_BOUNDARY_MODE: raise _invalid_boundary("source_boundary_mode")
        ordered = _sorted_artifacts(self.implementation_artifacts)
        for a in ordered:
            if not a.artifact_path.startswith(self.implementation_source_root) or a.artifact_path == self.implementation_source_root: raise _invalid_input("artifact_path:OUTSIDE_ROOT")
        _require_exact_int(self.implementation_artifact_count,"implementation_artifact_count")
        if self.implementation_artifact_count != len(ordered): raise _invalid_input("implementation_artifact_count")
        _validate_hash_format(self.source_tree_hash,"source_tree_hash")
        if self.source_tree_hash != _compute_source_tree_hash(self.source_boundary_mode, ordered): raise _hash_mismatch("source_tree_hash")
        _require_flags(self,{"source_files_exist_required":False,"repository_walk_allowed":False,"runtime_import_allowed":False,"runtime_execution_allowed":False,"implementation_file_creation_allowed":False,"baseline_mutation_allowed":False,"filesystem_mutation_allowed":False},"source_boundary:UNSAFE", boundary=True)
        if self.implementation_artifacts != ordered: raise _hash_mismatch("implementation_artifacts:ORDER")
        _assert_hash_matches(self,"decoder_implementation_source_boundary_hash",_payload)

@dataclass(frozen=True)
class DecoderImplementationRuntimeBoundary:
    _HASH_FIELD: ClassVar[str] = "decoder_implementation_runtime_boundary_hash"
    runtime_boundary_id: str; runtime_boundary_mode: str; declared_boundary_only: bool; baseline_decoder_import_allowed: bool; candidate_decoder_import_allowed: bool; fast_path_import_allowed: bool; implementation_import_allowed: bool; runtime_decoder_execution_allowed: bool; candidate_runtime_execution_allowed: bool; fast_path_runtime_execution_allowed: bool; implementation_runtime_execution_allowed: bool; replay_execution_allowed: bool; optimization_execution_allowed: bool; benchmark_execution_allowed: bool; network_allowed: bool; heavy_backend_import_allowed: bool; hardware_sdk_allowed: bool; filesystem_mutation_allowed: bool; decoder_implementation_runtime_boundary_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderImplementationRuntimeBoundary: raise _invalid_input()
        _require_text(self.runtime_boundary_id,"runtime_boundary_id")
        if self.runtime_boundary_mode != _RUNTIME_BOUNDARY_MODE: raise _invalid_boundary("runtime_boundary_mode")
        _require_flags(self,{"declared_boundary_only":True,"baseline_decoder_import_allowed":False,"candidate_decoder_import_allowed":False,"fast_path_import_allowed":False,"implementation_import_allowed":False,"runtime_decoder_execution_allowed":False,"candidate_runtime_execution_allowed":False,"fast_path_runtime_execution_allowed":False,"implementation_runtime_execution_allowed":False,"replay_execution_allowed":False,"optimization_execution_allowed":False,"benchmark_execution_allowed":False,"network_allowed":False,"heavy_backend_import_allowed":False,"hardware_sdk_allowed":False,"filesystem_mutation_allowed":False},"runtime_boundary:UNSAFE", boundary=True)
        _assert_hash_matches(self,"decoder_implementation_runtime_boundary_hash",_payload)

@dataclass(frozen=True)
class DecoderImplementationConfigBoundary:
    _HASH_FIELD: ClassVar[str] = "decoder_implementation_config_boundary_hash"
    config_boundary_id: str; config_mode: str; config_schema_hash: str; config_payload_hash: str; deterministic_config_ordering: bool; mutable_runtime_config_allowed: bool; environment_variable_dependency_allowed: bool; wall_clock_dependency_allowed: bool; randomness_dependency_allowed: bool; filesystem_probe_dependency_allowed: bool; network_dependency_allowed: bool; hardware_probe_dependency_allowed: bool; decoder_implementation_config_boundary_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderImplementationConfigBoundary: raise _invalid_input()
        _require_text(self.config_boundary_id,"config_boundary_id")
        if self.config_mode != _CONFIG_MODE: raise _invalid_boundary("config_mode")
        for n in ("config_schema_hash","config_payload_hash"): _validate_hash_format(getattr(self,n), n)
        _require_flags(self,{"deterministic_config_ordering":True,"mutable_runtime_config_allowed":False,"environment_variable_dependency_allowed":False,"wall_clock_dependency_allowed":False,"randomness_dependency_allowed":False,"filesystem_probe_dependency_allowed":False,"network_dependency_allowed":False,"hardware_probe_dependency_allowed":False},"config_boundary:UNSAFE", boundary=True)
        _assert_hash_matches(self,"decoder_implementation_config_boundary_hash",_payload)

@dataclass(frozen=True)
class DecoderImplementationBuildBoundary:
    _HASH_FIELD: ClassVar[str] = "decoder_implementation_build_boundary_hash"
    build_boundary_id: str; build_boundary_mode: str; build_manifest_hash: str; dependency_manifest_hash: str; source_boundary_hash: str; build_execution_allowed: bool; dependency_install_allowed: bool; network_resolution_allowed: bool; native_extension_build_allowed: bool; hardware_specific_build_allowed: bool; unpinned_dependency_allowed: bool; build_cache_authority_allowed: bool; decoder_implementation_build_boundary_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderImplementationBuildBoundary: raise _invalid_input()
        _require_text(self.build_boundary_id,"build_boundary_id")
        if self.build_boundary_mode != _BUILD_BOUNDARY_MODE: raise _invalid_boundary("build_boundary_mode")
        for n in ("build_manifest_hash","dependency_manifest_hash","source_boundary_hash"): _validate_hash_format(getattr(self,n), n)
        _require_flags(self,{"build_execution_allowed":False,"dependency_install_allowed":False,"network_resolution_allowed":False,"native_extension_build_allowed":False,"hardware_specific_build_allowed":False,"unpinned_dependency_allowed":False,"build_cache_authority_allowed":False},"build_boundary:UNSAFE", boundary=True)
        _assert_hash_matches(self,"decoder_implementation_build_boundary_hash",_payload)

@dataclass(frozen=True)
class DecoderImplementationEquivalenceBinding:
    _HASH_FIELD: ClassVar[str] = "decoder_implementation_equivalence_binding_hash"
    equivalence_binding_id: str; required_replay_equivalence_receipt_hash: str; required_fast_path_equivalence_receipt_hash: str; required_optimization_contract_hash: str; equivalence_mode: str; fast_path_equivalence_scope: str; declared_corpus_only: bool; output_schema_match_required: bool; output_payload_match_required: bool; canonical_ordering_match_required: bool; precision_policy: str; approximation_policy: str; equivalence_required_before_runtime: bool; implementation_valid_without_fast_path_equivalence: bool; decoder_implementation_equivalence_binding_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderImplementationEquivalenceBinding: raise _invalid_input()
        _require_text(self.equivalence_binding_id,"equivalence_binding_id")
        for n in ("required_replay_equivalence_receipt_hash","required_fast_path_equivalence_receipt_hash","required_optimization_contract_hash"): _validate_hash_format(getattr(self,n), n)
        if self.equivalence_mode != _EQUIVALENCE_MODE: raise _invalid_boundary("equivalence_mode")
        if self.fast_path_equivalence_scope != _EQUIVALENCE_SCOPE: raise _invalid_boundary("fast_path_equivalence_scope")
        if self.precision_policy != _PRECISION_POLICY: raise _invalid_boundary("precision_policy")
        if self.approximation_policy != _APPROXIMATION_POLICY: raise _invalid_boundary("approximation_policy")
        _require_flags(self,{"declared_corpus_only":True,"output_schema_match_required":True,"output_payload_match_required":True,"canonical_ordering_match_required":True,"equivalence_required_before_runtime":True,"implementation_valid_without_fast_path_equivalence":False},"equivalence_binding:UNSAFE", boundary=True)
        _assert_hash_matches(self,"decoder_implementation_equivalence_binding_hash",_payload)

@dataclass(frozen=True)
class DecoderImplementationAuditBoundary:
    _HASH_FIELD: ClassVar[str] = "decoder_implementation_audit_boundary_hash"
    audit_boundary_id: str; audit_mode: str; static_boundary_review_required: bool; source_hash_review_required: bool; no_decoder_mutation_review_required: bool; no_runtime_import_review_required: bool; no_runtime_execution_review_required: bool; no_benchmark_claim_review_required: bool; future_benchmark_ladder_required: bool; future_rollback_receipt_required: bool; future_promotion_receipt_required: bool; audit_complete: bool; decoder_implementation_audit_boundary_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderImplementationAuditBoundary: raise _invalid_input()
        _require_text(self.audit_boundary_id,"audit_boundary_id")
        if self.audit_mode != _AUDIT_MODE: raise _invalid_boundary("audit_mode")
        _require_flags(self,{"static_boundary_review_required":True,"source_hash_review_required":True,"no_decoder_mutation_review_required":True,"no_runtime_import_review_required":True,"no_runtime_execution_review_required":True,"no_benchmark_claim_review_required":True,"future_benchmark_ladder_required":True,"future_rollback_receipt_required":True,"future_promotion_receipt_required":True,"audit_complete":False},"audit_boundary:UNSAFE", boundary=True)
        _assert_hash_matches(self,"decoder_implementation_audit_boundary_hash",_payload)

@dataclass(frozen=True)
class DecoderImplementationRollbackGate:
    _HASH_FIELD: ClassVar[str] = "decoder_implementation_rollback_gate_hash"
    rollback_gate_id: str; rollback_gate_mode: str; rollback_receipt_required_before_promotion: bool; required_future_rollback_receipt_kind: str; required_future_rollback_release: str; rollback_path_deletion_allowed: bool; baseline_restore_required: bool; candidate_disable_required_on_failure: bool; promotion_blocked_without_rollback_receipt: bool; implementation_disable_required_on_failure: bool; decoder_implementation_rollback_gate_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderImplementationRollbackGate: raise _invalid_input()
        _require_text(self.rollback_gate_id,"rollback_gate_id")
        if self.rollback_gate_mode != _ROLLBACK_GATE_MODE: raise _invalid_boundary("rollback_gate_mode")
        if self.required_future_rollback_receipt_kind != _REQUIRED_ROLLBACK_KIND: raise _invalid_boundary("required_future_rollback_receipt_kind")
        if self.required_future_rollback_release != _REQUIRED_ROLLBACK_RELEASE: raise _invalid_boundary("required_future_rollback_release")
        _require_flags(self,{"rollback_receipt_required_before_promotion":True,"rollback_path_deletion_allowed":False,"baseline_restore_required":True,"candidate_disable_required_on_failure":True,"promotion_blocked_without_rollback_receipt":True,"implementation_disable_required_on_failure":True},"rollback_gate:UNSAFE", boundary=True)
        _assert_hash_matches(self,"decoder_implementation_rollback_gate_hash",_payload)

@dataclass(frozen=True)
class DecoderImplementationAuthorityBoundary:
    _HASH_FIELD: ClassVar[str] = "decoder_implementation_authority_boundary_hash"
    authority_boundary_id: str; authority_mode: str; candidate_adapter_only: bool; boundary_only: bool; runtime_authority_allowed: bool; implementation_authority_allowed: bool; benchmark_authority_allowed: bool; hardware_authority_allowed: bool; ml_decoder_authority_allowed: bool; probabilistic_decoder_authority_allowed: bool; qec_advantage_claim_allowed: bool; global_correctness_claim_allowed: bool; silent_replacement_allowed: bool; baseline_mutation_allowed: bool; candidate_promotion_allowed: bool; decoder_implementation_authority_boundary_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderImplementationAuthorityBoundary: raise _invalid_input()
        _require_text(self.authority_boundary_id,"authority_boundary_id")
        if self.authority_mode != _AUTHORITY_MODE: raise _invalid_boundary("authority_mode")
        _require_flags(self,{"candidate_adapter_only":True,"boundary_only":True,"runtime_authority_allowed":False,"implementation_authority_allowed":False,"benchmark_authority_allowed":False,"hardware_authority_allowed":False,"ml_decoder_authority_allowed":False,"probabilistic_decoder_authority_allowed":False,"qec_advantage_claim_allowed":False,"global_correctness_claim_allowed":False,"silent_replacement_allowed":False,"baseline_mutation_allowed":False,"candidate_promotion_allowed":False},"authority_boundary:UNSAFE", boundary=True)
        _assert_hash_matches(self,"decoder_implementation_authority_boundary_hash",_payload)

@dataclass(frozen=True)
class DecoderImplementationBoundaryReceipt:
    _HASH_FIELD: ClassVar[str] = "decoder_implementation_boundary_receipt_hash"
    receipt_version: str; receipt_kind: str; previous_release_tag: str; previous_release_url: str
    upstream_binding: DecoderImplementationUpstreamBinding; implementation_identity: DecoderImplementationIdentity; source_boundary: DecoderImplementationSourceBoundary; runtime_boundary: DecoderImplementationRuntimeBoundary; config_boundary: DecoderImplementationConfigBoundary; build_boundary: DecoderImplementationBuildBoundary; equivalence_binding: DecoderImplementationEquivalenceBinding; audit_boundary: DecoderImplementationAuditBoundary; rollback_gate: DecoderImplementationRollbackGate; authority_boundary: DecoderImplementationAuthorityBoundary
    implementation_artifact_count: int; implementation_boundary_safe: bool; candidate_remains_adapter_only: bool; runtime_enabled: bool; implementation_authority_allowed: bool; benchmark_claim_allowed: bool; speedup_claim_allowed: bool; promotion_allowed: bool; global_correctness_claim_allowed: bool
    decoder_implementation_boundary_receipt_hash: str
    def __post_init__(self) -> None:
        if type(self) is not DecoderImplementationBoundaryReceipt: raise _invalid_input()
        if self.receipt_version != IMPLEMENTATION_BOUNDARY_RELEASE: raise _invalid_input("receipt_version")
        if self.receipt_kind != RECEIPT_KIND: raise _invalid_input("receipt_kind")
        if self.previous_release_tag != PREVIOUS_RELEASE_TAG: raise _invalid_input("previous_release_tag")
        if self.previous_release_url != PREVIOUS_RELEASE_URL: raise _invalid_input("previous_release_url")
        for value, cls in ((self.upstream_binding,DecoderImplementationUpstreamBinding),(self.implementation_identity,DecoderImplementationIdentity),(self.source_boundary,DecoderImplementationSourceBoundary),(self.runtime_boundary,DecoderImplementationRuntimeBoundary),(self.config_boundary,DecoderImplementationConfigBoundary),(self.build_boundary,DecoderImplementationBuildBoundary),(self.equivalence_binding,DecoderImplementationEquivalenceBinding),(self.audit_boundary,DecoderImplementationAuditBoundary),(self.rollback_gate,DecoderImplementationRollbackGate),(self.authority_boundary,DecoderImplementationAuthorityBoundary)):
            _revalidate_exact_instance(value, cls)
        _require_exact_int(self.implementation_artifact_count,"implementation_artifact_count")
        if self.implementation_artifact_count != self.source_boundary.implementation_artifact_count: raise _invalid_input("implementation_artifact_count")
        if self.implementation_identity.associated_candidate_declaration_hash != self.upstream_binding.candidate_declaration_hash: raise _invalid_boundary("candidate_declaration_hash:MISMATCH")
        if self.implementation_identity.associated_fast_path_identity_hash != self.upstream_binding.fast_path_identity_hash: raise _invalid_boundary("fast_path_identity_hash:MISMATCH")
        if self.implementation_identity.associated_fast_path_equivalence_receipt_hash != self.upstream_binding.upstream_decoder_fast_path_equivalence_receipt_hash: raise _invalid_boundary("fast_path_equivalence_hash:MISMATCH")
        if self.build_boundary.source_boundary_hash != self.source_boundary.decoder_implementation_source_boundary_hash: raise _invalid_boundary("source_boundary_hash:MISMATCH")
        if self.equivalence_binding.required_replay_equivalence_receipt_hash != self.upstream_binding.upstream_decoder_replay_equivalence_receipt_hash: raise _invalid_boundary("replay_equivalence_hash:MISMATCH")
        if self.equivalence_binding.required_fast_path_equivalence_receipt_hash != self.upstream_binding.upstream_decoder_fast_path_equivalence_receipt_hash: raise _invalid_boundary("fast_path_equivalence_hash:MISMATCH")
        if self.equivalence_binding.required_optimization_contract_hash != self.upstream_binding.upstream_decoder_optimization_contract_hash: raise _invalid_boundary("optimization_contract_hash:MISMATCH")
        safe = _boundary_safe(self.upstream_binding,self.implementation_identity,self.source_boundary,self.runtime_boundary,self.config_boundary,self.build_boundary,self.equivalence_binding,self.audit_boundary,self.rollback_gate,self.authority_boundary)
        adapter = _candidate_remains_adapter_only(self.upstream_binding,self.implementation_identity,self.authority_boundary)
        _require_exact_bool(self.implementation_boundary_safe,"implementation_boundary_safe"); _require_exact_bool(self.candidate_remains_adapter_only,"candidate_remains_adapter_only")
        if self.implementation_boundary_safe is not safe or self.implementation_boundary_safe is not True: raise _invalid_boundary("implementation_boundary_safe")
        if self.candidate_remains_adapter_only is not adapter or self.candidate_remains_adapter_only is not True: raise _invalid_boundary("candidate_remains_adapter_only")
        _require_flags(self,{"runtime_enabled":False,"implementation_authority_allowed":False,"benchmark_claim_allowed":False,"speedup_claim_allowed":False,"promotion_allowed":False,"global_correctness_claim_allowed":False},"receipt:UNSAFE", boundary=True)
        _assert_hash_matches(self,"decoder_implementation_boundary_receipt_hash",_payload)

# builders
def build_decoder_implementation_upstream_binding(upstream_canonical_decoder_baseline_receipt_hash: str="a"*64, upstream_decoder_candidate_manifest_hash: str="b"*64, upstream_decoder_replay_equivalence_receipt_hash: str="c"*64, upstream_decoder_optimization_contract_hash: str="d"*64, upstream_decoder_fast_path_equivalence_receipt_hash: str="e"*64, candidate_declaration_hash: str="f"*64, fast_path_identity_hash: str="1"*64, candidate_name: str="declared candidate", candidate_version: str="1", replay_equivalence_proven_for_declared_corpus: bool=True, optimization_contract_safe: bool=True, fast_path_equivalence_proven_for_declared_corpus: bool=True, candidate_adapter_only: bool=True, candidate_promoted: bool=False, baseline_immutable: bool=True, baseline_mutation_allowed: bool=False, runtime_authority_allowed: bool=False, *, previous_release_tag: str=PREVIOUS_RELEASE_TAG, previous_release_url: str=PREVIOUS_RELEASE_URL, implementation_boundary_release: str=IMPLEMENTATION_BOUNDARY_RELEASE) -> DecoderImplementationUpstreamBinding:
    p = {"previous_release_tag": previous_release_tag, "previous_release_url": previous_release_url, "implementation_boundary_release": implementation_boundary_release, "upstream_canonical_decoder_baseline_receipt_hash": upstream_canonical_decoder_baseline_receipt_hash, "upstream_decoder_candidate_manifest_hash": upstream_decoder_candidate_manifest_hash, "upstream_decoder_replay_equivalence_receipt_hash": upstream_decoder_replay_equivalence_receipt_hash, "upstream_decoder_optimization_contract_hash": upstream_decoder_optimization_contract_hash, "upstream_decoder_fast_path_equivalence_receipt_hash": upstream_decoder_fast_path_equivalence_receipt_hash, "candidate_declaration_hash": candidate_declaration_hash, "fast_path_identity_hash": fast_path_identity_hash, "candidate_name": candidate_name, "candidate_version": candidate_version, "replay_equivalence_proven_for_declared_corpus": replay_equivalence_proven_for_declared_corpus, "optimization_contract_safe": optimization_contract_safe, "fast_path_equivalence_proven_for_declared_corpus": fast_path_equivalence_proven_for_declared_corpus, "candidate_adapter_only": candidate_adapter_only, "candidate_promoted": candidate_promoted, "baseline_immutable": baseline_immutable, "baseline_mutation_allowed": baseline_mutation_allowed, "runtime_authority_allowed": runtime_authority_allowed}
    return _build_dataclass(DecoderImplementationUpstreamBinding, p)
def build_decoder_implementation_identity(implementation_id: str="declared_decoder_implementation_boundary", implementation_name: str="Declared decoder implementation boundary", implementation_version: str="1", implementation_kind: str="DECLARED_DECODER_IMPLEMENTATION_BOUNDARY", associated_candidate_declaration_hash: str="f"*64, associated_fast_path_identity_hash: str="1"*64, associated_fast_path_equivalence_receipt_hash: str="e"*64, adapter_only: bool=True, boundary_only: bool=True, runtime_enabled: bool=False, importable_runtime_allowed: bool=False, implementation_authority_allowed: bool=False, promotion_allowed: bool=False, benchmark_claim_allowed: bool=False, speedup_claim_allowed: bool=False, hardware_authority_allowed: bool=False, qec_advantage_claim_allowed: bool=False, *, implementation_status: str=_IMPLEMENTATION_STATUS, implementation_mode: str=_IMPLEMENTATION_MODE) -> DecoderImplementationIdentity:
    p = {"implementation_id": implementation_id, "implementation_name": implementation_name, "implementation_version": implementation_version, "implementation_kind": implementation_kind, "implementation_status": implementation_status, "implementation_mode": implementation_mode, "associated_candidate_declaration_hash": associated_candidate_declaration_hash, "associated_fast_path_identity_hash": associated_fast_path_identity_hash, "associated_fast_path_equivalence_receipt_hash": associated_fast_path_equivalence_receipt_hash, "adapter_only": adapter_only, "boundary_only": boundary_only, "runtime_enabled": runtime_enabled, "importable_runtime_allowed": importable_runtime_allowed, "implementation_authority_allowed": implementation_authority_allowed, "promotion_allowed": promotion_allowed, "benchmark_claim_allowed": benchmark_claim_allowed, "speedup_claim_allowed": speedup_claim_allowed, "hardware_authority_allowed": hardware_authority_allowed, "qec_advantage_claim_allowed": qec_advantage_claim_allowed}
    return _build_dataclass(DecoderImplementationIdentity, p)
def build_decoder_implementation_artifact(artifact_id: str="artifact", artifact_path: str="implementation_boundaries/artifact.json", artifact_sha256: str="a"*64, artifact_role: str="IMPLEMENTATION_BOUNDARY_DECLARATION", artifact_language: str="JSON_DECLARATION", artifact_schema_hash: str="b"*64, executable_runtime_artifact: bool=False, import_allowed: bool=False, execution_allowed: bool=False, benchmark_allowed: bool=False, *, artifact_mode: str=_ARTIFACT_MODE) -> DecoderImplementationArtifact:
    p = {"artifact_id": artifact_id, "artifact_path": artifact_path, "artifact_sha256": artifact_sha256, "artifact_role": artifact_role, "artifact_mode": artifact_mode, "artifact_language": artifact_language, "artifact_schema_hash": artifact_schema_hash, "executable_runtime_artifact": executable_runtime_artifact, "import_allowed": import_allowed, "execution_allowed": execution_allowed, "benchmark_allowed": benchmark_allowed}
    return _build_dataclass(DecoderImplementationArtifact, p)
def build_decoder_implementation_source_boundary(implementation_artifacts: Sequence[DecoderImplementationArtifact], implementation_source_root: str="implementation_boundaries/", source_files_exist_required: bool=False, repository_walk_allowed: bool=False, runtime_import_allowed: bool=False, runtime_execution_allowed: bool=False, implementation_file_creation_allowed: bool=False, baseline_mutation_allowed: bool=False, filesystem_mutation_allowed: bool=False, *, source_boundary_mode: str=_SOURCE_BOUNDARY_MODE, implementation_artifact_count: int|None=None, source_tree_hash: str|None=None) -> DecoderImplementationSourceBoundary:
    arts=_sorted_artifacts(tuple(implementation_artifacts)); count=len(arts) if implementation_artifact_count is None else implementation_artifact_count; sth=_compute_source_tree_hash(source_boundary_mode, arts) if source_tree_hash is None else source_tree_hash
    p={"implementation_source_root":implementation_source_root,"source_boundary_mode":source_boundary_mode,"implementation_artifacts":arts,"implementation_artifact_count":count,"source_tree_hash":sth,"source_files_exist_required":source_files_exist_required,"repository_walk_allowed":repository_walk_allowed,"runtime_import_allowed":runtime_import_allowed,"runtime_execution_allowed":runtime_execution_allowed,"implementation_file_creation_allowed":implementation_file_creation_allowed,"baseline_mutation_allowed":baseline_mutation_allowed,"filesystem_mutation_allowed":filesystem_mutation_allowed}
    return _build_dataclass(DecoderImplementationSourceBoundary, p)
def build_decoder_implementation_runtime_boundary(runtime_boundary_id: str="runtime-boundary", runtime_boundary_mode: str=_RUNTIME_BOUNDARY_MODE, declared_boundary_only: bool=True, **kwargs: Any) -> DecoderImplementationRuntimeBoundary:
    flags={"baseline_decoder_import_allowed":False,"candidate_decoder_import_allowed":False,"fast_path_import_allowed":False,"implementation_import_allowed":False,"runtime_decoder_execution_allowed":False,"candidate_runtime_execution_allowed":False,"fast_path_runtime_execution_allowed":False,"implementation_runtime_execution_allowed":False,"replay_execution_allowed":False,"optimization_execution_allowed":False,"benchmark_execution_allowed":False,"network_allowed":False,"heavy_backend_import_allowed":False,"hardware_sdk_allowed":False,"filesystem_mutation_allowed":False}; flags.update(kwargs); p={"runtime_boundary_id":runtime_boundary_id,"runtime_boundary_mode":runtime_boundary_mode,"declared_boundary_only":declared_boundary_only,**flags}; return _build_dataclass(DecoderImplementationRuntimeBoundary, p)
def build_decoder_implementation_config_boundary(config_boundary_id: str="config-boundary", config_schema_hash: str="a"*64, config_payload_hash: str="b"*64, config_mode: str=_CONFIG_MODE, deterministic_config_ordering: bool=True, **kwargs: Any) -> DecoderImplementationConfigBoundary:
    flags={"mutable_runtime_config_allowed":False,"environment_variable_dependency_allowed":False,"wall_clock_dependency_allowed":False,"randomness_dependency_allowed":False,"filesystem_probe_dependency_allowed":False,"network_dependency_allowed":False,"hardware_probe_dependency_allowed":False}; flags.update(kwargs); p={"config_boundary_id":config_boundary_id,"config_mode":config_mode,"config_schema_hash":config_schema_hash,"config_payload_hash":config_payload_hash,"deterministic_config_ordering":deterministic_config_ordering,**flags}; return _build_dataclass(DecoderImplementationConfigBoundary, p)
def build_decoder_implementation_build_boundary(source_boundary_hash: str="3"*64, build_boundary_id: str="build-boundary", build_manifest_hash: str="c"*64, dependency_manifest_hash: str="d"*64, build_boundary_mode: str=_BUILD_BOUNDARY_MODE, **kwargs: Any) -> DecoderImplementationBuildBoundary:
    flags={"build_execution_allowed":False,"dependency_install_allowed":False,"network_resolution_allowed":False,"native_extension_build_allowed":False,"hardware_specific_build_allowed":False,"unpinned_dependency_allowed":False,"build_cache_authority_allowed":False}; flags.update(kwargs); p={"build_boundary_id":build_boundary_id,"build_boundary_mode":build_boundary_mode,"build_manifest_hash":build_manifest_hash,"dependency_manifest_hash":dependency_manifest_hash,"source_boundary_hash":source_boundary_hash,**flags}; return _build_dataclass(DecoderImplementationBuildBoundary, p)
def build_decoder_implementation_equivalence_binding(required_replay_equivalence_receipt_hash: str="c"*64, required_fast_path_equivalence_receipt_hash: str="e"*64, required_optimization_contract_hash: str="d"*64, equivalence_binding_id: str="equivalence-binding", equivalence_mode: str=_EQUIVALENCE_MODE, fast_path_equivalence_scope: str=_EQUIVALENCE_SCOPE, precision_policy: str=_PRECISION_POLICY, approximation_policy: str=_APPROXIMATION_POLICY, **kwargs: Any) -> DecoderImplementationEquivalenceBinding:
    flags={"declared_corpus_only":True,"output_schema_match_required":True,"output_payload_match_required":True,"canonical_ordering_match_required":True,"equivalence_required_before_runtime":True,"implementation_valid_without_fast_path_equivalence":False}; flags.update(kwargs); p={"equivalence_binding_id":equivalence_binding_id,"required_replay_equivalence_receipt_hash":required_replay_equivalence_receipt_hash,"required_fast_path_equivalence_receipt_hash":required_fast_path_equivalence_receipt_hash,"required_optimization_contract_hash":required_optimization_contract_hash,"equivalence_mode":equivalence_mode,"fast_path_equivalence_scope":fast_path_equivalence_scope,"precision_policy":precision_policy,"approximation_policy":approximation_policy,**flags}; return _build_dataclass(DecoderImplementationEquivalenceBinding, p)
def build_decoder_implementation_audit_boundary(audit_boundary_id: str="audit-boundary", audit_mode: str=_AUDIT_MODE, **kwargs: Any) -> DecoderImplementationAuditBoundary:
    flags={"static_boundary_review_required":True,"source_hash_review_required":True,"no_decoder_mutation_review_required":True,"no_runtime_import_review_required":True,"no_runtime_execution_review_required":True,"no_benchmark_claim_review_required":True,"future_benchmark_ladder_required":True,"future_rollback_receipt_required":True,"future_promotion_receipt_required":True,"audit_complete":False}; flags.update(kwargs); p={"audit_boundary_id":audit_boundary_id,"audit_mode":audit_mode,**flags}; return _build_dataclass(DecoderImplementationAuditBoundary, p)
def build_decoder_implementation_rollback_gate(rollback_gate_id: str="rollback-gate", rollback_gate_mode: str=_ROLLBACK_GATE_MODE, required_future_rollback_receipt_kind: str=_REQUIRED_ROLLBACK_KIND, required_future_rollback_release: str=_REQUIRED_ROLLBACK_RELEASE, **kwargs: Any) -> DecoderImplementationRollbackGate:
    flags={"rollback_receipt_required_before_promotion":True,"rollback_path_deletion_allowed":False,"baseline_restore_required":True,"candidate_disable_required_on_failure":True,"promotion_blocked_without_rollback_receipt":True,"implementation_disable_required_on_failure":True}; flags.update(kwargs); p={"rollback_gate_id":rollback_gate_id,"rollback_gate_mode":rollback_gate_mode,"required_future_rollback_receipt_kind":required_future_rollback_receipt_kind,"required_future_rollback_release":required_future_rollback_release,**flags}; return _build_dataclass(DecoderImplementationRollbackGate, p)
def build_decoder_implementation_authority_boundary(authority_boundary_id: str="authority-boundary", authority_mode: str=_AUTHORITY_MODE, **kwargs: Any) -> DecoderImplementationAuthorityBoundary:
    flags={"candidate_adapter_only":True,"boundary_only":True,"runtime_authority_allowed":False,"implementation_authority_allowed":False,"benchmark_authority_allowed":False,"hardware_authority_allowed":False,"ml_decoder_authority_allowed":False,"probabilistic_decoder_authority_allowed":False,"qec_advantage_claim_allowed":False,"global_correctness_claim_allowed":False,"silent_replacement_allowed":False,"baseline_mutation_allowed":False,"candidate_promotion_allowed":False}; flags.update(kwargs); p={"authority_boundary_id":authority_boundary_id,"authority_mode":authority_mode,**flags}; return _build_dataclass(DecoderImplementationAuthorityBoundary, p)
def build_decoder_implementation_boundary_receipt(upstream_binding: DecoderImplementationUpstreamBinding, implementation_identity: DecoderImplementationIdentity, source_boundary: DecoderImplementationSourceBoundary, runtime_boundary: DecoderImplementationRuntimeBoundary, config_boundary: DecoderImplementationConfigBoundary, build_boundary: DecoderImplementationBuildBoundary, equivalence_binding: DecoderImplementationEquivalenceBinding, audit_boundary: DecoderImplementationAuditBoundary, rollback_gate: DecoderImplementationRollbackGate, authority_boundary: DecoderImplementationAuthorityBoundary, *, receipt_version: str=IMPLEMENTATION_BOUNDARY_RELEASE, receipt_kind: str=RECEIPT_KIND, previous_release_tag: str=PREVIOUS_RELEASE_TAG, previous_release_url: str=PREVIOUS_RELEASE_URL, implementation_artifact_count: int|None=None, implementation_boundary_safe: bool|None=None, candidate_remains_adapter_only: bool|None=None, runtime_enabled: bool=False, implementation_authority_allowed: bool=False, benchmark_claim_allowed: bool=False, speedup_claim_allowed: bool=False, promotion_allowed: bool=False, global_correctness_claim_allowed: bool=False) -> DecoderImplementationBoundaryReceipt:
    count=source_boundary.implementation_artifact_count if implementation_artifact_count is None else implementation_artifact_count; safe=_boundary_safe(upstream_binding,implementation_identity,source_boundary,runtime_boundary,config_boundary,build_boundary,equivalence_binding,audit_boundary,rollback_gate,authority_boundary) if implementation_boundary_safe is None else implementation_boundary_safe; adapter=_candidate_remains_adapter_only(upstream_binding,implementation_identity,authority_boundary) if candidate_remains_adapter_only is None else candidate_remains_adapter_only
    p={"receipt_version":receipt_version,"receipt_kind":receipt_kind,"previous_release_tag":previous_release_tag,"previous_release_url":previous_release_url,"upstream_binding":upstream_binding,"implementation_identity":implementation_identity,"source_boundary":source_boundary,"runtime_boundary":runtime_boundary,"config_boundary":config_boundary,"build_boundary":build_boundary,"equivalence_binding":equivalence_binding,"audit_boundary":audit_boundary,"rollback_gate":rollback_gate,"authority_boundary":authority_boundary,"implementation_artifact_count":count,"implementation_boundary_safe":safe,"candidate_remains_adapter_only":adapter,"runtime_enabled":runtime_enabled,"implementation_authority_allowed":implementation_authority_allowed,"benchmark_claim_allowed":benchmark_claim_allowed,"speedup_claim_allowed":speedup_claim_allowed,"promotion_allowed":promotion_allowed,"global_correctness_claim_allowed":global_correctness_claim_allowed}
    return _build_dataclass(DecoderImplementationBoundaryReceipt, p)

# validators
def validate_decoder_implementation_upstream_binding(value: DecoderImplementationUpstreamBinding) -> DecoderImplementationUpstreamBinding: _revalidate_exact_instance(value,DecoderImplementationUpstreamBinding); return value
def validate_decoder_implementation_identity(value: DecoderImplementationIdentity) -> DecoderImplementationIdentity: _revalidate_exact_instance(value,DecoderImplementationIdentity); return value
def validate_decoder_implementation_artifact(value: DecoderImplementationArtifact) -> DecoderImplementationArtifact: _revalidate_exact_instance(value,DecoderImplementationArtifact); return value
def validate_decoder_implementation_source_boundary(value: DecoderImplementationSourceBoundary) -> DecoderImplementationSourceBoundary: _revalidate_exact_instance(value,DecoderImplementationSourceBoundary); return value
def validate_decoder_implementation_runtime_boundary(value: DecoderImplementationRuntimeBoundary) -> DecoderImplementationRuntimeBoundary: _revalidate_exact_instance(value,DecoderImplementationRuntimeBoundary); return value
def validate_decoder_implementation_config_boundary(value: DecoderImplementationConfigBoundary) -> DecoderImplementationConfigBoundary: _revalidate_exact_instance(value,DecoderImplementationConfigBoundary); return value
def validate_decoder_implementation_build_boundary(value: DecoderImplementationBuildBoundary) -> DecoderImplementationBuildBoundary: _revalidate_exact_instance(value,DecoderImplementationBuildBoundary); return value
def validate_decoder_implementation_equivalence_binding(value: DecoderImplementationEquivalenceBinding) -> DecoderImplementationEquivalenceBinding: _revalidate_exact_instance(value,DecoderImplementationEquivalenceBinding); return value
def validate_decoder_implementation_audit_boundary(value: DecoderImplementationAuditBoundary) -> DecoderImplementationAuditBoundary: _revalidate_exact_instance(value,DecoderImplementationAuditBoundary); return value
def validate_decoder_implementation_rollback_gate(value: DecoderImplementationRollbackGate) -> DecoderImplementationRollbackGate: _revalidate_exact_instance(value,DecoderImplementationRollbackGate); return value
def validate_decoder_implementation_authority_boundary(value: DecoderImplementationAuthorityBoundary) -> DecoderImplementationAuthorityBoundary: _revalidate_exact_instance(value,DecoderImplementationAuthorityBoundary); return value
def validate_decoder_implementation_boundary_receipt(value: DecoderImplementationBoundaryReceipt) -> DecoderImplementationBoundaryReceipt: _revalidate_exact_instance(value,DecoderImplementationBoundaryReceipt); return value
