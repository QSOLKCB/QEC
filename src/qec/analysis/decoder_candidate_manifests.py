from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Sequence

from qec.analysis._decoder_candidate_manifest_utils import (
    DecoderCandidateManifestError,
    DecoderCandidateManifestErrorCode,
    _assert_hash_matches,
    _base_payload,
    _canonical_json,
    _hash_payload,
    _HASH_RE,
    _hash_mismatch,
    _invalid_candidate,
    _invalid_input,
    _policy_flags_satisfied,
    _require_exact_bool,
    _require_policy_flags,
    _revalidate_exact_instance,
    _validate_hash_format,
)

MANIFEST_VERSION = "v166.1"
MANIFEST_KIND = "DecoderCandidateManifest"
PREVIOUS_RELEASE_TAG = "v166.0"
PREVIOUS_RELEASE_URL = "https://github.com/QSOLKCB/QEC/releases/tag/v166.0"

_CANDIDATE_KINDS = frozenset(
    {
        "CANDIDATE_DECODER_BACKEND",
        "ADAPTER_IMPLEMENTATION_CANDIDATE",
        "FAST_PATH_CANDIDATE",
        "REPLAY_BOUND_OPTIMIZATION_LAYER",
    }
)
_CANDIDATE_STATUS = "ADAPTER_ONLY_CANDIDATE"
_SOURCE_ROLE = "CANDIDATE_DECODER_SOURCE_DECLARATION"
_SOURCE_BOUNDARY_MODE = "DECLARED_CANDIDATE_SOURCE_HASH_BOUND"
_ALLOWED_CANDIDATE_SOURCE_ROOTS = (
    "candidate_decoders/",
    "external/decoder_candidates/",
    "src/qec/analysis/decoder_candidates/",
)
_CAPABILITY_MODE = "HYPOTHESIS_ONLY"
_CAPABILITIES = frozenset(
    {
        "SPARSE_HANDLING_HYPOTHESIS",
        "MEMORY_EFFICIENCY_HYPOTHESIS",
        "GRAPH_CONSTRUCTION_HYPOTHESIS",
        "CONVERGENCE_BEHAVIOR_HYPOTHESIS",
        "SCALING_BEHAVIOR_HYPOTHESIS",
        "FAST_PATH_HYPOTHESIS",
        "HARDWARE_UTILIZATION_HYPOTHESIS",
    }
)
_RUNTIME_BOUNDARY_MODE = "NO_RUNTIME_DECODER_EXECUTION"
_REQUIRED_FUTURE_RECEIPT_KIND = "DecoderReplayEquivalenceReceipt"
_REQUIRED_FUTURE_RELEASE = "v166.2"
_EQUIVALENCE_MODE = "EXACT_CANONICAL_OUTPUT_MATCH"
_PRECISION_POLICY = "DECLARED_EXACT_NO_HIDDEN_PRECISION_DRIFT"
_APPROXIMATION_POLICY = "NO_UNDECLARED_APPROXIMATION"
_PROMOTION_STATUS = "NOT_PROMOTED"
_DECODER_FORBIDDEN_ROOT = "src/qec/decoder"

_MAX_TEXT_LENGTH = 512

_FORBIDDEN_DECLARATION_TOKENS = (
    "silent decoder replacement",
    "candidate replaces baseline",
    "decoder replaced because faster",
    "speed proves correctness",
    "benchmark proves correctness",
    "benchmark marketing",
    "runtime promotion",
    "candidate decoder promoted",
    "candidate decoder authority",
    "probabilistic decoder authority",
    "probabilistic decoder promotion",
    "ml decoder authority",
    "hardware authority",
    "qec advantage proven",
    "mutation of canonical decoder",
    "deleting rollback path",
    "hidden precision drift",
    "undeclared approximation policy",
    "equivalence already proven",
    "output accepted as canonical",
    "exact correctness proven",
)
_SEMANTIC_GUARD_EXACT_ALLOWLIST = {
    _PRECISION_POLICY,
    _APPROXIMATION_POLICY,
}


_IDENTITY_REPLAY_POLICY = {
    "adapter_only": True,
    "promoted": False,
    "canonical_baseline_replacement": False,
    "runtime_authority_allowed": False,
}
_SOURCE_BOUNDARY_REPLAY_POLICY = {
    "candidate_import_allowed": False,
    "candidate_runtime_execution_allowed": False,
    "baseline_decoder_mutation_allowed": False,
    "filesystem_mutation_allowed": False,
}
_CAPABILITY_REPLAY_POLICY = {
    "performance_claim_allowed": False,
    "correctness_claim_allowed": False,
    "benchmark_claim_allowed": False,
    "qec_advantage_claim_allowed": False,
    "hardware_authority_allowed": False,
    "exact_equivalence_claimed": False,
}
_RUNTIME_REPLAY_POLICY = {
    "baseline_decoder_import_allowed": False,
    "candidate_import_allowed": False,
    "decoder_workload_execution_allowed": False,
    "benchmark_execution_allowed": False,
    "network_allowed": False,
    "heavy_backend_import_allowed": False,
    "hardware_sdk_allowed": False,
}
_EQUIVALENCE_REPLAY_POLICY = {
    "equivalence_required_before_promotion": True,
    "replay_corpus_required": True,
    "output_schema_match_required": True,
    "equivalence_proven": False,
    "candidate_output_authority_allowed": False,
}
_PROMOTION_REPLAY_POLICY = {
    "promotion_allowed_in_this_release": False,
    "runtime_authority_allowed": False,
    "silent_replacement_allowed": False,
    "probabilistic_promotion_allowed": False,
    "ml_decoder_authority_allowed": False,
    "benchmark_marketing_allowed": False,
    "rollback_receipt_required_before_promotion": True,
    "benchmark_ladder_required_before_performance_claims": True,
    "baseline_mutation_allowed": False,
}


def _normalize_semantics_text(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"\\[nrt]", " ", lowered)
    lowered = lowered.replace("_", " ").replace("-", " ")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def _check_forbidden_declaration_semantics(
    value: Any, field_name: str = "text"
) -> None:
    if not isinstance(value, str) or value in _SEMANTIC_GUARD_EXACT_ALLOWLIST:
        return
    normalized = _normalize_semantics_text(value)
    for token in _FORBIDDEN_DECLARATION_TOKENS:
        normalized_token = _normalize_semantics_text(token)
        if normalized_token in normalized:
            raise _invalid_input(
                f"{field_name}:FORBIDDEN_DECLARATION:{normalized_token.replace(' ', '_')}"
            )


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT_LENGTH:
        raise _invalid_input(f"{field_name}:TEXT")
    _check_forbidden_declaration_semantics(value, field_name)


def _validate_posix_relative_path(
    path: str, field_name: str, *, require_trailing_slash: bool = False
) -> None:
    if not isinstance(path, str) or not path:
        raise _invalid_input(f"{field_name}:TEXT")
    _check_forbidden_declaration_semantics(path, field_name)
    if "\\" in path:
        raise _invalid_input(f"{field_name}:BACKSLASH")
    if path.startswith("/"):
        raise _invalid_input(f"{field_name}:ABSOLUTE")
    if "//" in path:
        raise _invalid_input(f"{field_name}:DOUBLE_SLASH")
    if require_trailing_slash and not path.endswith("/"):
        raise _invalid_input(f"{field_name}:TRAILING_SLASH")
    parts = path.split("/")
    components = parts[:-1] if require_trailing_slash and parts[-1] == "" else parts
    if any(part in {"", ".", ".."} for part in components):
        raise _invalid_input(f"{field_name}:NON_CANONICAL_COMPONENT")
    canonical = PurePosixPath(path.rstrip("/")).as_posix() + (
        "/" if path.endswith("/") else ""
    )
    if canonical != path:
        raise _invalid_input(f"{field_name}:NON_CANONICAL")


def _validate_candidate_source_path(path: str) -> None:
    _validate_posix_relative_path(path, "candidate_source_path")
    if path == _DECODER_FORBIDDEN_ROOT or path.startswith(
        f"{_DECODER_FORBIDDEN_ROOT}/"
    ):
        raise _invalid_input("candidate_source_path:DECODER_ROOT_FORBIDDEN")
    if not any(
        path.startswith(root) and path != root
        for root in _ALLOWED_CANDIDATE_SOURCE_ROOTS
    ):
        raise _invalid_input("candidate_source_path:OUTSIDE_CANDIDATE_SOURCE_ROOT")


def _validate_candidate_source_root(root: str) -> None:
    _validate_posix_relative_path(
        root, "candidate_source_root", require_trailing_slash=True
    )
    if root == f"{_DECODER_FORBIDDEN_ROOT}/" or root.startswith(
        f"{_DECODER_FORBIDDEN_ROOT}/"
    ):
        raise _invalid_input("candidate_source_root:DECODER_ROOT_FORBIDDEN")
    if root not in _ALLOWED_CANDIDATE_SOURCE_ROOTS:
        raise _invalid_input("candidate_source_root:UNDECLARED_ROOT")


def _ordered_source_files(
    source_files: Sequence["DecoderCandidateSourceFile"],
) -> tuple["DecoderCandidateSourceFile", ...]:
    if not isinstance(source_files, (tuple, list)):
        raise _invalid_input("source_files:SEQUENCE")
    ordered = tuple(
        sorted(
            source_files,
            key=lambda item: (
                item.path if type(item) is DecoderCandidateSourceFile else ""
            ),
        )
    )
    if not ordered:
        raise _invalid_input("source_files:EMPTY")
    seen: set[str] = set()
    for source_file in ordered:
        _revalidate_exact_instance(source_file, DecoderCandidateSourceFile)
        if source_file.path in seen:
            raise _invalid_input("source_files:DUPLICATE_PATH")
        seen.add(source_file.path)
    return ordered


def _ordered_capabilities(capabilities: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(capabilities, (tuple, list)):
        raise _invalid_input("declared_capabilities:SEQUENCE")
    ordered = tuple(sorted(capabilities))
    if not ordered:
        raise _invalid_input("declared_capabilities:EMPTY")
    if len(set(ordered)) != len(ordered):
        raise _invalid_input("declared_capabilities:DUPLICATE")
    for capability in ordered:
        _require_text(capability, "declared_capability")
        if capability not in _CAPABILITIES:
            raise _invalid_input("declared_capability:UNKNOWN")
    return ordered


def _ordered_declarations(
    declarations: Sequence["DecoderCandidateDeclaration"],
) -> tuple["DecoderCandidateDeclaration", ...]:
    if not isinstance(declarations, (tuple, list)):
        raise _invalid_input("candidate_declarations:SEQUENCE")
    ordered = tuple(
        sorted(
            declarations,
            key=lambda item: (
                (
                    item.identity.candidate_name,
                    item.identity.candidate_version,
                    item.decoder_candidate_declaration_hash,
                )
                if type(item) is DecoderCandidateDeclaration
                else ("", "", "")
            ),
        )
    )
    if not ordered:
        raise _invalid_input("candidate_declarations:EMPTY")
    seen: set[tuple[str, str]] = set()
    for declaration in ordered:
        _revalidate_exact_instance(declaration, DecoderCandidateDeclaration)
        key = (
            declaration.identity.candidate_name,
            declaration.identity.candidate_version,
        )
        if key in seen:
            raise _invalid_input("candidate_declarations:DUPLICATE_CANDIDATE")
        seen.add(key)
    return ordered


def _source_file_payload(obj: "DecoderCandidateSourceFile") -> dict[str, Any]:
    return {"path": obj.path, "sha256": obj.sha256, "source_role": obj.source_role}


def _source_tree_payload(
    source_files: Sequence["DecoderCandidateSourceFile"],
) -> dict[str, Any]:
    return {
        "source_files": [
            _source_file_payload(source_file)
            for source_file in _ordered_source_files(source_files)
        ]
    }


def _compute_source_tree_hash(
    source_files: Sequence["DecoderCandidateSourceFile"],
) -> str:
    return _hash_payload(_source_tree_payload(source_files))


def _identity_payload(obj: "DecoderCandidateIdentity") -> dict[str, Any]:
    return _base_payload(obj, "decoder_candidate_identity_hash")


def _source_boundary_payload(obj: "DecoderCandidateSourceBoundary") -> dict[str, Any]:
    return _base_payload(obj, "decoder_candidate_source_boundary_hash")


def _capability_declaration_payload(
    obj: "DecoderCandidateCapabilityDeclaration",
) -> dict[str, Any]:
    return _base_payload(obj, "decoder_candidate_capability_declaration_hash")


def _runtime_boundary_payload(obj: "DecoderCandidateRuntimeBoundary") -> dict[str, Any]:
    return _base_payload(obj, "decoder_candidate_runtime_boundary_hash")


def _equivalence_precondition_payload(
    obj: "DecoderCandidateEquivalencePrecondition",
) -> dict[str, Any]:
    return _base_payload(obj, "decoder_candidate_equivalence_precondition_hash")


def _promotion_boundary_payload(
    obj: "DecoderCandidatePromotionBoundary",
) -> dict[str, Any]:
    return _base_payload(obj, "decoder_candidate_promotion_boundary_hash")


def _declaration_payload(obj: "DecoderCandidateDeclaration") -> dict[str, Any]:
    return _base_payload(obj, "decoder_candidate_declaration_hash")


def _manifest_payload(obj: "DecoderCandidateManifest") -> dict[str, Any]:
    return _base_payload(obj, "decoder_candidate_manifest_hash")


@dataclass(frozen=True)
class DecoderCandidateIdentity:
    candidate_name: str
    candidate_version: str
    candidate_release: str
    previous_release_tag: str
    previous_release_url: str
    upstream_canonical_decoder_baseline_receipt_hash: str
    candidate_kind: str
    candidate_status: str
    adapter_only: bool
    promoted: bool
    canonical_baseline_replacement: bool
    runtime_authority_allowed: bool
    decoder_candidate_identity_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderCandidateIdentity:
            raise _invalid_input()
        _require_text(self.candidate_name, "candidate_name")
        _require_text(self.candidate_version, "candidate_version")
        if self.candidate_release != MANIFEST_VERSION:
            raise _invalid_candidate("candidate_release")
        if self.previous_release_tag != PREVIOUS_RELEASE_TAG:
            raise _invalid_candidate("previous_release_tag")
        if self.previous_release_url != PREVIOUS_RELEASE_URL:
            raise _invalid_input("previous_release_url")
        _validate_hash_format(
            self.upstream_canonical_decoder_baseline_receipt_hash,
            "upstream_canonical_decoder_baseline_receipt_hash",
        )
        _require_text(self.candidate_kind, "candidate_kind")
        if self.candidate_kind not in _CANDIDATE_KINDS:
            raise _invalid_input("candidate_kind:UNKNOWN")
        if self.candidate_status != _CANDIDATE_STATUS:
            raise _invalid_candidate("candidate_status")
        _require_policy_flags(self, _IDENTITY_REPLAY_POLICY, "identity:UNSAFE")
        _assert_hash_matches(self, "decoder_candidate_identity_hash", _identity_payload)


@dataclass(frozen=True)
class DecoderCandidateSourceFile:
    path: str
    sha256: str
    source_role: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderCandidateSourceFile:
            raise _invalid_input()
        _validate_candidate_source_path(self.path)
        _validate_hash_format(self.sha256, "source_file.sha256")
        if self.source_role != _SOURCE_ROLE:
            raise _invalid_input("source_role")


@dataclass(frozen=True)
class DecoderCandidateSourceBoundary:
    candidate_source_root: str
    source_boundary_mode: str
    source_files: tuple[DecoderCandidateSourceFile, ...]
    source_file_count: int
    source_tree_hash: str
    candidate_import_allowed: bool
    candidate_runtime_execution_allowed: bool
    baseline_decoder_mutation_allowed: bool
    filesystem_mutation_allowed: bool
    decoder_candidate_source_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderCandidateSourceBoundary:
            raise _invalid_input()
        _validate_candidate_source_root(self.candidate_source_root)
        if self.source_boundary_mode != _SOURCE_BOUNDARY_MODE:
            raise _invalid_candidate("source_boundary_mode")
        ordered = _ordered_source_files(self.source_files)
        if self.source_files != ordered:
            raise _invalid_input("source_files:ORDER")
        if any(
            not source_file.path.startswith(self.candidate_source_root)
            for source_file in ordered
        ):
            raise _invalid_input("source_files:OUTSIDE_CANDIDATE_SOURCE_ROOT")
        if type(self.source_file_count) is not int or self.source_file_count != len(
            ordered
        ):
            raise _invalid_input("source_file_count")
        _validate_hash_format(self.source_tree_hash, "source_tree_hash")
        if _compute_source_tree_hash(ordered) != self.source_tree_hash:
            raise _hash_mismatch("source_tree_hash")
        _require_policy_flags(
            self, _SOURCE_BOUNDARY_REPLAY_POLICY, "source_boundary:UNSAFE"
        )
        _assert_hash_matches(
            self, "decoder_candidate_source_boundary_hash", _source_boundary_payload
        )


@dataclass(frozen=True)
class DecoderCandidateCapabilityDeclaration:
    capability_mode: str
    declared_capabilities: tuple[str, ...]
    capability_count: int
    performance_claim_allowed: bool
    correctness_claim_allowed: bool
    benchmark_claim_allowed: bool
    qec_advantage_claim_allowed: bool
    hardware_authority_allowed: bool
    exact_equivalence_claimed: bool
    decoder_candidate_capability_declaration_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderCandidateCapabilityDeclaration:
            raise _invalid_input()
        if self.capability_mode != _CAPABILITY_MODE:
            raise _invalid_candidate("capability_mode")
        ordered = _ordered_capabilities(self.declared_capabilities)
        if self.declared_capabilities != ordered:
            raise _invalid_input("declared_capabilities:ORDER")
        if type(self.capability_count) is not int or self.capability_count != len(
            ordered
        ):
            raise _invalid_input("capability_count")
        _require_policy_flags(
            self, _CAPABILITY_REPLAY_POLICY, "capability_declaration:UNSAFE"
        )
        _assert_hash_matches(
            self,
            "decoder_candidate_capability_declaration_hash",
            _capability_declaration_payload,
        )


@dataclass(frozen=True)
class DecoderCandidateRuntimeBoundary:
    runtime_boundary_mode: str
    baseline_decoder_import_allowed: bool
    candidate_import_allowed: bool
    decoder_workload_execution_allowed: bool
    benchmark_execution_allowed: bool
    network_allowed: bool
    heavy_backend_import_allowed: bool
    hardware_sdk_allowed: bool
    decoder_candidate_runtime_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderCandidateRuntimeBoundary:
            raise _invalid_input()
        if self.runtime_boundary_mode != _RUNTIME_BOUNDARY_MODE:
            raise _invalid_candidate("runtime_boundary_mode")
        _require_policy_flags(self, _RUNTIME_REPLAY_POLICY, "runtime_boundary:UNSAFE")
        _assert_hash_matches(
            self, "decoder_candidate_runtime_boundary_hash", _runtime_boundary_payload
        )


@dataclass(frozen=True)
class DecoderCandidateEquivalencePrecondition:
    upstream_canonical_decoder_baseline_receipt_hash: str
    required_future_receipt_kind: str
    required_future_release: str
    equivalence_required_before_promotion: bool
    equivalence_mode: str
    replay_corpus_required: bool
    output_schema_match_required: bool
    precision_policy: str
    approximation_policy: str
    equivalence_proven: bool
    candidate_output_authority_allowed: bool
    candidate_status_until_equivalence: str
    decoder_candidate_equivalence_precondition_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderCandidateEquivalencePrecondition:
            raise _invalid_input()
        _validate_hash_format(
            self.upstream_canonical_decoder_baseline_receipt_hash,
            "upstream_canonical_decoder_baseline_receipt_hash",
        )
        for field_name in (
            "required_future_receipt_kind",
            "required_future_release",
            "equivalence_mode",
            "precision_policy",
            "approximation_policy",
            "candidate_status_until_equivalence",
        ):
            _check_forbidden_declaration_semantics(
                getattr(self, field_name), field_name
            )
        if self.required_future_receipt_kind != _REQUIRED_FUTURE_RECEIPT_KIND:
            raise _invalid_candidate("required_future_receipt_kind")
        if self.required_future_release != _REQUIRED_FUTURE_RELEASE:
            raise _invalid_candidate("required_future_release")
        if self.equivalence_mode != _EQUIVALENCE_MODE:
            raise _invalid_candidate("equivalence_mode")
        if self.precision_policy != _PRECISION_POLICY:
            raise _invalid_candidate("precision_policy")
        if self.approximation_policy != _APPROXIMATION_POLICY:
            raise _invalid_candidate("approximation_policy")
        if self.candidate_status_until_equivalence != _CANDIDATE_STATUS:
            raise _invalid_candidate("candidate_status_until_equivalence")
        _require_policy_flags(
            self, _EQUIVALENCE_REPLAY_POLICY, "equivalence_precondition:UNSAFE"
        )
        _assert_hash_matches(
            self,
            "decoder_candidate_equivalence_precondition_hash",
            _equivalence_precondition_payload,
        )


@dataclass(frozen=True)
class DecoderCandidatePromotionBoundary:
    promotion_status: str
    promotion_allowed_in_this_release: bool
    runtime_authority_allowed: bool
    silent_replacement_allowed: bool
    probabilistic_promotion_allowed: bool
    ml_decoder_authority_allowed: bool
    benchmark_marketing_allowed: bool
    rollback_receipt_required_before_promotion: bool
    benchmark_ladder_required_before_performance_claims: bool
    baseline_mutation_allowed: bool
    decoder_candidate_promotion_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderCandidatePromotionBoundary:
            raise _invalid_input()
        if self.promotion_status != _PROMOTION_STATUS:
            raise _invalid_candidate("promotion_status")
        _require_policy_flags(
            self, _PROMOTION_REPLAY_POLICY, "promotion_boundary:UNSAFE"
        )
        _assert_hash_matches(
            self,
            "decoder_candidate_promotion_boundary_hash",
            _promotion_boundary_payload,
        )


def _declaration_excludes_decoder_roots(
    declaration: "DecoderCandidateDeclaration",
) -> bool:
    return not any(
        source_file.path == _DECODER_FORBIDDEN_ROOT
        or source_file.path.startswith(f"{_DECODER_FORBIDDEN_ROOT}/")
        for source_file in declaration.source_boundary.source_files
    )


def _compute_replay_safe_candidate_declaration(
    identity: DecoderCandidateIdentity,
    source_boundary: DecoderCandidateSourceBoundary,
    capability_declaration: DecoderCandidateCapabilityDeclaration,
    runtime_boundary: DecoderCandidateRuntimeBoundary,
    equivalence_precondition: DecoderCandidateEquivalencePrecondition,
    promotion_boundary: DecoderCandidatePromotionBoundary,
) -> bool:
    return all(
        (
            _policy_flags_satisfied(identity, _IDENTITY_REPLAY_POLICY),
            source_boundary.source_boundary_mode == _SOURCE_BOUNDARY_MODE,
            _policy_flags_satisfied(source_boundary, _SOURCE_BOUNDARY_REPLAY_POLICY),
            capability_declaration.capability_mode == _CAPABILITY_MODE,
            _policy_flags_satisfied(capability_declaration, _CAPABILITY_REPLAY_POLICY),
            _policy_flags_satisfied(runtime_boundary, _RUNTIME_REPLAY_POLICY),
            equivalence_precondition.required_future_receipt_kind
            == _REQUIRED_FUTURE_RECEIPT_KIND,
            equivalence_precondition.required_future_release
            == _REQUIRED_FUTURE_RELEASE,
            equivalence_precondition.equivalence_mode == _EQUIVALENCE_MODE,
            _policy_flags_satisfied(
                equivalence_precondition, _EQUIVALENCE_REPLAY_POLICY
            ),
            _policy_flags_satisfied(promotion_boundary, _PROMOTION_REPLAY_POLICY),
        )
    )


@dataclass(frozen=True)
class DecoderCandidateDeclaration:
    identity: DecoderCandidateIdentity
    source_boundary: DecoderCandidateSourceBoundary
    capability_declaration: DecoderCandidateCapabilityDeclaration
    runtime_boundary: DecoderCandidateRuntimeBoundary
    equivalence_precondition: DecoderCandidateEquivalencePrecondition
    promotion_boundary: DecoderCandidatePromotionBoundary
    replay_safe_candidate_declaration: bool
    decoder_candidate_declaration_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderCandidateDeclaration:
            raise _invalid_input()
        validate_decoder_candidate_identity(self.identity)
        validate_decoder_candidate_source_boundary(self.source_boundary)
        validate_decoder_candidate_capability_declaration(self.capability_declaration)
        validate_decoder_candidate_runtime_boundary(self.runtime_boundary)
        validate_decoder_candidate_equivalence_precondition(
            self.equivalence_precondition
        )
        validate_decoder_candidate_promotion_boundary(self.promotion_boundary)
        if (
            self.identity.upstream_canonical_decoder_baseline_receipt_hash
            != self.equivalence_precondition.upstream_canonical_decoder_baseline_receipt_hash
        ):
            raise _invalid_candidate(
                "upstream_canonical_decoder_baseline_receipt_hash:MISMATCH"
            )
        _require_exact_bool(
            self.replay_safe_candidate_declaration, "replay_safe_candidate_declaration"
        )
        if (
            self.replay_safe_candidate_declaration
            != _compute_replay_safe_candidate_declaration(
                self.identity,
                self.source_boundary,
                self.capability_declaration,
                self.runtime_boundary,
                self.equivalence_precondition,
                self.promotion_boundary,
            )
        ):
            raise _invalid_candidate("replay_safe_candidate_declaration")
        _assert_hash_matches(
            self, "decoder_candidate_declaration_hash", _declaration_payload
        )


def _declaration_replay_policy_satisfied(
    declaration: DecoderCandidateDeclaration,
) -> bool:
    return all(
        (
            declaration.replay_safe_candidate_declaration is True,
            _declaration_excludes_decoder_roots(declaration),
            _compute_replay_safe_candidate_declaration(
                declaration.identity,
                declaration.source_boundary,
                declaration.capability_declaration,
                declaration.runtime_boundary,
                declaration.equivalence_precondition,
                declaration.promotion_boundary,
            ),
        )
    )


def _compute_replay_safe_manifest(
    declarations: Sequence[DecoderCandidateDeclaration],
) -> bool:
    return all(
        _declaration_replay_policy_satisfied(declaration)
        for declaration in declarations
    )


@dataclass(frozen=True)
class DecoderCandidateManifest:
    manifest_version: str
    manifest_kind: str
    previous_release_tag: str
    previous_release_url: str
    upstream_canonical_decoder_baseline_receipt_hash: str
    candidate_declarations: tuple[DecoderCandidateDeclaration, ...]
    candidate_count: int
    all_candidates_adapter_only: bool
    replay_safe_decoder_candidate_manifest: bool
    decoder_candidate_manifest_hash: str

    def __post_init__(self) -> None:
        if type(self) is not DecoderCandidateManifest:
            raise _invalid_input()
        if (
            self.manifest_version != MANIFEST_VERSION
            or self.manifest_kind != MANIFEST_KIND
        ):
            raise _invalid_candidate("manifest_identity")
        if self.previous_release_tag != PREVIOUS_RELEASE_TAG:
            raise _invalid_candidate("previous_release_tag")
        if self.previous_release_url != PREVIOUS_RELEASE_URL:
            raise _invalid_input("previous_release_url")
        _validate_hash_format(
            self.upstream_canonical_decoder_baseline_receipt_hash,
            "upstream_canonical_decoder_baseline_receipt_hash",
        )
        ordered = _ordered_declarations(self.candidate_declarations)
        if self.candidate_declarations != ordered:
            raise _invalid_input("candidate_declarations:ORDER")
        if any(
            declaration.identity.upstream_canonical_decoder_baseline_receipt_hash
            != self.upstream_canonical_decoder_baseline_receipt_hash
            for declaration in ordered
        ):
            raise _invalid_candidate("candidate_declarations:UPSTREAM_HASH_MISMATCH")
        if type(self.candidate_count) is not int or self.candidate_count != len(
            ordered
        ):
            raise _invalid_input("candidate_count")
        for field_name in (
            "all_candidates_adapter_only",
            "replay_safe_decoder_candidate_manifest",
        ):
            _require_exact_bool(getattr(self, field_name), field_name)
        all_adapter = all(
            declaration.identity.adapter_only is True for declaration in ordered
        )
        if self.all_candidates_adapter_only != all_adapter:
            raise _invalid_candidate("all_candidates_adapter_only")
        if self.replay_safe_decoder_candidate_manifest != (
            all_adapter and _compute_replay_safe_manifest(ordered)
        ):
            raise _invalid_candidate("replay_safe_decoder_candidate_manifest")
        _assert_hash_matches(self, "decoder_candidate_manifest_hash", _manifest_payload)


def build_decoder_candidate_identity(
    candidate_name: str,
    candidate_version: str,
    upstream_canonical_decoder_baseline_receipt_hash: str,
    candidate_kind: str = "CANDIDATE_DECODER_BACKEND",
    candidate_release: str = MANIFEST_VERSION,
    previous_release_tag: str = PREVIOUS_RELEASE_TAG,
    previous_release_url: str = PREVIOUS_RELEASE_URL,
    candidate_status: str = _CANDIDATE_STATUS,
    adapter_only: bool = True,
    promoted: bool = False,
    canonical_baseline_replacement: bool = False,
    runtime_authority_allowed: bool = False,
) -> DecoderCandidateIdentity:
    payload = {
        "candidate_name": candidate_name,
        "candidate_version": candidate_version,
        "candidate_release": candidate_release,
        "previous_release_tag": previous_release_tag,
        "previous_release_url": previous_release_url,
        "upstream_canonical_decoder_baseline_receipt_hash": upstream_canonical_decoder_baseline_receipt_hash,
        "candidate_kind": candidate_kind,
        "candidate_status": candidate_status,
        "adapter_only": adapter_only,
        "promoted": promoted,
        "canonical_baseline_replacement": canonical_baseline_replacement,
        "runtime_authority_allowed": runtime_authority_allowed,
    }
    return DecoderCandidateIdentity(
        **payload, decoder_candidate_identity_hash=_hash_payload(payload)
    )


def build_decoder_candidate_source_file(
    path: str, sha256: str, source_role: str = _SOURCE_ROLE
) -> DecoderCandidateSourceFile:
    return DecoderCandidateSourceFile(path=path, sha256=sha256, source_role=source_role)


def build_decoder_candidate_source_boundary(
    source_files: Sequence[DecoderCandidateSourceFile],
    candidate_source_root: str = "candidate_decoders/",
    source_boundary_mode: str = _SOURCE_BOUNDARY_MODE,
    candidate_import_allowed: bool = False,
    candidate_runtime_execution_allowed: bool = False,
    baseline_decoder_mutation_allowed: bool = False,
    filesystem_mutation_allowed: bool = False,
) -> DecoderCandidateSourceBoundary:
    ordered = _ordered_source_files(source_files)
    source_tree_hash = _compute_source_tree_hash(ordered)
    payload = {
        "candidate_source_root": candidate_source_root,
        "source_boundary_mode": source_boundary_mode,
        "source_files": ordered,
        "source_file_count": len(ordered),
        "source_tree_hash": source_tree_hash,
        "candidate_import_allowed": candidate_import_allowed,
        "candidate_runtime_execution_allowed": candidate_runtime_execution_allowed,
        "baseline_decoder_mutation_allowed": baseline_decoder_mutation_allowed,
        "filesystem_mutation_allowed": filesystem_mutation_allowed,
    }
    return DecoderCandidateSourceBoundary(
        **payload, decoder_candidate_source_boundary_hash=_hash_payload(payload)
    )


def build_decoder_candidate_capability_declaration(
    declared_capabilities: Sequence[str],
    capability_mode: str = _CAPABILITY_MODE,
    performance_claim_allowed: bool = False,
    correctness_claim_allowed: bool = False,
    benchmark_claim_allowed: bool = False,
    qec_advantage_claim_allowed: bool = False,
    hardware_authority_allowed: bool = False,
    exact_equivalence_claimed: bool = False,
) -> DecoderCandidateCapabilityDeclaration:
    ordered = _ordered_capabilities(declared_capabilities)
    payload = {
        "capability_mode": capability_mode,
        "declared_capabilities": ordered,
        "capability_count": len(ordered),
        "performance_claim_allowed": performance_claim_allowed,
        "correctness_claim_allowed": correctness_claim_allowed,
        "benchmark_claim_allowed": benchmark_claim_allowed,
        "qec_advantage_claim_allowed": qec_advantage_claim_allowed,
        "hardware_authority_allowed": hardware_authority_allowed,
        "exact_equivalence_claimed": exact_equivalence_claimed,
    }
    return DecoderCandidateCapabilityDeclaration(
        **payload, decoder_candidate_capability_declaration_hash=_hash_payload(payload)
    )


def build_decoder_candidate_runtime_boundary(
    runtime_boundary_mode: str = _RUNTIME_BOUNDARY_MODE,
    baseline_decoder_import_allowed: bool = False,
    candidate_import_allowed: bool = False,
    decoder_workload_execution_allowed: bool = False,
    benchmark_execution_allowed: bool = False,
    network_allowed: bool = False,
    heavy_backend_import_allowed: bool = False,
    hardware_sdk_allowed: bool = False,
) -> DecoderCandidateRuntimeBoundary:
    payload = {
        "runtime_boundary_mode": runtime_boundary_mode,
        "baseline_decoder_import_allowed": baseline_decoder_import_allowed,
        "candidate_import_allowed": candidate_import_allowed,
        "decoder_workload_execution_allowed": decoder_workload_execution_allowed,
        "benchmark_execution_allowed": benchmark_execution_allowed,
        "network_allowed": network_allowed,
        "heavy_backend_import_allowed": heavy_backend_import_allowed,
        "hardware_sdk_allowed": hardware_sdk_allowed,
    }
    return DecoderCandidateRuntimeBoundary(
        **payload, decoder_candidate_runtime_boundary_hash=_hash_payload(payload)
    )


def build_decoder_candidate_equivalence_precondition(
    upstream_canonical_decoder_baseline_receipt_hash: str,
    required_future_receipt_kind: str = _REQUIRED_FUTURE_RECEIPT_KIND,
    required_future_release: str = _REQUIRED_FUTURE_RELEASE,
    equivalence_required_before_promotion: bool = True,
    equivalence_mode: str = _EQUIVALENCE_MODE,
    replay_corpus_required: bool = True,
    output_schema_match_required: bool = True,
    precision_policy: str = _PRECISION_POLICY,
    approximation_policy: str = _APPROXIMATION_POLICY,
    equivalence_proven: bool = False,
    candidate_output_authority_allowed: bool = False,
    candidate_status_until_equivalence: str = _CANDIDATE_STATUS,
) -> DecoderCandidateEquivalencePrecondition:
    payload = {
        "upstream_canonical_decoder_baseline_receipt_hash": upstream_canonical_decoder_baseline_receipt_hash,
        "required_future_receipt_kind": required_future_receipt_kind,
        "required_future_release": required_future_release,
        "equivalence_required_before_promotion": equivalence_required_before_promotion,
        "equivalence_mode": equivalence_mode,
        "replay_corpus_required": replay_corpus_required,
        "output_schema_match_required": output_schema_match_required,
        "precision_policy": precision_policy,
        "approximation_policy": approximation_policy,
        "equivalence_proven": equivalence_proven,
        "candidate_output_authority_allowed": candidate_output_authority_allowed,
        "candidate_status_until_equivalence": candidate_status_until_equivalence,
    }
    return DecoderCandidateEquivalencePrecondition(
        **payload,
        decoder_candidate_equivalence_precondition_hash=_hash_payload(payload),
    )


def build_decoder_candidate_promotion_boundary(
    promotion_status: str = _PROMOTION_STATUS,
    promotion_allowed_in_this_release: bool = False,
    runtime_authority_allowed: bool = False,
    silent_replacement_allowed: bool = False,
    probabilistic_promotion_allowed: bool = False,
    ml_decoder_authority_allowed: bool = False,
    benchmark_marketing_allowed: bool = False,
    rollback_receipt_required_before_promotion: bool = True,
    benchmark_ladder_required_before_performance_claims: bool = True,
    baseline_mutation_allowed: bool = False,
) -> DecoderCandidatePromotionBoundary:
    payload = {
        "promotion_status": promotion_status,
        "promotion_allowed_in_this_release": promotion_allowed_in_this_release,
        "runtime_authority_allowed": runtime_authority_allowed,
        "silent_replacement_allowed": silent_replacement_allowed,
        "probabilistic_promotion_allowed": probabilistic_promotion_allowed,
        "ml_decoder_authority_allowed": ml_decoder_authority_allowed,
        "benchmark_marketing_allowed": benchmark_marketing_allowed,
        "rollback_receipt_required_before_promotion": rollback_receipt_required_before_promotion,
        "benchmark_ladder_required_before_performance_claims": benchmark_ladder_required_before_performance_claims,
        "baseline_mutation_allowed": baseline_mutation_allowed,
    }
    return DecoderCandidatePromotionBoundary(
        **payload, decoder_candidate_promotion_boundary_hash=_hash_payload(payload)
    )


def build_decoder_candidate_declaration(
    identity: DecoderCandidateIdentity,
    source_boundary: DecoderCandidateSourceBoundary,
    capability_declaration: DecoderCandidateCapabilityDeclaration,
    runtime_boundary: DecoderCandidateRuntimeBoundary,
    equivalence_precondition: DecoderCandidateEquivalencePrecondition,
    promotion_boundary: DecoderCandidatePromotionBoundary,
) -> DecoderCandidateDeclaration:
    replay_safe = _compute_replay_safe_candidate_declaration(
        identity,
        source_boundary,
        capability_declaration,
        runtime_boundary,
        equivalence_precondition,
        promotion_boundary,
    )
    payload = {
        "identity": identity,
        "source_boundary": source_boundary,
        "capability_declaration": capability_declaration,
        "runtime_boundary": runtime_boundary,
        "equivalence_precondition": equivalence_precondition,
        "promotion_boundary": promotion_boundary,
        "replay_safe_candidate_declaration": replay_safe,
    }
    return DecoderCandidateDeclaration(
        **payload, decoder_candidate_declaration_hash=_hash_payload(payload)
    )


def build_decoder_candidate_manifest(
    upstream_canonical_decoder_baseline_receipt_hash: str,
    candidate_declarations: Sequence[DecoderCandidateDeclaration],
    manifest_version: str = MANIFEST_VERSION,
    manifest_kind: str = MANIFEST_KIND,
    previous_release_tag: str = PREVIOUS_RELEASE_TAG,
    previous_release_url: str = PREVIOUS_RELEASE_URL,
) -> DecoderCandidateManifest:
    ordered = _ordered_declarations(candidate_declarations)
    all_adapter = all(
        declaration.identity.adapter_only is True for declaration in ordered
    )
    replay_safe = all_adapter and _compute_replay_safe_manifest(ordered)
    payload = {
        "manifest_version": manifest_version,
        "manifest_kind": manifest_kind,
        "previous_release_tag": previous_release_tag,
        "previous_release_url": previous_release_url,
        "upstream_canonical_decoder_baseline_receipt_hash": upstream_canonical_decoder_baseline_receipt_hash,
        "candidate_declarations": ordered,
        "candidate_count": len(ordered),
        "all_candidates_adapter_only": all_adapter,
        "replay_safe_decoder_candidate_manifest": replay_safe,
    }
    return DecoderCandidateManifest(
        **payload, decoder_candidate_manifest_hash=_hash_payload(payload)
    )


def validate_decoder_candidate_identity(
    value: DecoderCandidateIdentity,
) -> DecoderCandidateIdentity:
    _revalidate_exact_instance(value, DecoderCandidateIdentity)
    return value


def validate_decoder_candidate_source_file(
    value: DecoderCandidateSourceFile,
) -> DecoderCandidateSourceFile:
    _revalidate_exact_instance(value, DecoderCandidateSourceFile)
    return value


def validate_decoder_candidate_source_boundary(
    value: DecoderCandidateSourceBoundary,
) -> DecoderCandidateSourceBoundary:
    _revalidate_exact_instance(value, DecoderCandidateSourceBoundary)
    return value


def validate_decoder_candidate_capability_declaration(
    value: DecoderCandidateCapabilityDeclaration,
) -> DecoderCandidateCapabilityDeclaration:
    _revalidate_exact_instance(value, DecoderCandidateCapabilityDeclaration)
    return value


def validate_decoder_candidate_runtime_boundary(
    value: DecoderCandidateRuntimeBoundary,
) -> DecoderCandidateRuntimeBoundary:
    _revalidate_exact_instance(value, DecoderCandidateRuntimeBoundary)
    return value


def validate_decoder_candidate_equivalence_precondition(
    value: DecoderCandidateEquivalencePrecondition,
) -> DecoderCandidateEquivalencePrecondition:
    _revalidate_exact_instance(value, DecoderCandidateEquivalencePrecondition)
    return value


def validate_decoder_candidate_promotion_boundary(
    value: DecoderCandidatePromotionBoundary,
) -> DecoderCandidatePromotionBoundary:
    _revalidate_exact_instance(value, DecoderCandidatePromotionBoundary)
    return value


def validate_decoder_candidate_declaration(
    value: DecoderCandidateDeclaration,
) -> DecoderCandidateDeclaration:
    _revalidate_exact_instance(value, DecoderCandidateDeclaration)
    return value


def validate_decoder_candidate_manifest(
    value: DecoderCandidateManifest,
) -> DecoderCandidateManifest:
    _revalidate_exact_instance(value, DecoderCandidateManifest)
    return value


__all__ = [
    "DecoderCandidateManifestErrorCode",
    "DecoderCandidateManifestError",
    "DecoderCandidateIdentity",
    "DecoderCandidateSourceFile",
    "DecoderCandidateSourceBoundary",
    "DecoderCandidateCapabilityDeclaration",
    "DecoderCandidateRuntimeBoundary",
    "DecoderCandidateEquivalencePrecondition",
    "DecoderCandidatePromotionBoundary",
    "DecoderCandidateDeclaration",
    "DecoderCandidateManifest",
    "build_decoder_candidate_identity",
    "build_decoder_candidate_source_file",
    "build_decoder_candidate_source_boundary",
    "build_decoder_candidate_capability_declaration",
    "build_decoder_candidate_runtime_boundary",
    "build_decoder_candidate_equivalence_precondition",
    "build_decoder_candidate_promotion_boundary",
    "build_decoder_candidate_declaration",
    "build_decoder_candidate_manifest",
    "validate_decoder_candidate_identity",
    "validate_decoder_candidate_source_file",
    "validate_decoder_candidate_source_boundary",
    "validate_decoder_candidate_capability_declaration",
    "validate_decoder_candidate_runtime_boundary",
    "validate_decoder_candidate_equivalence_precondition",
    "validate_decoder_candidate_promotion_boundary",
    "validate_decoder_candidate_declaration",
    "validate_decoder_candidate_manifest",
]
