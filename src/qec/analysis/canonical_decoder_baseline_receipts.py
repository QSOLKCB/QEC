from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, fields, is_dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

BASELINE_RELEASE = "v166.0"
PREVIOUS_RELEASE_TAG = "v165.9.4"
PREVIOUS_RELEASE_URL = "https://github.com/QSOLKCB/QEC/releases/tag/v165.9.4"
DECODER_ROOT = "src/qec/decoder/"
RECEIPT_KIND = "CanonicalDecoderBaselineReceipt"

_CANONICAL_DECODER_ROLE = "CANONICAL_DECODER_ORACLE"
_SOURCE_ROLE = "BASELINE_DECODER_SOURCE"
_SOURCE_BOUNDARY_MODE = "SOURCE_HASH_BOUND"
_CORPUS_MODE = "DECLARED_STATIC_REPLAY_CORPUS"
_SYNDROME_ORDERING_POLICY = "CANONICAL_LEXICOGRAPHIC_ORDER"
_EQUIVALENCE_MODE = "EXACT_CANONICAL_OUTPUT_MATCH"
_OUTPUT_ORDERING_POLICY = "CANONICAL_OUTPUT_LEXICOGRAPHIC_ORDER"
_PRECISION_POLICY = "DECLARED_EXACT_NO_HIDDEN_PRECISION_DRIFT"
_APPROXIMATION_POLICY = "NO_UNDECLARED_APPROXIMATION"
_CANDIDATE_STATUS = "ADAPTER_ONLY_BEFORE_PROMOTION"
_MUTATION_POLICY = "IMMUTABLE_CANONICAL_BASELINE"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_TEXT_LENGTH = 512

_FORBIDDEN_DECLARATION_TOKENS = (
    "silent decoder replacement",
    "decoder replaced because faster",
    "speed proves correctness",
    "benchmark proves correctness",
    "runtime promotion",
    "candidate decoder promoted",
    "probabilistic decoder authority",
    "ml decoder authority",
    "hardware authority",
    "qec advantage proven",
    "mutation of canonical decoder",
    "deleting rollback path",
    "hidden precision drift",
    "undeclared approximation policy",
)
_SEMANTIC_GUARD_EXACT_ALLOWLIST = {
    _PRECISION_POLICY,
    _APPROXIMATION_POLICY,
}


def _invalid_input() -> ValueError:
    return ValueError("INVALID_INPUT")


def _invalid_baseline() -> ValueError:
    return ValueError("INVALID_DECODER_BASELINE")


def _to_canonical_obj(value: Any) -> Any:
    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        return {k: _to_canonical_obj(v) for k, v in value.__dict__.items()}
    if isinstance(value, Mapping):
        return {str(k): _to_canonical_obj(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_canonical_obj(v) for v in value]
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _to_canonical_obj(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _base_payload(payload: Mapping[str, Any], hash_key: str) -> dict[str, Any]:
    out = dict(payload)
    out.pop(hash_key, None)
    return out


def _validate_hash_format(value: str) -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError("INVALID_HASH")


def _require_text(value: str) -> None:
    if not isinstance(value, str) or not value or len(value) > _MAX_TEXT_LENGTH:
        raise _invalid_input()
    _check_forbidden_declaration_semantics(value)


def _require_exact_bool(value: Any) -> None:
    if type(value) is not bool:
        raise _invalid_input()


def _normalize_semantics_text(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"\\[nrt]", " ", lowered)
    lowered = lowered.replace("_", " ").replace("-", " ")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def _check_forbidden_declaration_semantics(value: Any) -> None:
    if not isinstance(value, str) or value in _SEMANTIC_GUARD_EXACT_ALLOWLIST:
        return
    normalized = _normalize_semantics_text(value)
    for token in _FORBIDDEN_DECLARATION_TOKENS:
        if _normalize_semantics_text(token) in normalized:
            raise _invalid_input()


def _validate_decoder_path(path: str) -> None:
    if not isinstance(path, str) or not path:
        raise _invalid_input()
    _check_forbidden_declaration_semantics(path)
    if path.startswith("/") or path.startswith("\\"):
        raise _invalid_input()
    if ".." in PurePosixPath(path).parts:
        raise _invalid_input()
    if not path.startswith(DECODER_ROOT):
        raise _invalid_input()


def _ordered_source_files(source_files: Sequence["CanonicalDecoderSourceFile"]) -> tuple["CanonicalDecoderSourceFile", ...]:
    if not isinstance(source_files, (tuple, list)):
        raise _invalid_input()
    ordered = tuple(sorted(source_files, key=lambda item: item.path if type(item) is CanonicalDecoderSourceFile else ""))
    seen: set[str] = set()
    for source_file in ordered:
        _revalidate_exact_instance(source_file, CanonicalDecoderSourceFile)
        if source_file.path in seen:
            raise _invalid_input()
        seen.add(source_file.path)
    return ordered


def _ordered_protected_paths(protected_paths: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(protected_paths, (tuple, list)):
        raise _invalid_input()
    ordered = tuple(sorted(protected_paths))
    if not ordered:
        raise _invalid_input()
    if len(set(ordered)) != len(ordered):
        raise _invalid_input()
    for path in ordered:
        _validate_decoder_path(path)
    return ordered


def _source_file_payload(source_file: "CanonicalDecoderSourceFile") -> dict[str, Any]:
    return {
        "path": source_file.path,
        "sha256": source_file.sha256,
        "source_role": source_file.source_role,
    }


def _source_tree_payload(source_files: Sequence["CanonicalDecoderSourceFile"]) -> dict[str, Any]:
    ordered = _ordered_source_files(source_files)
    return {"source_files": [_source_file_payload(source_file) for source_file in ordered]}


def _compute_source_tree_hash(source_files: Sequence["CanonicalDecoderSourceFile"]) -> str:
    return _hash_payload(_source_tree_payload(source_files))


def _identity_payload(receipt: "CanonicalDecoderIdentity") -> dict[str, Any]:
    return _base_payload(receipt.__dict__, "canonical_decoder_identity_hash")


def _source_boundary_payload(receipt: "CanonicalDecoderSourceBoundary") -> dict[str, Any]:
    return _base_payload(receipt.__dict__, "canonical_decoder_source_boundary_hash")


def _replay_corpus_boundary_payload(receipt: "CanonicalDecoderReplayCorpusBoundary") -> dict[str, Any]:
    return _base_payload(receipt.__dict__, "canonical_decoder_replay_corpus_boundary_hash")


def _equivalence_policy_payload(receipt: "CanonicalDecoderEquivalencePolicy") -> dict[str, Any]:
    return _base_payload(receipt.__dict__, "canonical_decoder_equivalence_policy_hash")


def _immutability_boundary_payload(receipt: "CanonicalDecoderImmutabilityBoundary") -> dict[str, Any]:
    return _base_payload(receipt.__dict__, "canonical_decoder_immutability_boundary_hash")


def _baseline_receipt_payload(receipt: "CanonicalDecoderBaselineReceipt") -> dict[str, Any]:
    return _base_payload(receipt.__dict__, "canonical_decoder_baseline_receipt_hash")


def _revalidate_exact_instance(value: Any, cls: type[Any]) -> None:
    if type(value) is not cls or not is_dataclass(value) or set(value.__dict__.keys()) != {f.name for f in fields(cls)}:
        raise _invalid_input()
    post_init = getattr(value, "__post_init__", None)
    if callable(post_init):
        post_init()


@dataclass(frozen=True)
class CanonicalDecoderIdentity:
    decoder_name: str
    decoder_root: str
    baseline_release: str
    previous_release_tag: str
    previous_release_url: str
    decoder_role: str
    canonical_baseline: bool
    adapter_only: bool
    canonical_decoder_identity_hash: str

    def __post_init__(self) -> None:
        if type(self) is not CanonicalDecoderIdentity:
            raise _invalid_input()
        _require_text(self.decoder_name)
        if self.decoder_root != DECODER_ROOT:
            raise _invalid_baseline()
        if self.baseline_release != BASELINE_RELEASE:
            raise _invalid_baseline()
        if self.previous_release_tag != PREVIOUS_RELEASE_TAG:
            raise _invalid_baseline()
        if self.previous_release_url != PREVIOUS_RELEASE_URL:
            raise _invalid_input()
        if self.decoder_role != _CANONICAL_DECODER_ROLE:
            raise _invalid_input()
        _require_exact_bool(self.canonical_baseline)
        _require_exact_bool(self.adapter_only)
        if self.canonical_baseline is not True or self.adapter_only is not False:
            raise _invalid_baseline()
        _validate_hash_format(self.canonical_decoder_identity_hash)
        if _hash_payload(_identity_payload(self)) != self.canonical_decoder_identity_hash:
            raise ValueError("HASH_MISMATCH")


@dataclass(frozen=True)
class CanonicalDecoderSourceFile:
    path: str
    sha256: str
    source_role: str

    def __post_init__(self) -> None:
        if type(self) is not CanonicalDecoderSourceFile:
            raise _invalid_input()
        _validate_decoder_path(self.path)
        _validate_hash_format(self.sha256)
        if self.source_role != _SOURCE_ROLE:
            raise _invalid_input()


@dataclass(frozen=True)
class CanonicalDecoderSourceBoundary:
    decoder_root: str
    source_boundary_mode: str
    source_files: tuple[CanonicalDecoderSourceFile, ...]
    source_file_count: int
    source_tree_hash: str
    runtime_decoder_execution_allowed: bool
    decoder_import_allowed: bool
    mutation_allowed: bool
    canonical_decoder_source_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not CanonicalDecoderSourceBoundary:
            raise _invalid_input()
        if self.decoder_root != DECODER_ROOT or self.source_boundary_mode != _SOURCE_BOUNDARY_MODE:
            raise _invalid_baseline()
        ordered = _ordered_source_files(self.source_files)
        if not ordered:
            raise _invalid_input()
        if self.source_files != ordered:
            raise _invalid_input()
        if type(self.source_file_count) is not int or self.source_file_count != len(ordered):
            raise _invalid_input()
        _validate_hash_format(self.source_tree_hash)
        if _compute_source_tree_hash(ordered) != self.source_tree_hash:
            raise ValueError("HASH_MISMATCH")
        for flag in (self.runtime_decoder_execution_allowed, self.decoder_import_allowed, self.mutation_allowed):
            _require_exact_bool(flag)
            if flag is not False:
                raise _invalid_baseline()
        _validate_hash_format(self.canonical_decoder_source_boundary_hash)
        if _hash_payload(_source_boundary_payload(self)) != self.canonical_decoder_source_boundary_hash:
            raise ValueError("HASH_MISMATCH")


@dataclass(frozen=True)
class CanonicalDecoderReplayCorpusBoundary:
    corpus_name: str
    corpus_version: str
    corpus_hash: str
    corpus_mode: str
    syndrome_ordering_policy: str
    input_schema_hash: str
    output_schema_hash: str
    runtime_decoder_execution_allowed: bool
    candidate_replay_required_before_promotion: bool
    canonical_decoder_replay_corpus_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not CanonicalDecoderReplayCorpusBoundary:
            raise _invalid_input()
        _require_text(self.corpus_name)
        _require_text(self.corpus_version)
        for digest in (self.corpus_hash, self.input_schema_hash, self.output_schema_hash):
            _validate_hash_format(digest)
        if self.corpus_mode != _CORPUS_MODE:
            raise _invalid_input()
        if self.syndrome_ordering_policy != _SYNDROME_ORDERING_POLICY:
            raise _invalid_input()
        _require_exact_bool(self.runtime_decoder_execution_allowed)
        _require_exact_bool(self.candidate_replay_required_before_promotion)
        if self.runtime_decoder_execution_allowed is not False or self.candidate_replay_required_before_promotion is not True:
            raise _invalid_baseline()
        _validate_hash_format(self.canonical_decoder_replay_corpus_boundary_hash)
        if _hash_payload(_replay_corpus_boundary_payload(self)) != self.canonical_decoder_replay_corpus_boundary_hash:
            raise ValueError("HASH_MISMATCH")


@dataclass(frozen=True)
class CanonicalDecoderEquivalencePolicy:
    equivalence_mode: str
    output_ordering_policy: str
    precision_policy: str
    approximation_policy: str
    candidate_status_before_promotion: str
    benchmark_claims_allowed: bool
    hardware_authority_allowed: bool
    probabilistic_promotion_allowed: bool
    silent_replacement_allowed: bool
    canonical_decoder_equivalence_policy_hash: str

    def __post_init__(self) -> None:
        if type(self) is not CanonicalDecoderEquivalencePolicy:
            raise _invalid_input()
        if self.equivalence_mode != _EQUIVALENCE_MODE:
            raise _invalid_input()
        if self.output_ordering_policy != _OUTPUT_ORDERING_POLICY:
            raise _invalid_input()
        if self.precision_policy != _PRECISION_POLICY:
            raise _invalid_input()
        if self.approximation_policy != _APPROXIMATION_POLICY:
            raise _invalid_input()
        if self.candidate_status_before_promotion != _CANDIDATE_STATUS:
            raise _invalid_input()
        for value in (
            self.benchmark_claims_allowed,
            self.hardware_authority_allowed,
            self.probabilistic_promotion_allowed,
            self.silent_replacement_allowed,
        ):
            _require_exact_bool(value)
            if value is not False:
                raise _invalid_baseline()
        _validate_hash_format(self.canonical_decoder_equivalence_policy_hash)
        if _hash_payload(_equivalence_policy_payload(self)) != self.canonical_decoder_equivalence_policy_hash:
            raise ValueError("HASH_MISMATCH")


@dataclass(frozen=True)
class CanonicalDecoderImmutabilityBoundary:
    decoder_root: str
    protected_paths: tuple[str, ...]
    mutation_policy: str
    mutation_allowed: bool
    silent_replacement_allowed: bool
    candidate_implementation_allowed: bool
    runtime_promotion_allowed: bool
    rollback_required_for_future_promotion: bool
    canonical_decoder_immutability_boundary_hash: str

    def __post_init__(self) -> None:
        if type(self) is not CanonicalDecoderImmutabilityBoundary:
            raise _invalid_input()
        if self.decoder_root != DECODER_ROOT or self.mutation_policy != _MUTATION_POLICY:
            raise _invalid_baseline()
        ordered = _ordered_protected_paths(self.protected_paths)
        if self.protected_paths != ordered:
            raise _invalid_input()
        for value in (
            self.mutation_allowed,
            self.silent_replacement_allowed,
            self.candidate_implementation_allowed,
            self.runtime_promotion_allowed,
            self.rollback_required_for_future_promotion,
        ):
            _require_exact_bool(value)
        if (
            self.mutation_allowed is not False
            or self.silent_replacement_allowed is not False
            or self.candidate_implementation_allowed is not False
            or self.runtime_promotion_allowed is not False
            or self.rollback_required_for_future_promotion is not True
        ):
            raise _invalid_baseline()
        _validate_hash_format(self.canonical_decoder_immutability_boundary_hash)
        if _hash_payload(_immutability_boundary_payload(self)) != self.canonical_decoder_immutability_boundary_hash:
            raise ValueError("HASH_MISMATCH")


@dataclass(frozen=True)
class CanonicalDecoderBaselineReceipt:
    receipt_version: str
    receipt_kind: str
    upstream_graph_universe_claim_boundary_receipt_hash: str
    identity: CanonicalDecoderIdentity
    source_boundary: CanonicalDecoderSourceBoundary
    replay_corpus_boundary: CanonicalDecoderReplayCorpusBoundary
    equivalence_policy: CanonicalDecoderEquivalencePolicy
    immutability_boundary: CanonicalDecoderImmutabilityBoundary
    replay_safe_canonical_decoder_baseline: bool
    canonical_decoder_baseline_receipt_hash: str

    def __post_init__(self) -> None:
        if type(self) is not CanonicalDecoderBaselineReceipt:
            raise _invalid_input()
        if self.receipt_version != BASELINE_RELEASE or self.receipt_kind != RECEIPT_KIND:
            raise _invalid_baseline()
        _validate_hash_format(self.upstream_graph_universe_claim_boundary_receipt_hash)
        validate_canonical_decoder_identity(self.identity)
        validate_canonical_decoder_source_boundary(self.source_boundary)
        validate_canonical_decoder_replay_corpus_boundary(self.replay_corpus_boundary)
        validate_canonical_decoder_equivalence_policy(self.equivalence_policy)
        validate_canonical_decoder_immutability_boundary(self.immutability_boundary)
        _require_exact_bool(self.replay_safe_canonical_decoder_baseline)
        if self.replay_safe_canonical_decoder_baseline != _compute_replay_safe_canonical_decoder_baseline(
            self.identity,
            self.source_boundary,
            self.replay_corpus_boundary,
            self.equivalence_policy,
            self.immutability_boundary,
        ):
            raise _invalid_baseline()
        _validate_hash_format(self.canonical_decoder_baseline_receipt_hash)
        if _hash_payload(_baseline_receipt_payload(self)) != self.canonical_decoder_baseline_receipt_hash:
            raise ValueError("HASH_MISMATCH")


def build_canonical_decoder_identity(
    decoder_name: str,
    decoder_root: str = DECODER_ROOT,
    baseline_release: str = BASELINE_RELEASE,
    previous_release_tag: str = PREVIOUS_RELEASE_TAG,
    previous_release_url: str = PREVIOUS_RELEASE_URL,
    decoder_role: str = _CANONICAL_DECODER_ROLE,
    canonical_baseline: bool = True,
    adapter_only: bool = False,
) -> CanonicalDecoderIdentity:
    payload = {
        "decoder_name": decoder_name,
        "decoder_root": decoder_root,
        "baseline_release": baseline_release,
        "previous_release_tag": previous_release_tag,
        "previous_release_url": previous_release_url,
        "decoder_role": decoder_role,
        "canonical_baseline": canonical_baseline,
        "adapter_only": adapter_only,
    }
    return CanonicalDecoderIdentity(**payload, canonical_decoder_identity_hash=_hash_payload(payload))


def build_canonical_decoder_source_file(
    path: str,
    sha256: str,
    source_role: str = _SOURCE_ROLE,
) -> CanonicalDecoderSourceFile:
    return CanonicalDecoderSourceFile(path=path, sha256=sha256, source_role=source_role)


def build_canonical_decoder_source_boundary(
    source_files: Sequence[CanonicalDecoderSourceFile],
    decoder_root: str = DECODER_ROOT,
    source_boundary_mode: str = _SOURCE_BOUNDARY_MODE,
    runtime_decoder_execution_allowed: bool = False,
    decoder_import_allowed: bool = False,
    mutation_allowed: bool = False,
) -> CanonicalDecoderSourceBoundary:
    ordered = _ordered_source_files(source_files)
    source_tree_hash = _compute_source_tree_hash(ordered)
    payload = {
        "decoder_root": decoder_root,
        "source_boundary_mode": source_boundary_mode,
        "source_files": ordered,
        "source_file_count": len(ordered),
        "source_tree_hash": source_tree_hash,
        "runtime_decoder_execution_allowed": runtime_decoder_execution_allowed,
        "decoder_import_allowed": decoder_import_allowed,
        "mutation_allowed": mutation_allowed,
    }
    return CanonicalDecoderSourceBoundary(**payload, canonical_decoder_source_boundary_hash=_hash_payload(payload))


def build_canonical_decoder_replay_corpus_boundary(
    corpus_name: str,
    corpus_version: str,
    corpus_hash: str,
    input_schema_hash: str,
    output_schema_hash: str,
    corpus_mode: str = _CORPUS_MODE,
    syndrome_ordering_policy: str = _SYNDROME_ORDERING_POLICY,
    runtime_decoder_execution_allowed: bool = False,
    candidate_replay_required_before_promotion: bool = True,
) -> CanonicalDecoderReplayCorpusBoundary:
    payload = {
        "corpus_name": corpus_name,
        "corpus_version": corpus_version,
        "corpus_hash": corpus_hash,
        "corpus_mode": corpus_mode,
        "syndrome_ordering_policy": syndrome_ordering_policy,
        "input_schema_hash": input_schema_hash,
        "output_schema_hash": output_schema_hash,
        "runtime_decoder_execution_allowed": runtime_decoder_execution_allowed,
        "candidate_replay_required_before_promotion": candidate_replay_required_before_promotion,
    }
    return CanonicalDecoderReplayCorpusBoundary(
        **payload,
        canonical_decoder_replay_corpus_boundary_hash=_hash_payload(payload),
    )


def build_canonical_decoder_equivalence_policy(
    equivalence_mode: str = _EQUIVALENCE_MODE,
    output_ordering_policy: str = _OUTPUT_ORDERING_POLICY,
    precision_policy: str = _PRECISION_POLICY,
    approximation_policy: str = _APPROXIMATION_POLICY,
    candidate_status_before_promotion: str = _CANDIDATE_STATUS,
    benchmark_claims_allowed: bool = False,
    hardware_authority_allowed: bool = False,
    probabilistic_promotion_allowed: bool = False,
    silent_replacement_allowed: bool = False,
) -> CanonicalDecoderEquivalencePolicy:
    for value in (
        equivalence_mode,
        output_ordering_policy,
        precision_policy,
        approximation_policy,
        candidate_status_before_promotion,
    ):
        _check_forbidden_declaration_semantics(value)
    payload = {
        "equivalence_mode": equivalence_mode,
        "output_ordering_policy": output_ordering_policy,
        "precision_policy": precision_policy,
        "approximation_policy": approximation_policy,
        "candidate_status_before_promotion": candidate_status_before_promotion,
        "benchmark_claims_allowed": benchmark_claims_allowed,
        "hardware_authority_allowed": hardware_authority_allowed,
        "probabilistic_promotion_allowed": probabilistic_promotion_allowed,
        "silent_replacement_allowed": silent_replacement_allowed,
    }
    return CanonicalDecoderEquivalencePolicy(
        **payload,
        canonical_decoder_equivalence_policy_hash=_hash_payload(payload),
    )


def build_canonical_decoder_immutability_boundary(
    protected_paths: Sequence[str],
    decoder_root: str = DECODER_ROOT,
    mutation_policy: str = _MUTATION_POLICY,
    mutation_allowed: bool = False,
    silent_replacement_allowed: bool = False,
    candidate_implementation_allowed: bool = False,
    runtime_promotion_allowed: bool = False,
    rollback_required_for_future_promotion: bool = True,
) -> CanonicalDecoderImmutabilityBoundary:
    ordered = _ordered_protected_paths(protected_paths)
    payload = {
        "decoder_root": decoder_root,
        "protected_paths": ordered,
        "mutation_policy": mutation_policy,
        "mutation_allowed": mutation_allowed,
        "silent_replacement_allowed": silent_replacement_allowed,
        "candidate_implementation_allowed": candidate_implementation_allowed,
        "runtime_promotion_allowed": runtime_promotion_allowed,
        "rollback_required_for_future_promotion": rollback_required_for_future_promotion,
    }
    return CanonicalDecoderImmutabilityBoundary(
        **payload,
        canonical_decoder_immutability_boundary_hash=_hash_payload(payload),
    )


def _compute_replay_safe_canonical_decoder_baseline(
    identity: CanonicalDecoderIdentity,
    source_boundary: CanonicalDecoderSourceBoundary,
    replay_corpus_boundary: CanonicalDecoderReplayCorpusBoundary,
    equivalence_policy: CanonicalDecoderEquivalencePolicy,
    immutability_boundary: CanonicalDecoderImmutabilityBoundary,
) -> bool:
    return all(
        (
            identity.canonical_baseline is True,
            identity.adapter_only is False,
            source_boundary.source_boundary_mode == _SOURCE_BOUNDARY_MODE,
            source_boundary.runtime_decoder_execution_allowed is False,
            source_boundary.decoder_import_allowed is False,
            source_boundary.mutation_allowed is False,
            replay_corpus_boundary.corpus_mode == _CORPUS_MODE,
            replay_corpus_boundary.runtime_decoder_execution_allowed is False,
            replay_corpus_boundary.candidate_replay_required_before_promotion is True,
            equivalence_policy.equivalence_mode == _EQUIVALENCE_MODE,
            equivalence_policy.benchmark_claims_allowed is False,
            equivalence_policy.hardware_authority_allowed is False,
            equivalence_policy.probabilistic_promotion_allowed is False,
            equivalence_policy.silent_replacement_allowed is False,
            immutability_boundary.mutation_allowed is False,
            immutability_boundary.silent_replacement_allowed is False,
            immutability_boundary.candidate_implementation_allowed is False,
            immutability_boundary.runtime_promotion_allowed is False,
        )
    )


def build_canonical_decoder_baseline_receipt(
    upstream_graph_universe_claim_boundary_receipt_hash: str,
    identity: CanonicalDecoderIdentity,
    source_boundary: CanonicalDecoderSourceBoundary,
    replay_corpus_boundary: CanonicalDecoderReplayCorpusBoundary,
    equivalence_policy: CanonicalDecoderEquivalencePolicy,
    immutability_boundary: CanonicalDecoderImmutabilityBoundary,
) -> CanonicalDecoderBaselineReceipt:
    validate_canonical_decoder_identity(identity)
    validate_canonical_decoder_source_boundary(source_boundary)
    validate_canonical_decoder_replay_corpus_boundary(replay_corpus_boundary)
    validate_canonical_decoder_equivalence_policy(equivalence_policy)
    validate_canonical_decoder_immutability_boundary(immutability_boundary)
    replay_safe = _compute_replay_safe_canonical_decoder_baseline(
        identity,
        source_boundary,
        replay_corpus_boundary,
        equivalence_policy,
        immutability_boundary,
    )
    payload = {
        "receipt_version": BASELINE_RELEASE,
        "receipt_kind": RECEIPT_KIND,
        "upstream_graph_universe_claim_boundary_receipt_hash": upstream_graph_universe_claim_boundary_receipt_hash,
        "identity": identity,
        "source_boundary": source_boundary,
        "replay_corpus_boundary": replay_corpus_boundary,
        "equivalence_policy": equivalence_policy,
        "immutability_boundary": immutability_boundary,
        "replay_safe_canonical_decoder_baseline": replay_safe,
    }
    return CanonicalDecoderBaselineReceipt(
        **payload,
        canonical_decoder_baseline_receipt_hash=_hash_payload(payload),
    )


def validate_canonical_decoder_identity(receipt: CanonicalDecoderIdentity) -> None:
    _revalidate_exact_instance(receipt, CanonicalDecoderIdentity)


def validate_canonical_decoder_source_file(receipt: CanonicalDecoderSourceFile) -> None:
    _revalidate_exact_instance(receipt, CanonicalDecoderSourceFile)


def validate_canonical_decoder_source_boundary(receipt: CanonicalDecoderSourceBoundary) -> None:
    _revalidate_exact_instance(receipt, CanonicalDecoderSourceBoundary)


def validate_canonical_decoder_replay_corpus_boundary(receipt: CanonicalDecoderReplayCorpusBoundary) -> None:
    _revalidate_exact_instance(receipt, CanonicalDecoderReplayCorpusBoundary)


def validate_canonical_decoder_equivalence_policy(receipt: CanonicalDecoderEquivalencePolicy) -> None:
    _revalidate_exact_instance(receipt, CanonicalDecoderEquivalencePolicy)


def validate_canonical_decoder_immutability_boundary(receipt: CanonicalDecoderImmutabilityBoundary) -> None:
    _revalidate_exact_instance(receipt, CanonicalDecoderImmutabilityBoundary)


def validate_canonical_decoder_baseline_receipt(receipt: CanonicalDecoderBaselineReceipt) -> None:
    _revalidate_exact_instance(receipt, CanonicalDecoderBaselineReceipt)
