from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from qec.analysis.dataframe_backend_manifest import DataframeSchemaField, DataframeSchemaManifest, validate_schema_field, validate_schema_manifest
from qec.analysis.polars_pandas_equivalence_receipts import DataframeSchemaComparison, validate_dataframe_schema_comparison

_SCHEMA_VERSION = "SCHEMA_EQUIVALENCE_RECEIPT_V1"
_MAX_FIELDS = 4096
_MAX_MISMATCHES = 4096
_MAX_NAME_LENGTH = 128
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_EQUIVALENCE_MODES = {
    "EXACT_SCHEMA",
    "ORDER_INSENSITIVE_SCHEMA",
    "DECLARED_DTYPE_EQUIVALENCE",
    "DECLARED_NULLABILITY_EQUIVALENCE",
    "DECLARED_EVOLUTION_EQUIVALENCE",
}
_ALLOWED_COMPATIBILITY_POLICIES = {
    "STRICT_COMPATIBILITY",
    "FORWARD_COMPATIBLE",
    "BACKWARD_COMPATIBLE",
    "BIDIRECTIONAL_COMPATIBLE",
    "DECLARED_EXTERNAL_COMPATIBILITY",
}
_ALLOWED_SCHEMA_TRANSITIONS = {
    "NO_CHANGE",
    "COLUMN_ADDED",
    "COLUMN_REMOVED",
    "COLUMN_RENAMED",
    "COLUMN_REORDERED",
    "DTYPE_CHANGED",
    "NULLABILITY_CHANGED",
    "COLUMN_SPLIT",
    "COLUMN_MERGED",
}
_ALLOWED_DTYPE_EQUIVALENCE_MODES = {
    "EXACT_DTYPE",
    "NUMERIC_WIDTH_EQUIVALENT",
    "SIGNED_UNSIGNED_DECLARED",
    "STRING_CATEGORY_DECLARED",
    "DECLARED_EXTERNAL_DTYPE_EQUIVALENCE",
}
_ALLOWED_NULLABILITY_MODES = {
    "EXACT_NULLABILITY",
    "NULLABLE_TO_NONNULL_FORBIDDEN",
    "DECLARED_NULLABILITY_EQUIVALENCE",
}
_ALLOWED_ORDERING_MODES = {
    "EXACT_ORDER",
    "ORDER_INSENSITIVE",
    "DECLARED_SORT_EQUIVALENCE",
}
_ALLOWED_MISMATCH_KINDS = {
    "FIELD_COUNT_MISMATCH",
    "FIELD_NAME_MISMATCH",
    "FIELD_POSITION_MISMATCH",
    "DTYPE_MISMATCH",
    "NULLABILITY_MISMATCH",
    "ORDERING_MISMATCH",
    "TRANSITION_MISMATCH",
    "COMPATIBILITY_POLICY_MISMATCH",
}
_FORBIDDEN_RUNTIME_TOKENS = (
    "schema auto fixed",
    "automatic coercion",
    "runtime migration",
    "backend executed",
    "runtime dataframe",
    "silent conversion",
    "automatic optimization",
    "backend authority",
)


@dataclass(frozen=True)
class SchemaFieldEquivalence:
    left_field_hash: str
    right_field_hash: str
    equivalence_mode: str
    fields_equivalent: bool
    field_equivalence_hash: str


@dataclass(frozen=True)
class SchemaOrderingEquivalence:
    ordering_mode: str
    ordering_equivalent: bool
    ordering_equivalence_hash: str


@dataclass(frozen=True)
class SchemaNullabilityEquivalence:
    nullability_mode: str
    nullability_equivalent: bool
    nullability_equivalence_hash: str


@dataclass(frozen=True)
class SchemaDTypeEquivalence:
    dtype_equivalence_mode: str
    dtype_equivalent: bool
    dtype_equivalence_hash: str


@dataclass(frozen=True)
class SchemaEvolutionTransition:
    transition_type: str
    left_schema_hash: str
    right_schema_hash: str
    transition_allowed: bool
    schema_evolution_transition_hash: str


@dataclass(frozen=True)
class SchemaMismatchRecord:
    mismatch_index: int
    mismatch_kind: str
    left_hash: str
    right_hash: str
    reason: str
    schema_mismatch_record_hash: str


@dataclass(frozen=True)
class SchemaCompatibilityPolicy:
    compatibility_policy: str
    policy_reason: str
    compatibility_policy_hash: str


@dataclass(frozen=True)
class SchemaEquivalenceReceipt:
    schema_version: str
    left_schema_manifest_hash: str
    right_schema_manifest_hash: str
    field_equivalences: tuple[SchemaFieldEquivalence, ...]
    ordering_equivalence: SchemaOrderingEquivalence
    nullability_equivalence: SchemaNullabilityEquivalence
    dtype_equivalence: SchemaDTypeEquivalence
    evolution_transition: SchemaEvolutionTransition
    compatibility_policy: SchemaCompatibilityPolicy
    mismatches: tuple[SchemaMismatchRecord, ...] = field(default_factory=tuple)
    mismatch_count: int = 0
    schemas_equivalent: bool = False
    adapter_only: bool = True
    schema_equivalence_receipt_hash: str = ""


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_to_canonical_obj(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


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


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _base_payload(payload: Mapping[str, Any], hash_key: str) -> dict[str, Any]:
    out = dict(payload)
    out.pop(hash_key, None)
    return out


def _validate_hash_format(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character hex digest")


def _validate_dense_mismatch_indices(mismatches: Sequence[SchemaMismatchRecord]) -> None:
    if sorted(m.mismatch_index for m in mismatches) != list(range(len(mismatches))):
        raise ValueError("mismatch indices must be dense and zero-indexed")


def _validate_unique_mismatch_indices(mismatches: Sequence[SchemaMismatchRecord]) -> None:
    idx = [m.mismatch_index for m in mismatches]
    if len(idx) != len(set(idx)):
        raise ValueError("duplicate mismatch indices are not allowed")


def _validate_equivalence_semantics(receipt: SchemaEquivalenceReceipt) -> tuple[int, bool]:
    mismatch_count = len(receipt.mismatches)
    all_fields = all(f.fields_equivalent for f in receipt.field_equivalences)
    equivalent = all_fields and receipt.ordering_equivalence.ordering_equivalent and receipt.nullability_equivalence.nullability_equivalent and receipt.dtype_equivalence.dtype_equivalent and receipt.evolution_transition.transition_allowed and mismatch_count == 0
    return mismatch_count, equivalent


def _validate_policy_consistency(receipt: SchemaEquivalenceReceipt) -> None:
    if receipt.evolution_transition.transition_type == "NO_CHANGE":
        if receipt.left_schema_manifest_hash != receipt.right_schema_manifest_hash:
            raise ValueError("NO_CHANGE transition requires equal schema manifest hashes")
    if receipt.evolution_transition.transition_type == "COLUMN_REORDERED" and receipt.ordering_equivalence.ordering_mode == "EXACT_ORDER":
        raise ValueError("COLUMN_REORDERED transition is inconsistent with EXACT_ORDER")
    if any(fe.equivalence_mode == "DECLARED_EVOLUTION_EQUIVALENCE" for fe in receipt.field_equivalences):
        if not receipt.evolution_transition.transition_allowed:
            raise ValueError("DECLARED_EVOLUTION_EQUIVALENCE requires transition_allowed")
        if receipt.compatibility_policy.compatibility_policy == "STRICT_COMPATIBILITY":
            raise ValueError("DECLARED_EVOLUTION_EQUIVALENCE forbids STRICT_COMPATIBILITY")


def _check_no_forbidden_runtime_semantics(payload: Mapping[str, Any]) -> None:
    blob = _canonical_json(payload).lower()
    for token in _FORBIDDEN_RUNTIME_TOKENS:
        if token in blob:
            raise ValueError("runtime semantics, backend authority claims, and implicit coercion semantics are forbidden")


def build_schema_field_equivalence(left_field_hash: str, right_field_hash: str, equivalence_mode: str, fields_equivalent: bool) -> SchemaFieldEquivalence:
    payload = {"left_field_hash": left_field_hash, "right_field_hash": right_field_hash, "equivalence_mode": equivalence_mode, "fields_equivalent": fields_equivalent}
    obj = SchemaFieldEquivalence(**payload, field_equivalence_hash=_hash_payload(payload))
    validate_schema_field_equivalence(obj)
    return obj


def validate_schema_field_equivalence(item: SchemaFieldEquivalence) -> None:
    if item.equivalence_mode not in _ALLOWED_EQUIVALENCE_MODES:
        raise ValueError("invalid equivalence mode")
    if not isinstance(item.fields_equivalent, bool):
        raise ValueError("fields_equivalent must be a bool")
    _validate_hash_format(item.left_field_hash, "left_field_hash")
    _validate_hash_format(item.right_field_hash, "right_field_hash")
    _validate_hash_format(item.field_equivalence_hash, "field_equivalence_hash")
    expected = _hash_payload(_base_payload(item.__dict__, "field_equivalence_hash"))
    if expected != item.field_equivalence_hash:
        raise ValueError("field equivalence hash mismatch")


def build_schema_ordering_equivalence(ordering_mode: str, ordering_equivalent: bool) -> SchemaOrderingEquivalence:
    payload = {"ordering_mode": ordering_mode, "ordering_equivalent": ordering_equivalent}
    obj = SchemaOrderingEquivalence(**payload, ordering_equivalence_hash=_hash_payload(payload))
    validate_schema_ordering_equivalence(obj)
    return obj


def validate_schema_ordering_equivalence(item: SchemaOrderingEquivalence) -> None:
    if item.ordering_mode not in _ALLOWED_ORDERING_MODES:
        raise ValueError("invalid ordering equivalence")
    if not isinstance(item.ordering_equivalent, bool):
        raise ValueError("ordering_equivalent must be a bool")
    _validate_hash_format(item.ordering_equivalence_hash, "ordering_equivalence_hash")
    expected = _hash_payload(_base_payload(item.__dict__, "ordering_equivalence_hash"))
    if expected != item.ordering_equivalence_hash:
        raise ValueError("ordering equivalence hash mismatch")


def build_schema_nullability_equivalence(nullability_mode: str, nullability_equivalent: bool) -> SchemaNullabilityEquivalence:
    payload = {"nullability_mode": nullability_mode, "nullability_equivalent": nullability_equivalent}
    obj = SchemaNullabilityEquivalence(**payload, nullability_equivalence_hash=_hash_payload(payload))
    validate_schema_nullability_equivalence(obj)
    return obj


def validate_schema_nullability_equivalence(item: SchemaNullabilityEquivalence) -> None:
    if item.nullability_mode not in _ALLOWED_NULLABILITY_MODES:
        raise ValueError("invalid nullability equivalence")
    if not isinstance(item.nullability_equivalent, bool):
        raise ValueError("nullability_equivalent must be a bool")
    _validate_hash_format(item.nullability_equivalence_hash, "nullability_equivalence_hash")
    expected = _hash_payload(_base_payload(item.__dict__, "nullability_equivalence_hash"))
    if expected != item.nullability_equivalence_hash:
        raise ValueError("nullability equivalence hash mismatch")


def build_schema_dtype_equivalence(dtype_equivalence_mode: str, dtype_equivalent: bool) -> SchemaDTypeEquivalence:
    payload = {"dtype_equivalence_mode": dtype_equivalence_mode, "dtype_equivalent": dtype_equivalent}
    obj = SchemaDTypeEquivalence(**payload, dtype_equivalence_hash=_hash_payload(payload))
    validate_schema_dtype_equivalence(obj)
    return obj


def validate_schema_dtype_equivalence(item: SchemaDTypeEquivalence) -> None:
    if item.dtype_equivalence_mode not in _ALLOWED_DTYPE_EQUIVALENCE_MODES:
        raise ValueError("invalid dtype equivalence")
    if not isinstance(item.dtype_equivalent, bool):
        raise ValueError("dtype_equivalent must be a bool")
    _validate_hash_format(item.dtype_equivalence_hash, "dtype_equivalence_hash")
    expected = _hash_payload(_base_payload(item.__dict__, "dtype_equivalence_hash"))
    if expected != item.dtype_equivalence_hash:
        raise ValueError("dtype equivalence hash mismatch")


def build_schema_evolution_transition(transition_type: str, left_schema_hash: str, right_schema_hash: str, transition_allowed: bool) -> SchemaEvolutionTransition:
    payload = {"transition_type": transition_type, "left_schema_hash": left_schema_hash, "right_schema_hash": right_schema_hash, "transition_allowed": transition_allowed}
    obj = SchemaEvolutionTransition(**payload, schema_evolution_transition_hash=_hash_payload(payload))
    validate_schema_evolution_transition(obj)
    return obj


def validate_schema_evolution_transition(item: SchemaEvolutionTransition) -> None:
    if item.transition_type not in _ALLOWED_SCHEMA_TRANSITIONS:
        raise ValueError("invalid schema transition")
    if not isinstance(item.transition_allowed, bool):
        raise ValueError("transition_allowed must be a bool")
    _validate_hash_format(item.left_schema_hash, "left_schema_hash")
    _validate_hash_format(item.right_schema_hash, "right_schema_hash")
    _validate_hash_format(item.schema_evolution_transition_hash, "schema_evolution_transition_hash")
    expected = _hash_payload(_base_payload(item.__dict__, "schema_evolution_transition_hash"))
    if expected != item.schema_evolution_transition_hash:
        raise ValueError("schema evolution transition hash mismatch")


def build_schema_mismatch_record(mismatch_index: int, mismatch_kind: str, left_hash: str, right_hash: str, reason: str) -> SchemaMismatchRecord:
    payload = {"mismatch_index": mismatch_index, "mismatch_kind": mismatch_kind, "left_hash": left_hash, "right_hash": right_hash, "reason": reason}
    obj = SchemaMismatchRecord(**payload, schema_mismatch_record_hash=_hash_payload(payload))
    validate_schema_mismatch_record(obj)
    return obj


def validate_schema_mismatch_record(item: SchemaMismatchRecord) -> None:
    if item.mismatch_kind not in _ALLOWED_MISMATCH_KINDS:
        raise ValueError("invalid mismatch kind")
    if len(item.reason) > _MAX_NAME_LENGTH:
        raise ValueError("reason exceeds max length")
    _validate_hash_format(item.left_hash, "left_hash")
    _validate_hash_format(item.right_hash, "right_hash")
    _validate_hash_format(item.schema_mismatch_record_hash, "schema_mismatch_record_hash")
    _check_no_forbidden_runtime_semantics({"reason": item.reason})
    expected = _hash_payload(_base_payload(item.__dict__, "schema_mismatch_record_hash"))
    if expected != item.schema_mismatch_record_hash:
        raise ValueError("schema mismatch record hash mismatch")


def build_schema_compatibility_policy(compatibility_policy: str, policy_reason: str) -> SchemaCompatibilityPolicy:
    payload = {"compatibility_policy": compatibility_policy, "policy_reason": policy_reason}
    obj = SchemaCompatibilityPolicy(**payload, compatibility_policy_hash=_hash_payload(payload))
    validate_schema_compatibility_policy(obj)
    return obj


def validate_schema_compatibility_policy(item: SchemaCompatibilityPolicy) -> None:
    if item.compatibility_policy not in _ALLOWED_COMPATIBILITY_POLICIES:
        raise ValueError("invalid compatibility policy")
    if len(item.policy_reason) > _MAX_NAME_LENGTH:
        raise ValueError("policy_reason exceeds max length")
    _validate_hash_format(item.compatibility_policy_hash, "compatibility_policy_hash")
    _check_no_forbidden_runtime_semantics({"policy_reason": item.policy_reason})
    expected = _hash_payload(_base_payload(item.__dict__, "compatibility_policy_hash"))
    if expected != item.compatibility_policy_hash:
        raise ValueError("compatibility policy hash mismatch")


def build_schema_equivalence_receipt(
    left_schema_manifest: DataframeSchemaManifest,
    right_schema_manifest: DataframeSchemaManifest,
    schema_comparison: DataframeSchemaComparison,
    field_equivalences: Sequence[SchemaFieldEquivalence],
    ordering_equivalence: SchemaOrderingEquivalence,
    nullability_equivalence: SchemaNullabilityEquivalence,
    dtype_equivalence: SchemaDTypeEquivalence,
    evolution_transition: SchemaEvolutionTransition,
    compatibility_policy: SchemaCompatibilityPolicy,
    mismatches: Sequence[SchemaMismatchRecord],
    adapter_only: bool = True,
) -> SchemaEquivalenceReceipt:
    validate_schema_manifest(left_schema_manifest)
    validate_schema_manifest(right_schema_manifest)
    for f in left_schema_manifest.fields:
        validate_schema_field(DataframeSchemaField(**f.__dict__))
    for f in right_schema_manifest.fields:
        validate_schema_field(DataframeSchemaField(**f.__dict__))
    validate_dataframe_schema_comparison(schema_comparison)
    # P1: Bind schema_comparison to manifest hashes
    if schema_comparison.left_schema_manifest_hash != left_schema_manifest.schema_manifest_hash:
        raise ValueError("schema_comparison.left_schema_manifest_hash must match left_schema_manifest.schema_manifest_hash")
    if schema_comparison.right_schema_manifest_hash != right_schema_manifest.schema_manifest_hash:
        raise ValueError("schema_comparison.right_schema_manifest_hash must match right_schema_manifest.schema_manifest_hash")
    # P1: Bind evolution_transition hashes to manifest hashes
    if evolution_transition.left_schema_hash != left_schema_manifest.schema_manifest_hash:
        raise ValueError("evolution_transition.left_schema_hash must match left_schema_manifest.schema_manifest_hash")
    if evolution_transition.right_schema_hash != right_schema_manifest.schema_manifest_hash:
        raise ValueError("evolution_transition.right_schema_hash must match right_schema_manifest.schema_manifest_hash")

    field_eqs = tuple(field_equivalences)
    # P2: Canonicalize mismatch ordering by mismatch_index for deterministic hashing
    mm = tuple(sorted(mismatches, key=lambda m: m.mismatch_index))
    # Compute mismatch_count and schemas_equivalent directly from inputs
    mismatch_count = len(mm)
    all_fields = all(f.fields_equivalent for f in field_eqs)
    schemas_equivalent = (
        all_fields
        and ordering_equivalence.ordering_equivalent
        and nullability_equivalence.nullability_equivalent
        and dtype_equivalence.dtype_equivalent
        and evolution_transition.transition_allowed
        and mismatch_count == 0
    )
    obj = SchemaEquivalenceReceipt(
        schema_version=_SCHEMA_VERSION,
        left_schema_manifest_hash=left_schema_manifest.schema_manifest_hash,
        right_schema_manifest_hash=right_schema_manifest.schema_manifest_hash,
        field_equivalences=field_eqs,
        ordering_equivalence=ordering_equivalence,
        nullability_equivalence=nullability_equivalence,
        dtype_equivalence=dtype_equivalence,
        evolution_transition=evolution_transition,
        compatibility_policy=compatibility_policy,
        mismatches=mm,
        mismatch_count=mismatch_count,
        schemas_equivalent=schemas_equivalent,
        adapter_only=adapter_only,
        schema_equivalence_receipt_hash=_hash_payload(_base_payload({
            "schema_version": _SCHEMA_VERSION,
            "left_schema_manifest_hash": left_schema_manifest.schema_manifest_hash,
            "right_schema_manifest_hash": right_schema_manifest.schema_manifest_hash,
            "field_equivalences": field_eqs,
            "ordering_equivalence": ordering_equivalence,
            "nullability_equivalence": nullability_equivalence,
            "dtype_equivalence": dtype_equivalence,
            "evolution_transition": evolution_transition,
            "compatibility_policy": compatibility_policy,
            "mismatches": mm,
            "mismatch_count": mismatch_count,
            "schemas_equivalent": schemas_equivalent,
            "adapter_only": adapter_only,
            "schema_equivalence_receipt_hash": "",
        }, "schema_equivalence_receipt_hash")),
    )
    validate_schema_equivalence_receipt(obj)
    return obj


def validate_schema_equivalence_receipt(receipt: SchemaEquivalenceReceipt) -> None:
    if receipt.schema_version != _SCHEMA_VERSION:
        raise ValueError("invalid schema version")
    _validate_hash_format(receipt.left_schema_manifest_hash, "left_schema_manifest_hash")
    _validate_hash_format(receipt.right_schema_manifest_hash, "right_schema_manifest_hash")
    if type(receipt.adapter_only) is not bool or receipt.adapter_only is not True:
        raise ValueError("adapter_only must be exactly True")
    if len(receipt.field_equivalences) > _MAX_FIELDS or len(receipt.mismatches) > _MAX_MISMATCHES:
        raise ValueError("too many fields or mismatches")
    for e in receipt.field_equivalences:
        validate_schema_field_equivalence(e)
    validate_schema_ordering_equivalence(receipt.ordering_equivalence)
    validate_schema_nullability_equivalence(receipt.nullability_equivalence)
    validate_schema_dtype_equivalence(receipt.dtype_equivalence)
    validate_schema_evolution_transition(receipt.evolution_transition)
    validate_schema_compatibility_policy(receipt.compatibility_policy)
    for m in receipt.mismatches:
        validate_schema_mismatch_record(m)
    _validate_unique_mismatch_indices(receipt.mismatches)
    _validate_dense_mismatch_indices(receipt.mismatches)
    _validate_policy_consistency(receipt)
    mismatch_count, equivalent = _validate_equivalence_semantics(receipt)
    if receipt.mismatch_count != mismatch_count:
        raise ValueError("mismatch_count mismatch")
    if receipt.schemas_equivalent != equivalent:
        raise ValueError("schemas_equivalent mismatch")
    if equivalent and len(receipt.mismatches) != 0:
        raise ValueError("mismatch records forbidden when schemas_equivalent is True")
    if (not equivalent) and len(receipt.mismatches) == 0:
        raise ValueError("mismatch records required when schemas_equivalent is False")
    _check_no_forbidden_runtime_semantics(_base_payload(receipt.__dict__, "schema_equivalence_receipt_hash"))
    _validate_hash_format(receipt.schema_equivalence_receipt_hash, "schema_equivalence_receipt_hash")
    expected = _hash_payload(_base_payload(receipt.__dict__, "schema_equivalence_receipt_hash"))
    if expected != receipt.schema_equivalence_receipt_hash:
        raise ValueError("schema equivalence receipt hash mismatch")
