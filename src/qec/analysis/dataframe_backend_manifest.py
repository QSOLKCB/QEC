from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

_SCHEMA_VERSION = "DATAFRAME_BACKEND_MANIFEST_V1"
_MAX_FIELDS = 4096
_MAX_NAME_LENGTH = 128
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

_ALLOWED_BACKEND_NAMES = {
    "PANDAS",
    "POLARS",
    "PYARROW",
    "NUMPY_STRUCTURED",
    "CSV_ADAPTER",
    "PARQUET_ADAPTER",
    "UNKNOWN",
}
_ALLOWED_EXECUTION_MODES = {"EAGER", "LAZY", "HYBRID"}
_ALLOWED_NULL_POLICIES = {
    "NULL_EQUALS_NULL",
    "NULL_NOT_EQUAL",
    "NULLS_FORBIDDEN",
    "DECLARED_NULLABLE",
}
_ALLOWED_ROUNDING_POLICIES = {
    "EXACT",
    "IEEE754",
    "FIXED_PRECISION",
    "ROUND_HALF_EVEN",
    "ROUND_HALF_UP",
    "DECLARED_EXTERNAL",
}
_ALLOWED_ORDERING_POLICIES = {
    "PRESERVE_INPUT_ORDER",
    "DECLARED_SORT_KEYS",
    "STABLE_CANONICAL_ORDER",
    "EXTERNAL_ORDER_DECLARED",
}


@dataclass(frozen=True)
class DataframeSchemaField:
    field_name: str
    declared_dtype: str
    nullable: bool
    field_position: int
    field_hash: str


@dataclass(frozen=True)
class DataframeSchemaManifest:
    fields: tuple[DataframeSchemaField, ...]
    schema_field_count: int
    schema_manifest_hash: str


@dataclass(frozen=True)
class DataframeExecutionPolicy:
    execution_mode: str
    lazy_execution_allowed: bool
    eager_execution_allowed: bool
    execution_policy_hash: str


@dataclass(frozen=True)
class DataframePrecisionPolicy:
    rounding_policy: str
    float_precision_bits: int
    precision_policy_hash: str


@dataclass(frozen=True)
class DataframeOrderingPolicy:
    ordering_policy: str
    declared_sort_keys: tuple[str, ...] = field(default_factory=tuple)
    ordering_policy_hash: str = ""


@dataclass(frozen=True)
class DataframeNullPolicy:
    null_policy: str
    allow_null_values: bool
    null_policy_hash: str


@dataclass(frozen=True)
class DataframeBackendManifest:
    schema_version: str
    backend_name: str
    backend_version: str
    adapter_only: bool
    schema_manifest: DataframeSchemaManifest
    execution_policy: DataframeExecutionPolicy
    precision_policy: DataframePrecisionPolicy
    ordering_policy: DataframeOrderingPolicy
    null_policy: DataframeNullPolicy
    dataframe_backend_manifest_hash: str


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _base_payload(payload: Mapping[str, Any], hash_key: str) -> dict[str, Any]:
    base = dict(payload)
    base.pop(hash_key, None)
    return base


def _validate_strict_bool(value: Any, name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean, got {type(value).__name__}")


def _validate_strict_int(value: Any, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")


def _validate_strict_str(value: Any, name: str) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")


def _validate_hash_format(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character hex digest")


def _validate_dense_positions(fields: Sequence[DataframeSchemaField]) -> None:
    positions = sorted(f.field_position for f in fields)
    if positions != list(range(len(fields))):
        raise ValueError("schema field positions must be dense and zero-indexed")


def _validate_field_uniqueness(fields: Sequence[DataframeSchemaField]) -> None:
    names = [f.field_name for f in fields]
    positions = [f.field_position for f in fields]
    if len(names) != len(set(names)):
        raise ValueError("duplicate field names are not allowed")
    if len(positions) != len(set(positions)):
        raise ValueError("duplicate field positions are not allowed")


def build_schema_field(field_name: str, declared_dtype: str, nullable: bool, field_position: int) -> DataframeSchemaField:
    _validate_strict_str(field_name, "field_name")
    _validate_strict_str(declared_dtype, "declared_dtype")
    _validate_strict_bool(nullable, "nullable")
    _validate_strict_int(field_position, "field_position")
    payload = {
        "field_name": field_name,
        "declared_dtype": declared_dtype,
        "nullable": nullable,
        "field_position": field_position,
    }
    field_hash = _hash_payload(payload)
    item = DataframeSchemaField(**payload, field_hash=field_hash)
    validate_schema_field(item)
    return item


def validate_schema_field(field_obj: DataframeSchemaField) -> None:
    if not isinstance(field_obj, DataframeSchemaField):
        raise TypeError("schema field must be DataframeSchemaField")
    _validate_strict_str(field_obj.field_name, "field_name")
    _validate_strict_str(field_obj.declared_dtype, "declared_dtype")
    _validate_strict_bool(field_obj.nullable, "nullable")
    _validate_strict_int(field_obj.field_position, "field_position")
    if not field_obj.field_name or len(field_obj.field_name) > _MAX_NAME_LENGTH:
        raise ValueError("field_name must be non-empty and <= max length")
    if not field_obj.declared_dtype or len(field_obj.declared_dtype) > _MAX_NAME_LENGTH:
        raise ValueError("declared_dtype must be non-empty and <= max length")
    if field_obj.field_position < 0:
        raise ValueError("field_position must be non-negative")
    _validate_hash_format(field_obj.field_hash, "field_hash")
    expected = _hash_payload(_base_payload(field_obj.__dict__, "field_hash"))
    if expected != field_obj.field_hash:
        raise ValueError("schema field hash mismatch")


def build_schema_manifest(fields: Sequence[DataframeSchemaField]) -> DataframeSchemaManifest:
    if len(fields) > _MAX_FIELDS:
        raise ValueError("schema exceeds max fields")
    normalized = tuple(sorted(tuple(fields), key=lambda f: (f.field_position, f.field_name)))
    _validate_field_uniqueness(normalized)
    _validate_dense_positions(normalized)
    for item in normalized:
        validate_schema_field(item)
    payload = {
        "fields": [f.__dict__ for f in normalized],
        "schema_field_count": len(normalized),
    }
    return DataframeSchemaManifest(fields=normalized, schema_field_count=len(normalized), schema_manifest_hash=_hash_payload(payload))


def validate_schema_manifest(manifest: DataframeSchemaManifest) -> None:
    if not isinstance(manifest, DataframeSchemaManifest):
        raise TypeError("schema manifest must be DataframeSchemaManifest")
    if not isinstance(manifest.fields, tuple):
        raise TypeError("fields must be a tuple")
    _validate_strict_int(manifest.schema_field_count, "schema_field_count")
    if len(manifest.fields) > _MAX_FIELDS:
        raise ValueError("schema exceeds max fields")
    for item in manifest.fields:
        validate_schema_field(item)
    _validate_field_uniqueness(manifest.fields)
    _validate_dense_positions(manifest.fields)
    # Enforce canonical field ordering: fields must be sorted by (field_position, field_name)
    canonical_order = tuple(sorted(manifest.fields, key=lambda f: (f.field_position, f.field_name)))
    if manifest.fields != canonical_order:
        raise ValueError("fields must be in canonical order sorted by (field_position, field_name)")
    recomputed_count = len(manifest.fields)
    if manifest.schema_field_count != recomputed_count:
        raise ValueError("schema field count mismatch")
    _validate_hash_format(manifest.schema_manifest_hash, "schema_manifest_hash")
    expected = _hash_payload({"fields": [f.__dict__ for f in manifest.fields], "schema_field_count": recomputed_count})
    if expected != manifest.schema_manifest_hash:
        raise ValueError("schema manifest hash mismatch")


def build_execution_policy(execution_mode: str, lazy_execution_allowed: bool, eager_execution_allowed: bool) -> DataframeExecutionPolicy:
    _validate_strict_str(execution_mode, "execution_mode")
    _validate_strict_bool(lazy_execution_allowed, "lazy_execution_allowed")
    _validate_strict_bool(eager_execution_allowed, "eager_execution_allowed")
    payload = {
        "execution_mode": execution_mode,
        "lazy_execution_allowed": lazy_execution_allowed,
        "eager_execution_allowed": eager_execution_allowed,
    }
    obj = DataframeExecutionPolicy(**payload, execution_policy_hash=_hash_payload(payload))
    validate_execution_policy(obj)
    return obj


def validate_execution_policy(policy: DataframeExecutionPolicy) -> None:
    if not isinstance(policy, DataframeExecutionPolicy):
        raise TypeError("execution policy must be DataframeExecutionPolicy")
    _validate_strict_str(policy.execution_mode, "execution_mode")
    _validate_strict_bool(policy.lazy_execution_allowed, "lazy_execution_allowed")
    _validate_strict_bool(policy.eager_execution_allowed, "eager_execution_allowed")
    if policy.execution_mode not in _ALLOWED_EXECUTION_MODES:
        raise ValueError("invalid execution mode")
    _validate_hash_format(policy.execution_policy_hash, "execution_policy_hash")
    expected = _hash_payload(_base_payload(policy.__dict__, "execution_policy_hash"))
    if expected != policy.execution_policy_hash:
        raise ValueError("execution policy hash mismatch")


def build_precision_policy(rounding_policy: str, float_precision_bits: int) -> DataframePrecisionPolicy:
    _validate_strict_str(rounding_policy, "rounding_policy")
    _validate_strict_int(float_precision_bits, "float_precision_bits")
    payload = {"rounding_policy": rounding_policy, "float_precision_bits": float_precision_bits}
    obj = DataframePrecisionPolicy(**payload, precision_policy_hash=_hash_payload(payload))
    validate_precision_policy(obj)
    return obj


def validate_precision_policy(policy: DataframePrecisionPolicy) -> None:
    if not isinstance(policy, DataframePrecisionPolicy):
        raise TypeError("precision policy must be DataframePrecisionPolicy")
    _validate_strict_str(policy.rounding_policy, "rounding_policy")
    _validate_strict_int(policy.float_precision_bits, "float_precision_bits")
    if policy.rounding_policy not in _ALLOWED_ROUNDING_POLICIES:
        raise ValueError("invalid rounding policy")
    if policy.float_precision_bits <= 0:
        raise ValueError("float_precision_bits must be positive")
    _validate_hash_format(policy.precision_policy_hash, "precision_policy_hash")
    if _hash_payload(_base_payload(policy.__dict__, "precision_policy_hash")) != policy.precision_policy_hash:
        raise ValueError("precision policy hash mismatch")


def build_ordering_policy(ordering_policy: str, declared_sort_keys: Sequence[str]) -> DataframeOrderingPolicy:
    _validate_strict_str(ordering_policy, "ordering_policy")
    # Reject plain strings since str is a Sequence[str] but would split into characters
    if isinstance(declared_sort_keys, str):
        raise TypeError("declared_sort_keys must be a sequence of strings, not a single string")
    keys = tuple(declared_sort_keys)
    for key in keys:
        _validate_strict_str(key, "declared_sort_keys element")
    payload = {"ordering_policy": ordering_policy, "declared_sort_keys": list(keys)}
    obj = DataframeOrderingPolicy(ordering_policy=ordering_policy, declared_sort_keys=keys, ordering_policy_hash=_hash_payload(payload))
    validate_ordering_policy(obj)
    return obj


def validate_ordering_policy(policy: DataframeOrderingPolicy) -> None:
    if not isinstance(policy, DataframeOrderingPolicy):
        raise TypeError("ordering policy must be DataframeOrderingPolicy")
    _validate_strict_str(policy.ordering_policy, "ordering_policy")
    if not isinstance(policy.declared_sort_keys, tuple):
        raise TypeError("declared_sort_keys must be a tuple")
    for key in policy.declared_sort_keys:
        _validate_strict_str(key, "declared_sort_keys element")
    if policy.ordering_policy not in _ALLOWED_ORDERING_POLICIES:
        raise ValueError("invalid ordering policy")
    if len(policy.declared_sort_keys) != len(set(policy.declared_sort_keys)):
        raise ValueError("declared_sort_keys must be unique")
    _validate_hash_format(policy.ordering_policy_hash, "ordering_policy_hash")
    if _hash_payload({"ordering_policy": policy.ordering_policy, "declared_sort_keys": list(policy.declared_sort_keys)}) != policy.ordering_policy_hash:
        raise ValueError("ordering policy hash mismatch")


def build_null_policy(null_policy: str, allow_null_values: bool) -> DataframeNullPolicy:
    _validate_strict_str(null_policy, "null_policy")
    _validate_strict_bool(allow_null_values, "allow_null_values")
    # Disallow contradictory null policy combinations
    if null_policy == "NULLS_FORBIDDEN" and allow_null_values:
        raise ValueError("contradictory null policy: NULLS_FORBIDDEN cannot have allow_null_values=True")
    payload = {"null_policy": null_policy, "allow_null_values": allow_null_values}
    obj = DataframeNullPolicy(**payload, null_policy_hash=_hash_payload(payload))
    validate_null_policy(obj)
    return obj


def validate_null_policy(policy: DataframeNullPolicy) -> None:
    if not isinstance(policy, DataframeNullPolicy):
        raise TypeError("null policy must be DataframeNullPolicy")
    _validate_strict_str(policy.null_policy, "null_policy")
    _validate_strict_bool(policy.allow_null_values, "allow_null_values")
    if policy.null_policy not in _ALLOWED_NULL_POLICIES:
        raise ValueError("invalid null policy")
    # Disallow contradictory null policy combinations
    if policy.null_policy == "NULLS_FORBIDDEN" and policy.allow_null_values:
        raise ValueError("contradictory null policy: NULLS_FORBIDDEN cannot have allow_null_values=True")
    _validate_hash_format(policy.null_policy_hash, "null_policy_hash")
    if _hash_payload(_base_payload(policy.__dict__, "null_policy_hash")) != policy.null_policy_hash:
        raise ValueError("null policy hash mismatch")


def build_dataframe_backend_manifest(
    backend_name: str,
    backend_version: str,
    adapter_only: bool,
    schema_manifest: DataframeSchemaManifest,
    execution_policy: DataframeExecutionPolicy,
    precision_policy: DataframePrecisionPolicy,
    ordering_policy: DataframeOrderingPolicy,
    null_policy: DataframeNullPolicy,
) -> DataframeBackendManifest:
    _validate_strict_str(backend_name, "backend_name")
    _validate_strict_str(backend_version, "backend_version")
    _validate_strict_bool(adapter_only, "adapter_only")
    if not backend_version:
        raise ValueError("backend_version must be non-empty")
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "backend_name": backend_name,
        "backend_version": backend_version,
        "adapter_only": adapter_only,
        "schema_manifest": schema_manifest.__dict__ | {"fields": [f.__dict__ for f in schema_manifest.fields]},
        "execution_policy": execution_policy.__dict__,
        "precision_policy": precision_policy.__dict__,
        "ordering_policy": {"ordering_policy": ordering_policy.ordering_policy, "declared_sort_keys": list(ordering_policy.declared_sort_keys), "ordering_policy_hash": ordering_policy.ordering_policy_hash},
        "null_policy": null_policy.__dict__,
    }
    manifest_hash = _hash_payload(payload)
    obj = DataframeBackendManifest(
        schema_version=_SCHEMA_VERSION,
        backend_name=backend_name,
        backend_version=backend_version,
        adapter_only=adapter_only,
        schema_manifest=schema_manifest,
        execution_policy=execution_policy,
        precision_policy=precision_policy,
        ordering_policy=ordering_policy,
        null_policy=null_policy,
        dataframe_backend_manifest_hash=manifest_hash,
    )
    validate_dataframe_backend_manifest(obj)
    return obj


def validate_dataframe_backend_manifest(manifest: DataframeBackendManifest) -> None:
    if not isinstance(manifest, DataframeBackendManifest):
        raise TypeError("manifest must be DataframeBackendManifest")
    _validate_strict_str(manifest.schema_version, "schema_version")
    _validate_strict_str(manifest.backend_name, "backend_name")
    _validate_strict_str(manifest.backend_version, "backend_version")
    _validate_strict_bool(manifest.adapter_only, "adapter_only")
    if not manifest.backend_version:
        raise ValueError("backend_version must be non-empty")
    if manifest.schema_version != _SCHEMA_VERSION:
        raise ValueError("unsupported schema version")
    if manifest.backend_name not in _ALLOWED_BACKEND_NAMES:
        raise ValueError("invalid backend name")
    if manifest.adapter_only is not True:
        raise ValueError("adapter_only must be true")
    # Validate policy objects are correct types before dereferencing
    if not isinstance(manifest.schema_manifest, DataframeSchemaManifest):
        raise TypeError("schema_manifest must be DataframeSchemaManifest")
    if not isinstance(manifest.execution_policy, DataframeExecutionPolicy):
        raise TypeError("execution_policy must be DataframeExecutionPolicy")
    if not isinstance(manifest.precision_policy, DataframePrecisionPolicy):
        raise TypeError("precision_policy must be DataframePrecisionPolicy")
    if not isinstance(manifest.ordering_policy, DataframeOrderingPolicy):
        raise TypeError("ordering_policy must be DataframeOrderingPolicy")
    if not isinstance(manifest.null_policy, DataframeNullPolicy):
        raise TypeError("null_policy must be DataframeNullPolicy")
    validate_schema_manifest(manifest.schema_manifest)
    validate_execution_policy(manifest.execution_policy)
    validate_precision_policy(manifest.precision_policy)
    validate_ordering_policy(manifest.ordering_policy)
    validate_null_policy(manifest.null_policy)
    # Validate declared_sort_keys exist in schema fields
    if manifest.ordering_policy.ordering_policy == "DECLARED_SORT_KEYS":
        schema_field_names = {f.field_name for f in manifest.schema_manifest.fields}
        for key in manifest.ordering_policy.declared_sort_keys:
            if key not in schema_field_names:
                raise ValueError(f"declared_sort_key '{key}' not found in schema fields")
    # Reconcile null policy with field nullability
    if manifest.null_policy.null_policy == "NULLS_FORBIDDEN" or not manifest.null_policy.allow_null_values:
        for field_obj in manifest.schema_manifest.fields:
            if field_obj.nullable:
                raise ValueError(f"field '{field_obj.field_name}' is nullable but null policy forbids nulls")
    _validate_hash_format(manifest.dataframe_backend_manifest_hash, "dataframe_backend_manifest_hash")
    payload = {
        "schema_version": manifest.schema_version,
        "backend_name": manifest.backend_name,
        "backend_version": manifest.backend_version,
        "adapter_only": manifest.adapter_only,
        "schema_manifest": manifest.schema_manifest.__dict__ | {"fields": [f.__dict__ for f in manifest.schema_manifest.fields]},
        "execution_policy": manifest.execution_policy.__dict__,
        "precision_policy": manifest.precision_policy.__dict__,
        "ordering_policy": {"ordering_policy": manifest.ordering_policy.ordering_policy, "declared_sort_keys": list(manifest.ordering_policy.declared_sort_keys), "ordering_policy_hash": manifest.ordering_policy.ordering_policy_hash},
        "null_policy": manifest.null_policy.__dict__,
    }
    if _hash_payload(payload) != manifest.dataframe_backend_manifest_hash:
        raise ValueError("dataframe backend manifest hash mismatch")
