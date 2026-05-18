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
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _base_payload(payload: Mapping[str, Any], hash_key: str) -> dict[str, Any]:
    base = dict(payload)
    base.pop(hash_key, None)
    return base


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
    payload = {
        "field_name": field_name,
        "declared_dtype": declared_dtype,
        "nullable": bool(nullable),
        "field_position": int(field_position),
    }
    field_hash = _hash_payload(payload)
    item = DataframeSchemaField(**payload, field_hash=field_hash)
    validate_schema_field(item)
    return item


def validate_schema_field(field_obj: DataframeSchemaField) -> None:
    if not isinstance(field_obj, DataframeSchemaField):
        raise TypeError("schema field must be DataframeSchemaField")
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
    if len(manifest.fields) > _MAX_FIELDS:
        raise ValueError("schema exceeds max fields")
    for item in manifest.fields:
        validate_schema_field(item)
    _validate_field_uniqueness(manifest.fields)
    _validate_dense_positions(manifest.fields)
    recomputed_count = len(manifest.fields)
    if manifest.schema_field_count != recomputed_count:
        raise ValueError("schema field count mismatch")
    _validate_hash_format(manifest.schema_manifest_hash, "schema_manifest_hash")
    expected = _hash_payload({"fields": [f.__dict__ for f in manifest.fields], "schema_field_count": recomputed_count})
    if expected != manifest.schema_manifest_hash:
        raise ValueError("schema manifest hash mismatch")


def build_execution_policy(execution_mode: str, lazy_execution_allowed: bool, eager_execution_allowed: bool) -> DataframeExecutionPolicy:
    payload = {
        "execution_mode": execution_mode,
        "lazy_execution_allowed": bool(lazy_execution_allowed),
        "eager_execution_allowed": bool(eager_execution_allowed),
    }
    obj = DataframeExecutionPolicy(**payload, execution_policy_hash=_hash_payload(payload))
    validate_execution_policy(obj)
    return obj


def validate_execution_policy(policy: DataframeExecutionPolicy) -> None:
    if policy.execution_mode not in _ALLOWED_EXECUTION_MODES:
        raise ValueError("invalid execution mode")
    _validate_hash_format(policy.execution_policy_hash, "execution_policy_hash")
    expected = _hash_payload(_base_payload(policy.__dict__, "execution_policy_hash"))
    if expected != policy.execution_policy_hash:
        raise ValueError("execution policy hash mismatch")


def build_precision_policy(rounding_policy: str, float_precision_bits: int) -> DataframePrecisionPolicy:
    payload = {"rounding_policy": rounding_policy, "float_precision_bits": int(float_precision_bits)}
    obj = DataframePrecisionPolicy(**payload, precision_policy_hash=_hash_payload(payload))
    validate_precision_policy(obj)
    return obj


def validate_precision_policy(policy: DataframePrecisionPolicy) -> None:
    if policy.rounding_policy not in _ALLOWED_ROUNDING_POLICIES:
        raise ValueError("invalid rounding policy")
    if policy.float_precision_bits <= 0:
        raise ValueError("float_precision_bits must be positive")
    _validate_hash_format(policy.precision_policy_hash, "precision_policy_hash")
    if _hash_payload(_base_payload(policy.__dict__, "precision_policy_hash")) != policy.precision_policy_hash:
        raise ValueError("precision policy hash mismatch")


def build_ordering_policy(ordering_policy: str, declared_sort_keys: Sequence[str]) -> DataframeOrderingPolicy:
    keys = tuple(declared_sort_keys)
    payload = {"ordering_policy": ordering_policy, "declared_sort_keys": list(keys)}
    obj = DataframeOrderingPolicy(ordering_policy=ordering_policy, declared_sort_keys=keys, ordering_policy_hash=_hash_payload(payload))
    validate_ordering_policy(obj)
    return obj


def validate_ordering_policy(policy: DataframeOrderingPolicy) -> None:
    if policy.ordering_policy not in _ALLOWED_ORDERING_POLICIES:
        raise ValueError("invalid ordering policy")
    if len(policy.declared_sort_keys) != len(set(policy.declared_sort_keys)):
        raise ValueError("declared_sort_keys must be unique")
    _validate_hash_format(policy.ordering_policy_hash, "ordering_policy_hash")
    if _hash_payload({"ordering_policy": policy.ordering_policy, "declared_sort_keys": list(policy.declared_sort_keys)}) != policy.ordering_policy_hash:
        raise ValueError("ordering policy hash mismatch")


def build_null_policy(null_policy: str, allow_null_values: bool) -> DataframeNullPolicy:
    payload = {"null_policy": null_policy, "allow_null_values": bool(allow_null_values)}
    obj = DataframeNullPolicy(**payload, null_policy_hash=_hash_payload(payload))
    validate_null_policy(obj)
    return obj


def validate_null_policy(policy: DataframeNullPolicy) -> None:
    if policy.null_policy not in _ALLOWED_NULL_POLICIES:
        raise ValueError("invalid null policy")
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
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "backend_name": backend_name,
        "backend_version": backend_version,
        "adapter_only": bool(adapter_only),
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
        adapter_only=bool(adapter_only),
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
    if manifest.schema_version != _SCHEMA_VERSION:
        raise ValueError("unsupported schema version")
    if manifest.backend_name not in _ALLOWED_BACKEND_NAMES:
        raise ValueError("invalid backend name")
    if manifest.adapter_only is not True:
        raise ValueError("adapter_only must be true")
    validate_schema_manifest(manifest.schema_manifest)
    validate_execution_policy(manifest.execution_policy)
    validate_precision_policy(manifest.precision_policy)
    validate_ordering_policy(manifest.ordering_policy)
    validate_null_policy(manifest.null_policy)
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
