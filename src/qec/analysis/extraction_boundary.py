from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION = "v151.0"
_STATUS_CONSISTENT = "CONSISTENT"
_STATUS_DETERMINISM_VIOLATION = "DETERMINISM_VIOLATION"
_ALLOWED_STATUS = (_STATUS_CONSISTENT, _STATUS_DETERMINISM_VIOLATION)

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _invalid() -> ValueError:
    return ValueError("INVALID_INPUT")


def _validate_sha256_hex(value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _invalid()


def _validate_non_empty_str(value: str) -> None:
    if not isinstance(value, str) or value == "":
        raise _invalid()


def _validate_query_fields(query_fields: tuple[str, ...]) -> None:
    if not isinstance(query_fields, tuple) or len(query_fields) == 0:
        raise _invalid()
    for field in query_fields:
        if not isinstance(field, str) or field == "":
            raise _invalid()
    if len(set(query_fields)) != len(query_fields):
        raise _invalid()


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _invalid()
        return value
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, dict):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise _invalid()
        return {key: _canonicalize_json(value[key]) for key in sorted(keys)}
    raise _invalid()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExtractionInput:
    raw_bytes_hash: str
    source_type: str
    extraction_config_hash: str
    query_fields: tuple[str, ...]
    input_hash: str

    def __post_init__(self) -> None:
        _validate_sha256_hex(self.raw_bytes_hash)
        _validate_non_empty_str(self.source_type)
        _validate_sha256_hex(self.extraction_config_hash)
        _validate_query_fields(self.query_fields)
        _validate_sha256_hex(self.input_hash)
        if self.computed_stable_hash() != self.input_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "raw_bytes_hash": self.raw_bytes_hash,
            "source_type": self.source_type,
            "extraction_config_hash": self.extraction_config_hash,
            "query_fields": self.query_fields,
            "input_hash": self.input_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex({
            "raw_bytes_hash": self.raw_bytes_hash,
            "source_type": self.source_type,
            "extraction_config_hash": self.extraction_config_hash,
            "query_fields": self.query_fields,
        })


@dataclass(frozen=True)
class ExtractionConfigContract:
    contract_version: str
    backend_name: str
    backend_version: str
    schema_hash: str
    locale: str
    query_fields: tuple[str, ...]
    config_hash: str

    def __post_init__(self) -> None:
        _validate_non_empty_str(self.contract_version)
        _validate_non_empty_str(self.backend_name)
        _validate_non_empty_str(self.backend_version)
        _validate_sha256_hex(self.schema_hash)
        _validate_non_empty_str(self.locale)
        _validate_query_fields(self.query_fields)
        _validate_sha256_hex(self.config_hash)
        if self.computed_stable_hash() != self.config_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "contract_version": self.contract_version,
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "schema_hash": self.schema_hash,
            "locale": self.locale,
            "query_fields": self.query_fields,
            "config_hash": self.config_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex({
            "contract_version": self.contract_version,
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "schema_hash": self.schema_hash,
            "locale": self.locale,
            "query_fields": self.query_fields,
        })


@dataclass(frozen=True)
class ExtractedField:
    """Raw untrusted field record for v151.0.

    Contains only ``field_name`` and ``raw_value``; no normalization or
    semantic processing occurs in this boundary layer.
    """

    field_name: str
    raw_value: _JSONScalar

    def __post_init__(self) -> None:
        if not isinstance(self.field_name, str) or self.field_name == "":
            raise _invalid()
        if isinstance(self.raw_value, float) and not math.isfinite(self.raw_value):
            raise _invalid()
        if not isinstance(self.raw_value, (str, int, float, bool, type(None))):
            raise _invalid()

    def to_dict(self) -> dict[str, _JSONValue]:
        return {"field_name": self.field_name, "raw_value": self.raw_value}


@dataclass(frozen=True)
class ExtractionResult:
    extracted_fields: tuple[ExtractedField, ...]
    extraction_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.extracted_fields, tuple):
            raise _invalid()
        if len(self.extracted_fields) == 0:
            raise _invalid()
        for field in self.extracted_fields:
            if not isinstance(field, ExtractedField):
                raise _invalid()
        names = tuple(field.field_name for field in self.extracted_fields)
        if len(set(names)) != len(names):
            raise _invalid()
        _validate_sha256_hex(self.extraction_hash)
        if self.computed_stable_hash() != self.extraction_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "extracted_fields": tuple(field.to_dict() for field in self.extracted_fields),
            "extraction_hash": self.extraction_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex({"extracted_fields": tuple(field.to_dict() for field in self.extracted_fields)})


@dataclass(frozen=True)
class ExtractionReceipt:
    version: str
    raw_bytes_hash: str
    extraction_config_hash: str
    input_hash: str
    config_hash: str
    extraction_hash: str
    query_fields: tuple[str, ...]
    determinism_status: str
    stable_hash: str

    def __post_init__(self) -> None:
        if self.version != _VERSION:
            raise _invalid()
        _validate_sha256_hex(self.raw_bytes_hash)
        _validate_sha256_hex(self.extraction_config_hash)
        _validate_sha256_hex(self.input_hash)
        _validate_sha256_hex(self.config_hash)
        _validate_sha256_hex(self.extraction_hash)
        _validate_query_fields(self.query_fields)
        if self.determinism_status not in _ALLOWED_STATUS:
            raise _invalid()
        _validate_sha256_hex(self.stable_hash)
        if self.computed_stable_hash() != self.stable_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "version": self.version,
            "raw_bytes_hash": self.raw_bytes_hash,
            "extraction_config_hash": self.extraction_config_hash,
            "input_hash": self.input_hash,
            "config_hash": self.config_hash,
            "extraction_hash": self.extraction_hash,
            "query_fields": self.query_fields,
            "determinism_status": self.determinism_status,
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex({
            "version": self.version,
            "raw_bytes_hash": self.raw_bytes_hash,
            "extraction_config_hash": self.extraction_config_hash,
            "input_hash": self.input_hash,
            "config_hash": self.config_hash,
            "extraction_hash": self.extraction_hash,
            "query_fields": self.query_fields,
            "determinism_status": self.determinism_status,
        })


def run_extraction_boundary(
    extraction_input: ExtractionInput,
    extraction_config: ExtractionConfigContract,
    extraction_result: ExtractionResult,
    previous_extraction_hash: str | None = None,
) -> ExtractionReceipt:
    """Seal extraction artifacts into a deterministic v151.0 receipt.

    Determinism status semantics are explicit:
    - if ``previous_extraction_hash`` is ``None``: baseline ``CONSISTENT``
    - if it equals ``extraction_result.extraction_hash``: ``CONSISTENT``
    - otherwise: ``DETERMINISM_VIOLATION``
    """

    if not isinstance(extraction_input, ExtractionInput):
        raise _invalid()
    if not isinstance(extraction_config, ExtractionConfigContract):
        raise _invalid()
    if not isinstance(extraction_result, ExtractionResult):
        raise _invalid()
    if previous_extraction_hash is not None and not isinstance(previous_extraction_hash, str):
        raise _invalid()
    if previous_extraction_hash is not None:
        _validate_sha256_hex(previous_extraction_hash)

    if extraction_input.query_fields != extraction_config.query_fields:
        raise _invalid()

    extracted_field_names = tuple(field.field_name for field in extraction_result.extracted_fields)
    if extracted_field_names != extraction_input.query_fields:
        raise _invalid()

    determinism_status = _STATUS_CONSISTENT
    if previous_extraction_hash is not None and previous_extraction_hash != extraction_result.extraction_hash:
        determinism_status = _STATUS_DETERMINISM_VIOLATION

    base = {
        "version": _VERSION,
        "raw_bytes_hash": extraction_input.raw_bytes_hash,
        "extraction_config_hash": extraction_input.extraction_config_hash,
        "input_hash": extraction_input.input_hash,
        "config_hash": extraction_config.config_hash,
        "extraction_hash": extraction_result.extraction_hash,
        "query_fields": extraction_input.query_fields,
        "determinism_status": determinism_status,
    }
    return ExtractionReceipt(**base, stable_hash=_sha256_hex(base))
