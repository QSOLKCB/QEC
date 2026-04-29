from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP
import json
import re
from types import MappingProxyType
import unicodedata
from typing import Any, Mapping

from qec.analysis.canonical_hashing import CanonicalHashingError, canonical_json, canonicalize_json, sha256_hex
from qec.analysis.extraction_boundary import ExtractedField, ExtractionResult

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VERSION = "v151.1"
_STATUS = "CANONICALIZED"
_FIELD_TYPES = {"STRING", "INTEGER", "DECIMAL", "BOOLEAN", "DATE", "MONEY", "JSON"}
_ROUNDING = {"HALF_EVEN": ROUND_HALF_EVEN, "HALF_UP": ROUND_HALF_UP, "TRUNCATE": ROUND_DOWN}


def _invalid() -> ValueError:
    return ValueError("INVALID_INPUT")


def _canonical_json(value: Any) -> str:
    try:
        return canonical_json(value)
    except CanonicalHashingError as exc:
        raise _invalid() from exc


def _sha256_hex(value: Any) -> str:
    try:
        return sha256_hex(value)
    except CanonicalHashingError as exc:
        raise _invalid() from exc


def _validate_sha(value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _invalid()


def _validate_non_empty_string(value: str) -> None:
    if not isinstance(value, str) or value == "":
        raise _invalid()


def _normalize_string(value: str, required: bool) -> str | None:
    normalized = unicodedata.normalize("NFC", value).replace("\u00A0", " ")
    normalized = " ".join(normalized.split())
    if normalized == "":
        if required:
            raise _invalid()
        return None
    return normalized


def _validate_grouped_integer_part(integer_part: str, thousands_sep: str) -> None:
    groups = integer_part.split(thousands_sep)
    if any(group == "" for group in groups):
        raise _invalid()
    if not groups[0].isdigit() or len(groups[0]) > 3:
        raise _invalid()
    if any((not group.isdigit()) or len(group) != 3 for group in groups[1:]):
        raise _invalid()


def _parse_decimal_text(raw: Any, decimal_sep: str, thousands_sep: str) -> Decimal:
    if isinstance(raw, bool):
        raise _invalid()
    if isinstance(raw, int):
        return Decimal(raw)
    if isinstance(raw, float):
        raise _invalid()
    if not isinstance(raw, str):
        raise _invalid()

    txt = raw.strip()
    if txt == "":
        raise _invalid()

    normalized_txt = txt.replace("\u00A0", " ")
    normalized_thousands_sep = thousands_sep.replace("\u00A0", " ")

    if normalized_thousands_sep:
        decimal_parts = normalized_txt.split(decimal_sep)
        if len(decimal_parts) > 2:
            raise _invalid()
        integer_part = decimal_parts[0]
        if normalized_thousands_sep in integer_part:
            _validate_grouped_integer_part(integer_part, normalized_thousands_sep)
        normalized_txt = normalized_txt.replace(normalized_thousands_sep, "")

    normalized_txt = normalized_txt.replace(decimal_sep, ".")
    try:
        dec = Decimal(normalized_txt)
    except InvalidOperation as exc:
        raise _invalid() from exc
    if not dec.is_finite():
        raise _invalid()
    return dec


def _json_pairs_hook(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if not isinstance(key, str) or key == "" or key in out:
            raise _invalid()
        out[key] = value
    return out


def _extract_json_span(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
        if m is None:
            raise _invalid()
        stripped = m.group(1).strip()
    decoder = json.JSONDecoder(object_pairs_hook=_json_pairs_hook)
    span_start: int | None = None
    span_text: str | None = None
    for start, char in enumerate(stripped):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(stripped, idx=start)
        except Exception:
            continue
        if stripped[end:].strip() != "":
            continue
        if span_start is not None:
            raise _invalid()
        span_start = start
        span_text = stripped[start:end]
    if span_start is None or span_text is None:
        raise _invalid()
    if any(char in "{[" for char in stripped[:span_start]):
        raise _invalid()
    return span_text


def _normalize_json(raw: Any) -> Any:
    if not isinstance(raw, str):
        raise _invalid()
    span = _extract_json_span(raw)
    try:
        value = json.loads(span, object_pairs_hook=_json_pairs_hook, parse_constant=lambda _x: (_ for _ in ()).throw(_invalid()))
    except Exception as exc:
        raise _invalid() from exc
    try:
        return canonicalize_json(value)
    except CanonicalHashingError as exc:
        raise _invalid() from exc


def _validate_numeric_bound(field_type: str, bound: str | int | None) -> None:
    if bound is None:
        return
    try:
        if field_type == "INTEGER":
            if isinstance(bound, int):
                return
            if not isinstance(bound, str):
                raise _invalid()
            Decimal(bound)
            int(bound)
            return
        if field_type == "DECIMAL":
            Decimal(str(bound))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise _invalid() from exc


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class CanonicalFieldSpec:
    field_name: str
    field_type: str
    required: bool
    format_pattern: str | None
    min_value: str | int | None
    max_value: str | int | None
    field_hash: str

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.field_name)
        if self.field_type not in _FIELD_TYPES or not isinstance(self.required, bool):
            raise _invalid()
        if self.format_pattern is not None and not isinstance(self.format_pattern, str):
            raise _invalid()
        if self.min_value is not None and isinstance(self.min_value, bool):
            raise _invalid()
        if self.max_value is not None and isinstance(self.max_value, bool):
            raise _invalid()
        _validate_numeric_bound(self.field_type, self.min_value)
        _validate_numeric_bound(self.field_type, self.max_value)
        _validate_sha(self.field_hash)
        if self.computed_stable_hash() != self.field_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "field_type": self.field_type,
            "required": self.required,
            "format_pattern": self.format_pattern,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "field_hash": self.field_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex({k: v for k, v in self.to_dict().items() if k != "field_hash"})


@dataclass(frozen=True)
class CanonicalizationSchema:
    schema_version: str
    document_type: str
    fields: tuple[CanonicalFieldSpec, ...]
    schema_hash: str

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.schema_version)
        _validate_non_empty_string(self.document_type)
        if not isinstance(self.fields, tuple) or not self.fields:
            raise _invalid()
        names = []
        for field in self.fields:
            if not isinstance(field, CanonicalFieldSpec):
                raise _invalid()
            names.append(field.field_name)
        if len(set(names)) != len(names):
            raise _invalid()
        _validate_sha(self.schema_hash)
        if self.computed_stable_hash() != self.schema_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "fields": tuple(f.to_dict() for f in self.fields),
            "schema_hash": self.schema_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex(
            {
                "schema_version": self.schema_version,
                "document_type": self.document_type,
                "fields": tuple(f.to_dict() for f in self.fields),
            }
        )


@dataclass(frozen=True)
class LocaleContract:
    locale_id: str
    date_format: str
    decimal_separator: str
    thousands_separator: str
    currency_code: str
    currency_minor_unit_exponent: int
    rounding_mode: str
    locale_hash: str

    def __post_init__(self) -> None:
        for s in (
            self.locale_id,
            self.date_format,
            self.decimal_separator,
            self.currency_code,
            self.rounding_mode,
        ):
            _validate_non_empty_string(s)
        if self.decimal_separator not in {".", ","}:
            raise _invalid()
        if self.thousands_separator not in {",", ".", " ", "\u00A0", ""}:
            raise _invalid()
        if self.decimal_separator == self.thousands_separator:
            raise _invalid()
        if not re.fullmatch(r"[A-Z]{3}", self.currency_code):
            raise _invalid()
        if (
            not isinstance(self.currency_minor_unit_exponent, int)
            or isinstance(self.currency_minor_unit_exponent, bool)
            or not (0 <= self.currency_minor_unit_exponent <= 4)
        ):
            raise _invalid()
        if self.rounding_mode not in _ROUNDING:
            raise _invalid()
        _validate_sha(self.locale_hash)
        if self.computed_stable_hash() != self.locale_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "locale_id": self.locale_id,
            "date_format": self.date_format,
            "decimal_separator": self.decimal_separator,
            "thousands_separator": self.thousands_separator,
            "currency_code": self.currency_code,
            "currency_minor_unit_exponent": self.currency_minor_unit_exponent,
            "rounding_mode": self.rounding_mode,
            "locale_hash": self.locale_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex({k: v for k, v in self.to_dict().items() if k != "locale_hash"})


@dataclass(frozen=True)
class CanonicalDocument:
    version: str
    extraction_hash: str
    schema_hash: str
    locale_hash: str
    canonical_payload: Mapping[str, Any]
    canonical_json: str
    canonical_hash: str

    def __post_init__(self) -> None:
        if self.version != _VERSION:
            raise _invalid()
        _validate_sha(self.extraction_hash)
        _validate_sha(self.schema_hash)
        _validate_sha(self.locale_hash)
        _validate_sha(self.canonical_hash)
        if not isinstance(self.canonical_payload, Mapping):
            raise _invalid()
        frozen_payload = _deep_freeze(dict(self.canonical_payload))
        object.__setattr__(self, "canonical_payload", frozen_payload)
        if self.canonical_json != _canonical_json(self.canonical_payload):
            raise _invalid()
        if self.canonical_hash != _sha256_hex(self.canonical_payload):
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "extraction_hash": self.extraction_hash,
            "schema_hash": self.schema_hash,
            "locale_hash": self.locale_hash,
            "canonical_payload": canonicalize_json(self.canonical_payload),
            "canonical_json": self.canonical_json,
            "canonical_hash": self.canonical_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex({k: v for k, v in self.to_dict().items() if k != "canonical_hash"})


@dataclass(frozen=True)
class CanonicalizationReceipt:
    version: str
    extraction_hash: str
    schema_hash: str
    locale_hash: str
    canonical_hash: str
    canonical_document: CanonicalDocument
    status: str
    stable_hash: str

    def __post_init__(self) -> None:
        if self.version != _VERSION or self.status != _STATUS or not isinstance(self.canonical_document, CanonicalDocument):
            raise _invalid()
        _validate_sha(self.extraction_hash)
        _validate_sha(self.schema_hash)
        _validate_sha(self.locale_hash)
        _validate_sha(self.canonical_hash)
        _validate_sha(self.stable_hash)
        if self.canonical_document.canonical_hash != self.canonical_hash:
            raise _invalid()
        if (
            self.canonical_document.extraction_hash != self.extraction_hash
            or self.canonical_document.schema_hash != self.schema_hash
            or self.canonical_document.locale_hash != self.locale_hash
        ):
            raise _invalid()
        if self.computed_stable_hash() != self.stable_hash:
            raise _invalid()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "extraction_hash": self.extraction_hash,
            "schema_hash": self.schema_hash,
            "locale_hash": self.locale_hash,
            "canonical_hash": self.canonical_hash,
            "canonical_document": self.canonical_document.to_dict(),
            "status": self.status,
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def computed_stable_hash(self) -> str:
        return _sha256_hex({k: v for k, v in self.to_dict().items() if k != "stable_hash"})


def run_canonicalization_engine(extraction_result: ExtractionResult, schema: CanonicalizationSchema, locale: LocaleContract) -> CanonicalizationReceipt:
    if not isinstance(extraction_result, ExtractionResult) or not isinstance(schema, CanonicalizationSchema) or not isinstance(locale, LocaleContract):
        raise _invalid()
    fields = extraction_result.extracted_fields
    if len(fields) != len(schema.fields):
        raise _invalid()
    payload: dict[str, Any] = {}
    for i, spec in enumerate(schema.fields):
        ef = fields[i]
        if not isinstance(ef, ExtractedField) or ef.field_name != spec.field_name:
            raise _invalid()
        raw = ef.raw_value
        if raw is None:
            if spec.required:
                raise _invalid()
            payload[spec.field_name] = None
            continue
        if spec.field_type == "STRING":
            if not isinstance(raw, str):
                raise _invalid()
            payload[spec.field_name] = _normalize_string(raw, spec.required)
        elif spec.field_type == "INTEGER":
            dec = _parse_decimal_text(raw, locale.decimal_separator, locale.thousands_separator)
            if dec != dec.to_integral_value():
                raise _invalid()
            val = int(dec)
            if spec.min_value is not None and val < int(spec.min_value):
                raise _invalid()
            if spec.max_value is not None and val > int(spec.max_value):
                raise _invalid()
            payload[spec.field_name] = val
        elif spec.field_type == "DECIMAL":
            dec = _parse_decimal_text(raw, locale.decimal_separator, locale.thousands_separator)
            if spec.min_value is not None and dec < Decimal(str(spec.min_value)):
                raise _invalid()
            if spec.max_value is not None and dec > Decimal(str(spec.max_value)):
                raise _invalid()
            payload[spec.field_name] = format(dec.normalize(), "f").rstrip("0").rstrip(".") if dec != 0 else "0"
        elif spec.field_type == "BOOLEAN":
            t = str(raw).strip().lower()
            if t in {"true", "yes", "1", "y"}:
                payload[spec.field_name] = True
            elif t in {"false", "no", "0", "n"}:
                payload[spec.field_name] = False
            else:
                raise _invalid()
        elif spec.field_type == "DATE":
            if not isinstance(raw, str):
                raise _invalid()
            try:
                payload[spec.field_name] = datetime.strptime(raw.strip(), locale.date_format).date().isoformat()
            except Exception as exc:
                raise _invalid() from exc
        elif spec.field_type == "MONEY":
            if not isinstance(raw, str):
                raise _invalid()
            text = raw.strip()
            parts = text.split()
            amount_text = text
            if len(parts) == 2:
                if parts[0] != locale.currency_code:
                    raise _invalid()
                amount_text = parts[1]
            dec = _parse_decimal_text(amount_text, locale.decimal_separator, locale.thousands_separator)
            if dec < 0:
                raise _invalid()
            quantum = Decimal(1).scaleb(-locale.currency_minor_unit_exponent)
            rounded = dec.quantize(quantum, rounding=_ROUNDING[locale.rounding_mode])
            payload[spec.field_name] = {
                "amount_minor_units": int(rounded.scaleb(locale.currency_minor_unit_exponent)),
                "currency_code": locale.currency_code,
                "minor_unit_exponent": locale.currency_minor_unit_exponent,
            }
        elif spec.field_type == "JSON":
            payload[spec.field_name] = _normalize_json(raw)
        else:
            raise _invalid()
    canonical_payload = canonicalize_json(payload)
    canonical_payload_dict = dict(canonical_payload)
    canonical_j = _canonical_json(canonical_payload_dict)
    canonical_h = _sha256_hex(canonical_payload_dict)
    doc_base = {
        "version": _VERSION,
        "extraction_hash": extraction_result.extraction_hash,
        "schema_hash": schema.schema_hash,
        "locale_hash": locale.locale_hash,
        "canonical_payload": canonical_payload_dict,
        "canonical_json": canonical_j,
    }
    doc = CanonicalDocument(**doc_base, canonical_hash=canonical_h)
    rec_base = {
        "version": _VERSION,
        "extraction_hash": extraction_result.extraction_hash,
        "schema_hash": schema.schema_hash,
        "locale_hash": locale.locale_hash,
        "canonical_hash": doc.canonical_hash,
        "canonical_document": doc,
        "status": _STATUS,
    }
    return CanonicalizationReceipt(**rec_base, stable_hash=_sha256_hex({**rec_base, "canonical_document": doc.to_dict()}))
