# SPDX-License-Identifier: MIT
"""v138.2.11 — deterministic multi-model invocation matrix layer."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

MULTI_MODEL_INVOCATION_MATRIX_VERSION = "v138.2.11"

SUPPORTED_EXECUTION_MODES: Tuple[str, ...] = ("planned", "simulated", "observed")
SUPPORTED_INVOCATION_STATUSES: Tuple[str, ...] = ("pending", "completed", "failed", "invalid")

CANONICAL_PROVIDER_NAMES: Tuple[str, ...] = ("openai", "sider", "anthropic", "xai")
CANONICAL_MODEL_NAMES: Tuple[str, ...] = ("chatgpt_native", "chatgpt_5_4_sider", "claude", "grok")


class InvocationMatrixValidationError(ValueError):
    """Raised when invocation matrix input violates deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise InvocationMatrixValidationError(f"{field} contains non-canonical numeric value")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda x: str(x)):
            key = str(raw_key)
            if key in normalized:
                raise InvocationMatrixValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise InvocationMatrixValidationError(f"{field} contains unsupported type: {type(value).__name__}")


def _normalize_required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise InvocationMatrixValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise InvocationMatrixValidationError(f"{field} must be non-empty")
    return normalized


def _normalize_non_negative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvocationMatrixValidationError(f"{field} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise InvocationMatrixValidationError(f"{field} must be >= 0")
    return normalized


def _normalize_metadata_mapping(value: Any, *, field: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InvocationMatrixValidationError(f"{field} must be a mapping")
    return _canonicalize_value(dict(value), field=field)


def _extract_prompt_hash(canonical_prompt_artifact: Any) -> str:
    if isinstance(canonical_prompt_artifact, Mapping):
        receipt = canonical_prompt_artifact.get("receipt", {})
        if isinstance(receipt, Mapping):
            prompt_hash = receipt.get("prompt_hash")
            if isinstance(prompt_hash, str) and prompt_hash:
                return prompt_hash
        prompt_hash = canonical_prompt_artifact.get("prompt_hash")
        if isinstance(prompt_hash, str) and prompt_hash:
            return prompt_hash
    prompt_receipt = getattr(canonical_prompt_artifact, "receipt", None)
    prompt_hash = getattr(prompt_receipt, "prompt_hash", None)
    if isinstance(prompt_hash, str) and prompt_hash:
        return prompt_hash
    prompt_hash = getattr(canonical_prompt_artifact, "prompt_hash", None)
    if isinstance(prompt_hash, str) and prompt_hash:
        return prompt_hash
    raise InvocationMatrixValidationError("canonical_prompt_artifact must provide receipt.prompt_hash")


def _ordered_records(records: Sequence["InvocationRecord"]) -> Tuple["InvocationRecord", ...]:
    return tuple(
        sorted(
            records,
            key=lambda r: (r.provider_name, r.model_name, int(r.repetition_index), r.invocation_id),
        )
    )


def _matrix_payload(prompt_hash: str, invocation_specs: Sequence["ModelInvocationSpec"], records: Sequence["InvocationRecord"]) -> Dict[str, Any]:
    return {
        "prompt_hash": prompt_hash,
        "invocation_specs": [spec.to_dict() for spec in invocation_specs],
        "records": [record.to_dict() for record in records],
    }


def _build_receipt(prompt_hash: str, invocation_specs: Sequence["ModelInvocationSpec"], records: Sequence["InvocationRecord"], *, validation_passed: bool) -> "InvocationMatrixReceipt":
    matrix_hash = _stable_hash(_matrix_payload(prompt_hash, invocation_specs, records))
    provisional = InvocationMatrixReceipt(
        prompt_hash=prompt_hash,
        matrix_hash=matrix_hash,
        receipt_hash="",
        validation_passed=validation_passed,
    )
    return InvocationMatrixReceipt(
        prompt_hash=provisional.prompt_hash,
        matrix_hash=provisional.matrix_hash,
        receipt_hash=provisional.stable_hash(),
        validation_passed=provisional.validation_passed,
    )


def _extract_spec_mapping(raw_spec: "ModelInvocationSpec | Mapping[str, Any]") -> Dict[str, Any]:
    if isinstance(raw_spec, ModelInvocationSpec):
        return raw_spec.to_dict()
    if isinstance(raw_spec, Mapping):
        return dict(raw_spec)
    raise InvocationMatrixValidationError("invocation spec must be ModelInvocationSpec or mapping")


def _normalize_invocation_spec(raw_spec: "ModelInvocationSpec | Mapping[str, Any]", *, fallback_prompt_hash: str) -> "ModelInvocationSpec":
    spec_map = _extract_spec_mapping(raw_spec)
    execution_mode = _normalize_required_text(spec_map.get("execution_mode"), field="spec.execution_mode").lower()
    if execution_mode not in SUPPORTED_EXECUTION_MODES:
        raise InvocationMatrixValidationError("spec.execution_mode must be one of: planned, simulated, observed")

    prompt_hash = spec_map.get("prompt_hash")
    if not isinstance(prompt_hash, str) or not prompt_hash.strip():
        prompt_hash = fallback_prompt_hash

    return ModelInvocationSpec(
        invocation_id=_normalize_required_text(spec_map.get("invocation_id"), field="spec.invocation_id"),
        model_name=_normalize_required_text(spec_map.get("model_name"), field="spec.model_name"),
        provider_name=_normalize_required_text(spec_map.get("provider_name"), field="spec.provider_name").lower(),
        route_name=_normalize_required_text(spec_map.get("route_name"), field="spec.route_name"),
        prompt_hash=_normalize_required_text(prompt_hash, field="spec.prompt_hash"),
        repetition_index=_normalize_non_negative_int(spec_map.get("repetition_index", 0), field="spec.repetition_index"),
        execution_mode=execution_mode,
        metadata=_normalize_metadata_mapping(spec_map.get("metadata", {}), field="spec.metadata"),
    )


def _normalize_record(raw_record: "InvocationRecord | Mapping[str, Any]") -> "InvocationRecord":
    if isinstance(raw_record, InvocationRecord):
        record_map = raw_record.to_dict()
    elif isinstance(raw_record, Mapping):
        record_map = dict(raw_record)
    else:
        raise InvocationMatrixValidationError("record must be InvocationRecord or mapping")

    status = _normalize_required_text(record_map.get("status"), field="record.status").lower()
    if status not in SUPPORTED_INVOCATION_STATUSES:
        raise InvocationMatrixValidationError("record.status must be one of: pending, completed, failed, invalid")

    response_hash = record_map.get("response_hash")
    if response_hash is not None:
        response_hash = _normalize_required_text(response_hash, field="record.response_hash")

    return InvocationRecord(
        invocation_id=_normalize_required_text(record_map.get("invocation_id"), field="record.invocation_id"),
        model_name=_normalize_required_text(record_map.get("model_name"), field="record.model_name"),
        provider_name=_normalize_required_text(record_map.get("provider_name"), field="record.provider_name").lower(),
        route_name=_normalize_required_text(record_map.get("route_name"), field="record.route_name"),
        prompt_hash=_normalize_required_text(record_map.get("prompt_hash"), field="record.prompt_hash"),
        repetition_index=_normalize_non_negative_int(record_map.get("repetition_index", 0), field="record.repetition_index"),
        execution_mode=_normalize_required_text(record_map.get("execution_mode"), field="record.execution_mode").lower(),
        response_hash=response_hash,
        status=status,
        metadata=_normalize_metadata_mapping(record_map.get("metadata", {}), field="record.metadata"),
    )


@dataclass(frozen=True)
class ModelInvocationSpec:
    invocation_id: str
    model_name: str
    provider_name: str
    route_name: str
    prompt_hash: str
    repetition_index: int
    execution_mode: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "model_name": self.model_name,
            "provider_name": self.provider_name,
            "route_name": self.route_name,
            "prompt_hash": self.prompt_hash,
            "repetition_index": int(self.repetition_index),
            "execution_mode": self.execution_mode,
            "metadata": _canonicalize_value(dict(self.metadata), field="spec.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class InvocationRecord:
    invocation_id: str
    model_name: str
    provider_name: str
    route_name: str
    prompt_hash: str
    repetition_index: int
    execution_mode: str
    response_hash: str | None
    status: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invocation_id": self.invocation_id,
            "model_name": self.model_name,
            "provider_name": self.provider_name,
            "route_name": self.route_name,
            "prompt_hash": self.prompt_hash,
            "repetition_index": int(self.repetition_index),
            "execution_mode": self.execution_mode,
            "response_hash": self.response_hash,
            "status": self.status,
            "metadata": _canonicalize_value(dict(self.metadata), field="record.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class InvocationMatrixReceipt:
    prompt_hash: str
    matrix_hash: str
    receipt_hash: str
    validation_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "receipt_hash": self.receipt_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "matrix_hash": self.matrix_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class InvocationMatrixValidationReport:
    valid: bool
    errors: Tuple[str, ...]
    error_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": bool(self.valid),
            "errors": list(self.errors),
            "error_count": int(self.error_count),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class MultiModelInvocationMatrix:
    prompt_hash: str
    invocation_specs: Tuple[ModelInvocationSpec, ...]
    records: Tuple[InvocationRecord, ...]
    receipt: InvocationMatrixReceipt
    validation: InvocationMatrixValidationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "invocation_specs": [spec.to_dict() for spec in self.invocation_specs],
            "records": [record.to_dict() for record in self.records],
            "receipt": self.receipt.to_dict(),
            "validation": self.validation.to_dict(),
            "matrix_version": MULTI_MODEL_INVOCATION_MATRIX_VERSION,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def simulate_invocation_records(
    invocation_specs: Sequence[ModelInvocationSpec | Mapping[str, Any]],
    *,
    response_hashes: Mapping[str, str | None] | None = None,
    status: str = "completed",
) -> Tuple[InvocationRecord, ...]:
    normalized_status = _normalize_required_text(status, field="status").lower()
    if normalized_status not in SUPPORTED_INVOCATION_STATUSES:
        raise InvocationMatrixValidationError("record.status must be one of: pending, completed, failed, invalid")

    response_lookup: Mapping[str, str | None] = response_hashes or {}
    normalized_specs = tuple(_normalize_invocation_spec(spec, fallback_prompt_hash="") for spec in invocation_specs)

    records = []
    for spec in normalized_specs:
        supplied_hash = response_lookup.get(spec.invocation_id)
        normalized_response_hash = None
        if supplied_hash is not None:
            normalized_response_hash = _normalize_required_text(supplied_hash, field="response_hashes.<invocation_id>")

        records.append(
            InvocationRecord(
                invocation_id=spec.invocation_id,
                model_name=spec.model_name,
                provider_name=spec.provider_name,
                route_name=spec.route_name,
                prompt_hash=spec.prompt_hash,
                repetition_index=spec.repetition_index,
                execution_mode="simulated",
                response_hash=normalized_response_hash,
                status=normalized_status,
                metadata=_canonicalize_value(dict(spec.metadata), field="record.metadata"),
            )
        )

    return _ordered_records(records)


def validate_invocation_matrix(
    matrix: MultiModelInvocationMatrix | Mapping[str, Any],
) -> InvocationMatrixValidationReport:
    errors: list[str] = []

    if isinstance(matrix, MultiModelInvocationMatrix):
        prompt_hash = matrix.prompt_hash
        specs = tuple(matrix.invocation_specs)
        records = tuple(matrix.records)
        receipt = matrix.receipt
    elif isinstance(matrix, Mapping):
        prompt_hash = str(matrix.get("prompt_hash", ""))
        specs_list: list[ModelInvocationSpec] = []
        for spec in matrix.get("invocation_specs", ()):
            try:
                specs_list.append(_normalize_invocation_spec(spec, fallback_prompt_hash=prompt_hash))
            except InvocationMatrixValidationError as exc:
                errors.append(str(exc))
        specs = tuple(specs_list)

        record_list: list[InvocationRecord] = []
        for record in matrix.get("records", ()):
            try:
                record_list.append(_normalize_record(record))
            except InvocationMatrixValidationError as exc:
                errors.append(str(exc))
        records = tuple(record_list)
        receipt_raw = matrix.get("receipt", {})
        if isinstance(receipt_raw, Mapping):
            receipt = InvocationMatrixReceipt(
                prompt_hash=str(receipt_raw.get("prompt_hash", "")),
                matrix_hash=str(receipt_raw.get("matrix_hash", "")),
                receipt_hash=str(receipt_raw.get("receipt_hash", "")),
                validation_passed=bool(receipt_raw.get("validation_passed", False)),
            )
        else:
            receipt = InvocationMatrixReceipt(prompt_hash="", matrix_hash="", receipt_hash="", validation_passed=False)
    else:
        return InvocationMatrixValidationReport(valid=False, errors=("matrix must be MultiModelInvocationMatrix or mapping",), error_count=1)

    if not isinstance(prompt_hash, str) or not prompt_hash.strip():
        errors.append("matrix.prompt_hash must be non-empty")

    invocation_ids = [spec.invocation_id for spec in specs]
    if len(set(invocation_ids)) != len(invocation_ids):
        errors.append("invocation_id must be unique")
    record_invocation_ids = [record.invocation_id for record in records]
    if len(set(record_invocation_ids)) != len(record_invocation_ids):
        errors.append("record.invocation_id must be unique")

    for spec in specs:
        if spec.prompt_hash != prompt_hash:
            errors.append("spec.prompt_hash mismatch")
        if spec.repetition_index < 0:
            errors.append("spec.repetition_index must be >= 0")
        if spec.execution_mode not in SUPPORTED_EXECUTION_MODES:
            errors.append("spec.execution_mode must be one of: planned, simulated, observed")

    for record in records:
        if record.prompt_hash != prompt_hash:
            errors.append("record.prompt_hash mismatch")
        if record.repetition_index < 0:
            errors.append("record.repetition_index must be >= 0")
        if record.execution_mode not in SUPPORTED_EXECUTION_MODES:
            errors.append("record.execution_mode must be one of: planned, simulated, observed")
        if record.status not in SUPPORTED_INVOCATION_STATUSES:
            errors.append("record.status must be one of: pending, completed, failed, invalid")

    if len(records) != len(specs):
        errors.append("record count must equal spec count")
    if set(record_invocation_ids) != set(invocation_ids):
        errors.append("record.invocation_id set must match spec.invocation_id set")

    expected_order = _ordered_records(records)
    if expected_order != tuple(records):
        errors.append("records not in deterministic order")

    expected_receipt = _build_receipt(prompt_hash, specs, records, validation_passed=receipt.validation_passed)
    if receipt.prompt_hash != prompt_hash:
        errors.append("receipt.prompt_hash mismatch")
    if receipt.matrix_hash != expected_receipt.matrix_hash:
        errors.append("receipt.matrix_hash mismatch")
    if receipt.receipt_hash != expected_receipt.receipt_hash:
        errors.append("receipt.receipt_hash mismatch")

    deduped_errors = tuple(dict.fromkeys(errors))
    return InvocationMatrixValidationReport(
        valid=not deduped_errors,
        errors=deduped_errors,
        error_count=len(deduped_errors),
    )


def build_multi_model_invocation_matrix(
    canonical_prompt_artifact: Any,
    invocation_specs: Sequence[ModelInvocationSpec | Mapping[str, Any]],
) -> MultiModelInvocationMatrix:
    prompt_hash = _extract_prompt_hash(canonical_prompt_artifact)

    validation_errors: list[str] = []
    normalized_specs: list[ModelInvocationSpec] = []
    for raw_spec in invocation_specs:
        try:
            normalized_specs.append(_normalize_invocation_spec(raw_spec, fallback_prompt_hash=prompt_hash))
        except InvocationMatrixValidationError as exc:
            validation_errors.append(str(exc))

    ordered_specs = tuple(
        sorted(
            normalized_specs,
            key=lambda s: (s.provider_name, s.model_name, int(s.repetition_index), s.invocation_id),
        )
    )

    records = _ordered_records(
        [
            InvocationRecord(
                invocation_id=spec.invocation_id,
                model_name=spec.model_name,
                provider_name=spec.provider_name,
                route_name=spec.route_name,
                prompt_hash=spec.prompt_hash,
                repetition_index=spec.repetition_index,
                execution_mode=spec.execution_mode,
                response_hash=None,
                status="pending",
                metadata=_canonicalize_value(dict(spec.metadata), field="record.metadata"),
            )
            for spec in ordered_specs
        ]
    )

    provisional_report = InvocationMatrixValidationReport(
        valid=not validation_errors,
        errors=tuple(dict.fromkeys(validation_errors)),
        error_count=len(tuple(dict.fromkeys(validation_errors))),
    )

    receipt = _build_receipt(prompt_hash, ordered_specs, records, validation_passed=provisional_report.valid)
    matrix = MultiModelInvocationMatrix(
        prompt_hash=prompt_hash,
        invocation_specs=ordered_specs,
        records=records,
        receipt=receipt,
        validation=provisional_report,
    )

    final_report_base = validate_invocation_matrix(matrix)
    merged_errors = tuple(dict.fromkeys((*provisional_report.errors, *final_report_base.errors)))
    final_report = InvocationMatrixValidationReport(
        valid=not merged_errors,
        errors=merged_errors,
        error_count=len(merged_errors),
    )
    final_receipt = _build_receipt(prompt_hash, ordered_specs, records, validation_passed=final_report.valid)
    return MultiModelInvocationMatrix(
        prompt_hash=prompt_hash,
        invocation_specs=ordered_specs,
        records=records,
        receipt=final_receipt,
        validation=final_report,
    )


def invocation_matrix_projection(
    matrix_or_parts: MultiModelInvocationMatrix | Mapping[str, Any],
) -> Dict[str, Any]:
    if isinstance(matrix_or_parts, MultiModelInvocationMatrix):
        matrix = matrix_or_parts
    elif isinstance(matrix_or_parts, Mapping):
        prompt_hash = str(matrix_or_parts.get("prompt_hash", "")).strip()
        specs = tuple(
            _normalize_invocation_spec(spec, fallback_prompt_hash=prompt_hash)
            for spec in matrix_or_parts.get("invocation_specs", ())
        )
        records = tuple(_normalize_record(record) for record in matrix_or_parts.get("records", ()))
        receipt = _build_receipt(prompt_hash, specs, records, validation_passed=True)
        matrix = MultiModelInvocationMatrix(
            prompt_hash=prompt_hash,
            invocation_specs=specs,
            records=records,
            receipt=receipt,
            validation=InvocationMatrixValidationReport(valid=True, errors=(), error_count=0),
        )
    else:
        raise InvocationMatrixValidationError("matrix_or_parts must be MultiModelInvocationMatrix or mapping")

    ordered_models = tuple(dict.fromkeys(record.model_name for record in matrix.records))
    ordered_providers = tuple(dict.fromkeys(record.provider_name for record in matrix.records))
    ordered_routes = tuple(dict.fromkeys(record.route_name for record in matrix.records))
    repetition_count = len({record.repetition_index for record in matrix.records})

    return {
        "prompt_hash": matrix.prompt_hash,
        "ordered_models": list(ordered_models),
        "ordered_providers": list(ordered_providers),
        "ordered_routes": list(ordered_routes),
        "repetition_count": int(repetition_count),
        "matrix_hash": matrix.receipt.matrix_hash,
    }
