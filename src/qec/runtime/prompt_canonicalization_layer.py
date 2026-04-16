# SPDX-License-Identifier: MIT
"""v138.2.10 — deterministic prompt canonicalization layer.

This module defines deterministic prompt artifacts and replay-safe receipts for
frontier comparative evaluation infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Tuple

PROMPT_CANONICALIZATION_LAYER_VERSION = "v138.2.10"

_SHA256_HEX_CHARS: frozenset = frozenset("0123456789abcdef")


class PromptCanonicalizationValidationError(ValueError):
    """Raised when prompt canonicalization input violates deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise PromptCanonicalizationValidationError(f"{field} contains non-canonical numeric value")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda x: str(x)):
            key = str(raw_key)
            if key in normalized:
                raise PromptCanonicalizationValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise PromptCanonicalizationValidationError(f"{field} contains unsupported type: {type(value).__name__}")


def _normalize_required_text(value: Any, *, field: str) -> str:
    if value is None:
        raise PromptCanonicalizationValidationError(f"{field} must not be None")
    if not isinstance(value, str):
        raise PromptCanonicalizationValidationError(f"{field} must be a string")
    return value.strip()


def _normalize_optional_text(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PromptCanonicalizationValidationError(f"{field} must be a string when provided")
    return value.strip()


def _normalize_repetition_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PromptCanonicalizationValidationError("spec.repetition_count must be an integer")
    return int(value)


def _normalize_policy_flags(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raw_flags = [value]
    elif isinstance(value, (list, tuple)):
        raw_flags = list(value)
    else:
        raise PromptCanonicalizationValidationError("spec.policy_flags must be a sequence of strings")

    normalized = []
    for raw_flag in raw_flags:
        if raw_flag is None or not isinstance(raw_flag, str):
            raise PromptCanonicalizationValidationError("spec.policy_flags entries must be strings")
        normalized.append(raw_flag.strip().lower())

    sorted_flags = tuple(sorted(normalized))
    if len(set(sorted_flags)) != len(sorted_flags):
        raise PromptCanonicalizationValidationError("spec.policy_flags contains duplicate values after normalization")
    return sorted_flags


def _normalize_metadata_mapping(value: Any, *, field: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PromptCanonicalizationValidationError(f"{field} must be a mapping")
    return _canonicalize_value(dict(value), field=field)


def _extract_mapping(raw_spec: PromptSpec | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(raw_spec, PromptSpec):
        return raw_spec.to_dict()
    if isinstance(raw_spec, Mapping):
        return dict(raw_spec)
    raise PromptCanonicalizationValidationError("spec must be PromptSpec or mapping")


def _collect_validation_errors_from_mapping(spec_map: Mapping[str, Any]) -> Tuple[str, ...]:
    errors: list[str] = []

    required_text_fields = (
        ("prompt_id", "spec.prompt_id must be non-empty"),
        ("prompt_text", "spec.prompt_text must be non-empty"),
        ("model_name", "spec.model_name must be non-empty"),
        ("invocation_route", "spec.invocation_route must be non-empty"),
    )
    for field_name, error_text in required_text_fields:
        value = spec_map.get(field_name)
        if value is None or not isinstance(value, str) or not value.strip():
            errors.append(error_text)

    system_prompt = spec_map.get("system_prompt")
    if system_prompt is not None:
        if not isinstance(system_prompt, str):
            errors.append("spec.system_prompt must be a string when provided")
        elif not system_prompt.strip():
            errors.append("spec.system_prompt must be non-empty when provided")

    repetition_count = spec_map.get("repetition_count")
    if isinstance(repetition_count, bool) or not isinstance(repetition_count, int):
        errors.append("spec.repetition_count must be an integer")
    elif int(repetition_count) <= 0:
        errors.append("spec.repetition_count must be > 0")

    policy_flags_raw = spec_map.get("policy_flags", ())
    normalized_flags: list[str] = []
    if policy_flags_raw is None:
        policy_flags_raw = ()
    if isinstance(policy_flags_raw, str):
        policy_iter = (policy_flags_raw,)
    elif isinstance(policy_flags_raw, (list, tuple)):
        policy_iter = tuple(policy_flags_raw)
    else:
        policy_iter = ()
        errors.append("spec.policy_flags must be a sequence of strings")

    for flag in policy_iter:
        if flag is None or not isinstance(flag, str):
            errors.append("spec.policy_flags entries must be strings")
            continue
        normalized_flags.append(flag.strip().lower())

    if len(set(normalized_flags)) != len(normalized_flags):
        errors.append("spec.policy_flags must be unique after normalization")

    for field_name in ("wrapper_metadata", "metadata"):
        field_value = spec_map.get(field_name, {})
        if field_value is None:
            field_value = {}
        if not isinstance(field_value, Mapping):
            errors.append(f"spec.{field_name} must be a mapping")
            continue
        try:
            _canonicalize_value(dict(field_value), field=f"spec.{field_name}")
        except PromptCanonicalizationValidationError as exc:
            errors.append(str(exc))

    return tuple(errors)


def _normalize_prompt_spec_permissive(spec_map: Mapping[str, Any]) -> PromptSpec:
    normalized_flags: list[str] = []
    policy_raw = spec_map.get("policy_flags", ())
    if isinstance(policy_raw, str):
        policy_values = (policy_raw,)
    elif isinstance(policy_raw, (list, tuple)):
        policy_values = tuple(policy_raw)
    else:
        policy_values = ()
    for value in policy_values:
        if isinstance(value, str):
            normalized_flags.append(value.strip().lower())
    deduped_flags = tuple(sorted(set(normalized_flags)))

    wrapper_metadata = spec_map.get("wrapper_metadata", {})
    metadata = spec_map.get("metadata", {})
    wrapper_map = wrapper_metadata if isinstance(wrapper_metadata, Mapping) else {}
    metadata_map = metadata if isinstance(metadata, Mapping) else {}

    repetition_raw = spec_map.get("repetition_count", 0)
    repetition_count = int(repetition_raw) if isinstance(repetition_raw, int) and not isinstance(repetition_raw, bool) else 0

    def _safe_text(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    system_prompt = spec_map.get("system_prompt")
    normalized_system_prompt = system_prompt.strip() if isinstance(system_prompt, str) else None

    return PromptSpec(
        prompt_id=_safe_text(spec_map.get("prompt_id")),
        prompt_text=_safe_text(spec_map.get("prompt_text")),
        system_prompt=normalized_system_prompt,
        wrapper_metadata=_canonicalize_value(dict(wrapper_map), field="spec.wrapper_metadata"),
        model_name=_safe_text(spec_map.get("model_name")),
        invocation_route=_safe_text(spec_map.get("invocation_route")),
        repetition_count=repetition_count,
        temperature_setting=_safe_text(spec_map.get("temperature_setting")),
        policy_flags=deduped_flags,
        metadata=_canonicalize_value(dict(metadata_map), field="spec.metadata"),
    )


@dataclass(frozen=True)
class PromptSpec:
    prompt_id: str
    prompt_text: str
    system_prompt: str | None
    wrapper_metadata: Dict[str, Any]
    model_name: str
    invocation_route: str
    repetition_count: int
    temperature_setting: str
    policy_flags: Tuple[str, ...]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "system_prompt": self.system_prompt,
            "wrapper_metadata": _canonicalize_value(dict(self.wrapper_metadata), field="spec.wrapper_metadata"),
            "model_name": self.model_name,
            "invocation_route": self.invocation_route,
            "repetition_count": int(self.repetition_count),
            "temperature_setting": self.temperature_setting,
            "policy_flags": list(self.policy_flags),
            "metadata": _canonicalize_value(dict(self.metadata), field="spec.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class CanonicalPromptReceipt:
    prompt_hash: str
    spec_hash: str
    wrapper_hash: str
    receipt_hash: str
    validation_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "spec_hash": self.spec_hash,
            "wrapper_hash": self.wrapper_hash,
            "receipt_hash": self.receipt_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "spec_hash": self.spec_hash,
            "wrapper_hash": self.wrapper_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class PromptCanonicalizationValidationReport:
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
class CanonicalPromptArtifact:
    spec: PromptSpec
    receipt: CanonicalPromptReceipt
    validation: PromptCanonicalizationValidationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "receipt": self.receipt.to_dict(),
            "validation": self.validation.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def normalize_prompt_spec(raw_spec: PromptSpec | Mapping[str, Any]) -> PromptSpec:
    spec_map = _extract_mapping(raw_spec)
    return PromptSpec(
        prompt_id=_normalize_required_text(spec_map.get("prompt_id"), field="spec.prompt_id"),
        prompt_text=_normalize_required_text(spec_map.get("prompt_text"), field="spec.prompt_text"),
        system_prompt=_normalize_optional_text(spec_map.get("system_prompt"), field="spec.system_prompt"),
        wrapper_metadata=_normalize_metadata_mapping(spec_map.get("wrapper_metadata", {}), field="spec.wrapper_metadata"),
        model_name=_normalize_required_text(spec_map.get("model_name"), field="spec.model_name"),
        invocation_route=_normalize_required_text(spec_map.get("invocation_route"), field="spec.invocation_route"),
        repetition_count=_normalize_repetition_count(spec_map.get("repetition_count", 0)),
        temperature_setting=_normalize_required_text(spec_map.get("temperature_setting"), field="spec.temperature_setting"),
        policy_flags=_normalize_policy_flags(spec_map.get("policy_flags", ())),
        metadata=_normalize_metadata_mapping(spec_map.get("metadata", {}), field="spec.metadata"),
    )


def _build_prompt_hash_payload(spec: PromptSpec) -> Dict[str, Any]:
    return {
        "prompt_text": spec.prompt_text,
        "system_prompt": spec.system_prompt,
        "model_name": spec.model_name,
        "invocation_route": spec.invocation_route,
        "repetition_count": int(spec.repetition_count),
        "temperature_setting": spec.temperature_setting,
        "policy_flags": list(spec.policy_flags),
    }


def _normalize_sha256_hex(value: Any, *, field: str) -> str:
    if value is None or not isinstance(value, str):
        raise PromptCanonicalizationValidationError(f"{field} must be a string")
    text = value.strip().lower()
    if len(text) != 64:
        raise PromptCanonicalizationValidationError(f"{field} must be a 64-character SHA-256 hex string")
    if not frozenset(text) <= _SHA256_HEX_CHARS:
        raise PromptCanonicalizationValidationError(f"{field} must be lowercase SHA-256 hex")
    return text


def validate_prompt_spec(
    raw_spec: PromptSpec | Mapping[str, Any],
    artifact: CanonicalPromptArtifact | Mapping[str, Any] | None = None,
) -> PromptCanonicalizationValidationReport:
    spec_map = _extract_mapping(raw_spec)
    errors = list(_collect_validation_errors_from_mapping(spec_map))

    if artifact is not None:
        if isinstance(artifact, CanonicalPromptArtifact):
            artifact_spec = artifact.spec
            receipt_map: Mapping[str, Any] = artifact.receipt.to_dict()
        elif isinstance(artifact, Mapping):
            artifact_spec = build_canonical_prompt_artifact(artifact.get("spec", {})).spec
            receipt_raw = artifact.get("receipt", {})
            receipt_map = receipt_raw if isinstance(receipt_raw, Mapping) else {}
        else:
            errors.append("artifact must be CanonicalPromptArtifact or mapping")
            artifact_spec = build_canonical_prompt_artifact(spec_map).spec
            receipt_map = {}

        expected_prompt_hash = _stable_hash(_build_prompt_hash_payload(artifact_spec))
        expected_spec_hash = artifact_spec.stable_hash()
        expected_wrapper_hash = _stable_hash(artifact_spec.to_dict()["wrapper_metadata"])
        provided_validation_passed = bool(receipt_map.get("validation_passed", False))
        expected_receipt_hash = _stable_hash(
            {
                "prompt_hash": expected_prompt_hash,
                "spec_hash": expected_spec_hash,
                "wrapper_hash": expected_wrapper_hash,
                "validation_passed": provided_validation_passed,
            }
        )

        if receipt_map.get("prompt_hash") != expected_prompt_hash:
            errors.append("receipt.prompt_hash mismatch")
        if receipt_map.get("spec_hash") != expected_spec_hash:
            errors.append("receipt.spec_hash mismatch")
        if receipt_map.get("wrapper_hash") != expected_wrapper_hash:
            errors.append("receipt.wrapper_hash mismatch")
        if receipt_map.get("receipt_hash") != expected_receipt_hash:
            errors.append("receipt.receipt_hash mismatch")

        expected_validation_passed = len(_collect_validation_errors_from_mapping(artifact_spec.to_dict())) == 0
        if provided_validation_passed != expected_validation_passed:
            errors.append("receipt.validation_passed mismatch")

    ordered_unique_errors = tuple(dict.fromkeys(errors))
    return PromptCanonicalizationValidationReport(
        valid=not ordered_unique_errors,
        errors=ordered_unique_errors,
        error_count=len(ordered_unique_errors),
    )


def build_canonical_prompt_artifact(raw_spec: PromptSpec | Mapping[str, Any]) -> CanonicalPromptArtifact:
    spec_map = _extract_mapping(raw_spec)
    validation_errors = list(_collect_validation_errors_from_mapping(spec_map))

    try:
        spec = normalize_prompt_spec(spec_map)
    except PromptCanonicalizationValidationError as exc:
        validation_errors.append(str(exc))
        spec = _normalize_prompt_spec_permissive(spec_map)

    report = PromptCanonicalizationValidationReport(
        valid=not validation_errors,
        errors=tuple(dict.fromkeys(validation_errors)),
        error_count=len(tuple(dict.fromkeys(validation_errors))),
    )

    spec_hash = spec.stable_hash()
    wrapper_hash = _stable_hash(spec.to_dict()["wrapper_metadata"])
    prompt_hash = _stable_hash(_build_prompt_hash_payload(spec))

    provisional_receipt = CanonicalPromptReceipt(
        prompt_hash=prompt_hash,
        spec_hash=spec_hash,
        wrapper_hash=wrapper_hash,
        receipt_hash="",
        validation_passed=report.valid,
    )
    receipt = CanonicalPromptReceipt(
        prompt_hash=provisional_receipt.prompt_hash,
        spec_hash=provisional_receipt.spec_hash,
        wrapper_hash=provisional_receipt.wrapper_hash,
        receipt_hash=provisional_receipt.stable_hash(),
        validation_passed=provisional_receipt.validation_passed,
    )

    return CanonicalPromptArtifact(spec=spec, receipt=receipt, validation=report)


def canonical_prompt_projection(
    artifact_or_spec: CanonicalPromptArtifact | PromptSpec | Mapping[str, Any],
) -> Dict[str, Any]:
    if isinstance(artifact_or_spec, CanonicalPromptArtifact):
        artifact = artifact_or_spec
    else:
        artifact = build_canonical_prompt_artifact(artifact_or_spec)

    return {
        "prompt_hash": artifact.receipt.prompt_hash,
        "model_name": artifact.spec.model_name,
        "invocation_route": artifact.spec.invocation_route,
        "repetition_count": int(artifact.spec.repetition_count),
        "policy_flags": list(artifact.spec.policy_flags),
        "wrapper_hash": artifact.receipt.wrapper_hash,
    }
