# SPDX-License-Identifier: MIT
"""v138.3.4 — deterministic hardware admissibility proof pack."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

HARDWARE_ADMISSIBILITY_PROOF_PACK_VERSION = "v138.3.4"
_SUPPORTED_COMPONENT_TYPES = ("admissibility", "tension", "recovery", "firewall")
_SUPPORTED_VERDICTS = ("allow", "deny", "recover_only")


class HardwareAdmissibilityProofPackValidationError(ValueError):
    """Raised when hardware proof pack inputs violate deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _normalize_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise HardwareAdmissibilityProofPackValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise HardwareAdmissibilityProofPackValidationError(f"{field} must be non-empty")
    return normalized


def _normalize_hash(value: Any, *, field: str) -> str:
    text = _normalize_text(value, field=field)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise HardwareAdmissibilityProofPackValidationError(f"{field} must be a lowercase 64-char sha256 hex string")
    return text


def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise HardwareAdmissibilityProofPackValidationError(f"{field} contains non-finite float")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda item: str(item)):
            key = str(raw_key)
            if key in normalized:
                raise HardwareAdmissibilityProofPackValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise HardwareAdmissibilityProofPackValidationError(f"{field} contains unsupported type: {type(value).__name__}")


def _extract_mapping(payload: Any, *, field: str) -> Dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        mapped = payload.to_dict()
        if isinstance(mapped, Mapping):
            return dict(mapped)
    raise HardwareAdmissibilityProofPackValidationError(f"{field} must be mapping-compatible")


def _component_sort_key(component: "HardwareProofComponent") -> Tuple[str, str]:
    return (component.component_type, component.component_id)


def _proof_pack_hash_payload(pack: "HardwareAdmissibilityProofPack") -> Dict[str, Any]:
    return {
        "state_id": pack.state_id,
        "components": [component.to_dict() for component in pack.components],
        "verdict": pack.verdict,
        "proof_lineage_hash": pack.proof_lineage_hash,
    }


def _build_proof_lineage_hash(*, state_id: str, components: Sequence["HardwareProofComponent"], verdict: str) -> str:
    return _stable_hash(
        {
            "state_id": state_id,
            "components": [
                {
                    "component_id": c.component_id,
                    "component_type": c.component_type,
                    "source_hash": c.source_hash,
                }
                for c in components
            ],
            "verdict": verdict,
        }
    )


@dataclass(frozen=True)
class HardwareProofComponent:
    component_id: str
    component_type: str
    source_hash: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "source_hash": self.source_hash,
            "metadata": _canonicalize_value(dict(self.metadata), field="component.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class HardwareValidationReceipt:
    proof_pack_hash: str
    validation_hash: str
    receipt_hash: str
    validation_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proof_pack_hash": self.proof_pack_hash,
            "validation_hash": self.validation_hash,
            "receipt_hash": self.receipt_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "proof_pack_hash": self.proof_pack_hash,
            "validation_hash": self.validation_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class HardwareProofValidationReport:
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
class HardwareAdmissibilityProofPack:
    state_id: str
    components: Tuple[HardwareProofComponent, ...]
    verdict: str
    proof_lineage_hash: str
    receipt: HardwareValidationReceipt
    validation: HardwareProofValidationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hardware_admissibility_proof_pack_version": HARDWARE_ADMISSIBILITY_PROOF_PACK_VERSION,
            "state_id": self.state_id,
            "components": [component.to_dict() for component in self.components],
            "verdict": self.verdict,
            "proof_lineage_hash": self.proof_lineage_hash,
            "receipt": self.receipt.to_dict(),
            "validation": self.validation.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def _normalize_component(raw: Any, *, field: str) -> HardwareProofComponent:
    if isinstance(raw, HardwareProofComponent):
        component_id = _normalize_text(raw.component_id, field=f"{field}.component_id")
        component_type = _normalize_text(raw.component_type, field=f"{field}.component_type")
        source_hash = _normalize_hash(raw.source_hash, field=f"{field}.source_hash")
        metadata = dict(_canonicalize_value(dict(raw.metadata), field=f"{field}.metadata"))
    elif isinstance(raw, Mapping):
        component_id = _normalize_text(raw.get("component_id"), field=f"{field}.component_id")
        component_type = _normalize_text(raw.get("component_type"), field=f"{field}.component_type")
        source_hash = _normalize_hash(raw.get("source_hash"), field=f"{field}.source_hash")
        raw_meta = raw.get("metadata")
        if raw_meta is None:
            raw_meta = {}
        if not isinstance(raw_meta, Mapping):
            raise HardwareAdmissibilityProofPackValidationError(
                f"{field}.metadata must be a mapping or null"
            )
        metadata = dict(_canonicalize_value(raw_meta, field=f"{field}.metadata"))
    else:
        raise HardwareAdmissibilityProofPackValidationError(f"{field} must be HardwareProofComponent or mapping")

    if component_type not in _SUPPORTED_COMPONENT_TYPES:
        raise HardwareAdmissibilityProofPackValidationError(
            f"{field}.component_type unsupported: {component_type}"
        )

    return HardwareProofComponent(
        component_id=component_id,
        component_type=component_type,
        source_hash=source_hash,
        metadata=metadata,
    )


def _make_component(*, component_id: str, component_type: str, source_hash: str, state_id: str) -> HardwareProofComponent:
    return HardwareProofComponent(
        component_id=component_id,
        component_type=component_type,
        source_hash=source_hash,
        metadata={"state_id": state_id, "source": component_type},
    )


def build_hardware_admissibility_proof_pack(
    projection: Mapping[str, Any] | Any,
    tension: Mapping[str, Any] | Any,
    recovery: Mapping[str, Any] | Any,
    firewall: Mapping[str, Any] | Any,
) -> HardwareAdmissibilityProofPack:
    projection_map = _extract_mapping(projection, field="projection")
    tension_map = _extract_mapping(tension, field="tension")
    recovery_map = _extract_mapping(recovery, field="recovery")
    firewall_map = _extract_mapping(firewall, field="firewall")

    state_id = _normalize_text(projection_map.get("state_id"), field="projection.state_id")
    if _normalize_text(tension_map.get("state_id"), field="tension.state_id") != state_id:
        raise HardwareAdmissibilityProofPackValidationError("lineage mismatch: tension.state_id")
    if _normalize_text(recovery_map.get("state_id"), field="recovery.state_id") != state_id:
        raise HardwareAdmissibilityProofPackValidationError("lineage mismatch: recovery.state_id")
    if _normalize_text(firewall_map.get("state_id"), field="firewall.state_id") != state_id:
        raise HardwareAdmissibilityProofPackValidationError("lineage mismatch: firewall.state_id")

    firewall_verdict = firewall_map.get("verdict")
    if not isinstance(firewall_verdict, Mapping):
        raise HardwareAdmissibilityProofPackValidationError("firewall.verdict must be a mapping")
    verdict = _normalize_text(firewall_verdict.get("verdict"), field="firewall.verdict.verdict")
    if verdict not in _SUPPORTED_VERDICTS:
        raise HardwareAdmissibilityProofPackValidationError(f"unsupported verdict: {verdict}")

    projection_receipt = _extract_mapping(projection_map.get("receipt"), field="projection.receipt")
    tension_receipt = _extract_mapping(tension_map.get("receipt"), field="tension.receipt")
    recovery_receipt = _extract_mapping(recovery_map.get("receipt"), field="recovery.receipt")
    firewall_receipt = _extract_mapping(firewall_map.get("receipt"), field="firewall.receipt")

    components = tuple(
        sorted(
            (
                _make_component(
                    component_id=f"{state_id}:admissibility",
                    component_type="admissibility",
                    source_hash=_normalize_hash(projection_receipt.get("proof_hash"), field="projection.receipt.proof_hash"),
                    state_id=state_id,
                ),
                _make_component(
                    component_id=f"{state_id}:tension",
                    component_type="tension",
                    source_hash=_normalize_hash(tension_receipt.get("tension_hash"), field="tension.receipt.tension_hash"),
                    state_id=state_id,
                ),
                _make_component(
                    component_id=f"{state_id}:recovery",
                    component_type="recovery",
                    source_hash=_normalize_hash(recovery_receipt.get("recovery_hash"), field="recovery.receipt.recovery_hash"),
                    state_id=state_id,
                ),
                _make_component(
                    component_id=f"{state_id}:firewall",
                    component_type="firewall",
                    source_hash=_normalize_hash(firewall_receipt.get("firewall_hash"), field="firewall.receipt.firewall_hash"),
                    state_id=state_id,
                ),
            ),
            key=_component_sort_key,
        )
    )

    proof_lineage_hash = _build_proof_lineage_hash(state_id=state_id, components=components, verdict=verdict)

    provisional_validation = HardwareProofValidationReport(valid=True, errors=(), error_count=0)
    provisional_pack = HardwareAdmissibilityProofPack(
        state_id=state_id,
        components=components,
        verdict=verdict,
        proof_lineage_hash=proof_lineage_hash,
        receipt=HardwareValidationReceipt(
            proof_pack_hash="",
            validation_hash=provisional_validation.stable_hash(),
            receipt_hash="",
            validation_passed=True,
        ),
        validation=provisional_validation,
    )
    proof_pack_hash = _stable_hash(_proof_pack_hash_payload(provisional_pack))

    provisional_receipt = HardwareValidationReceipt(
        proof_pack_hash=proof_pack_hash,
        validation_hash=provisional_validation.stable_hash(),
        receipt_hash="",
        validation_passed=True,
    )
    first_pass_pack = HardwareAdmissibilityProofPack(
        state_id=state_id,
        components=components,
        verdict=verdict,
        proof_lineage_hash=proof_lineage_hash,
        receipt=HardwareValidationReceipt(
            proof_pack_hash=proof_pack_hash,
            validation_hash=provisional_validation.stable_hash(),
            receipt_hash=provisional_receipt.stable_hash(),
            validation_passed=True,
        ),
        validation=provisional_validation,
    )

    validation = validate_hardware_admissibility_proof_pack(first_pass_pack)
    final_receipt_base = HardwareValidationReceipt(
        proof_pack_hash=proof_pack_hash,
        validation_hash=validation.stable_hash(),
        receipt_hash="",
        validation_passed=validation.valid,
    )
    final_receipt = HardwareValidationReceipt(
        proof_pack_hash=proof_pack_hash,
        validation_hash=validation.stable_hash(),
        receipt_hash=final_receipt_base.stable_hash(),
        validation_passed=validation.valid,
    )

    return HardwareAdmissibilityProofPack(
        state_id=state_id,
        components=components,
        verdict=verdict,
        proof_lineage_hash=proof_lineage_hash,
        receipt=final_receipt,
        validation=validation,
    )


def validate_hardware_admissibility_proof_pack(
    proof_pack: HardwareAdmissibilityProofPack | Mapping[str, Any],
) -> HardwareProofValidationReport:
    errors: list[str] = []

    if isinstance(proof_pack, HardwareAdmissibilityProofPack):
        payload = proof_pack.to_dict()
    elif isinstance(proof_pack, Mapping):
        payload = dict(proof_pack)
    else:
        return HardwareProofValidationReport(
            valid=False,
            errors=("proof_pack must be HardwareAdmissibilityProofPack or Mapping",),
            error_count=1,
        )

    pack_version = payload.get("hardware_admissibility_proof_pack_version")
    if pack_version != HARDWARE_ADMISSIBILITY_PROOF_PACK_VERSION:
        errors.append(
            f"hardware_admissibility_proof_pack_version must be {HARDWARE_ADMISSIBILITY_PROOF_PACK_VERSION!r}"
        )

    try:
        state_id = _normalize_text(payload.get("state_id"), field="state_id")
    except HardwareAdmissibilityProofPackValidationError as exc:
        errors.append(str(exc))
        state_id = ""

    components: list[HardwareProofComponent] = []
    raw_components = payload.get("components")
    if not isinstance(raw_components, Sequence) or isinstance(raw_components, (str, bytes)):
        errors.append("components must be a sequence")
    else:
        for index, raw_component in enumerate(raw_components):
            try:
                components.append(_normalize_component(raw_component, field=f"components[{index}]") )
            except HardwareAdmissibilityProofPackValidationError as exc:
                errors.append(str(exc))

    sorted_components = tuple(sorted(components, key=_component_sort_key))
    if [c.to_dict() for c in components] != [c.to_dict() for c in sorted_components]:
        errors.append("components must be sorted by (component_type, component_id)")

    component_ids = [c.component_id for c in components]
    if len(component_ids) != len(set(component_ids)):
        errors.append("duplicate component ids are not allowed")

    component_types = [c.component_type for c in components]
    for required_type in _SUPPORTED_COMPONENT_TYPES:
        count = component_types.count(required_type)
        if count == 0:
            errors.append(f"required component type missing: {required_type}")
        elif count > 1:
            errors.append(f"component type {required_type} present {count} times (expected exactly once)")

    for component in components:
        metadata_state = component.metadata.get("state_id")
        if metadata_state is not None and str(metadata_state).strip() != state_id:
            errors.append(f"component {component.component_id} lineage mismatch")

    try:
        verdict = _normalize_text(payload.get("verdict"), field="verdict")
        if verdict not in _SUPPORTED_VERDICTS:
            errors.append(f"verdict unsupported: {verdict}")
    except HardwareAdmissibilityProofPackValidationError as exc:
        errors.append(str(exc))
        verdict = ""

    try:
        proof_lineage_hash = _normalize_hash(payload.get("proof_lineage_hash"), field="proof_lineage_hash")
        expected_lineage_hash = _build_proof_lineage_hash(state_id=state_id, components=sorted_components, verdict=verdict)
        if proof_lineage_hash != expected_lineage_hash:
            errors.append("proof_lineage_hash mismatch")
    except HardwareAdmissibilityProofPackValidationError as exc:
        errors.append(str(exc))

    validation_map = payload.get("validation")
    validation_obj: HardwareProofValidationReport | None = None
    if not isinstance(validation_map, Mapping):
        errors.append("validation must be a mapping")
    else:
        raw_validation_errors = validation_map.get("errors", ())
        if isinstance(raw_validation_errors, (str, bytes)) or not isinstance(raw_validation_errors, Sequence):
            errors.append("validation.errors must be a sequence")
            validation_errors: Tuple[str, ...] = ()
        else:
            validation_errors = tuple(str(item) for item in raw_validation_errors)

        raw_error_count = validation_map.get("error_count", 0)
        try:
            validation_error_count = int(raw_error_count)
        except (TypeError, ValueError):
            errors.append("validation.error_count must be an integer")
            validation_error_count = 0

        validation_obj = HardwareProofValidationReport(
            valid=bool(validation_map.get("valid")),
            errors=validation_errors,
            error_count=validation_error_count,
        )
        if validation_obj.error_count != len(validation_obj.errors):
            errors.append("validation.error_count mismatch")

    receipt_map = payload.get("receipt")
    if not isinstance(receipt_map, Mapping):
        errors.append("receipt must be a mapping")
    else:
        try:
            receipt = HardwareValidationReceipt(
                proof_pack_hash=_normalize_hash(receipt_map.get("proof_pack_hash"), field="receipt.proof_pack_hash"),
                validation_hash=_normalize_hash(receipt_map.get("validation_hash"), field="receipt.validation_hash"),
                receipt_hash=_normalize_hash(receipt_map.get("receipt_hash"), field="receipt.receipt_hash"),
                validation_passed=bool(receipt_map.get("validation_passed")),
            )
            if receipt.receipt_hash != receipt.stable_hash():
                errors.append("receipt.receipt_hash lineage mismatch")

            try:
                expected_pack_hash = _stable_hash(
                    {
                        "state_id": state_id,
                        "components": [component.to_dict() for component in sorted_components],
                        "verdict": verdict,
                        "proof_lineage_hash": payload.get("proof_lineage_hash"),
                    }
                )
            except (TypeError, ValueError):
                errors.append("receipt.proof_pack_hash invalid canonical payload")
            else:
                if receipt.proof_pack_hash != expected_pack_hash:
                    errors.append("receipt.proof_pack_hash mismatch")

            if validation_obj is not None and receipt.validation_hash != validation_obj.stable_hash():
                errors.append("receipt.validation_hash mismatch")
        except HardwareAdmissibilityProofPackValidationError as exc:
            errors.append(str(exc))

    valid = len(errors) == 0
    if isinstance(receipt_map, Mapping) and bool(receipt_map.get("validation_passed")) != valid:
        errors.append("receipt.validation_passed mismatch")
        valid = False

    if isinstance(validation_map, Mapping) and bool(validation_map.get("valid")) != valid:
        errors.append("validation.valid mismatch")
        valid = False

    try:
        canonical = _canonical_json(payload)
        if _canonical_json(json.loads(canonical)) != canonical:
            errors.append("canonical JSON stability violation")
            valid = False
    except (TypeError, ValueError):
        errors.append("canonical JSON stability violation")
        valid = False

    return HardwareProofValidationReport(valid=valid, errors=tuple(errors), error_count=len(errors))


def hardware_proof_projection(proof_pack: HardwareAdmissibilityProofPack | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(proof_pack, HardwareAdmissibilityProofPack):
        return {
            "verdict": proof_pack.verdict,
            "proof_lineage_hash": proof_pack.proof_lineage_hash,
            "proof_pack_hash": proof_pack.receipt.proof_pack_hash,
            "receipt_hash": proof_pack.receipt.receipt_hash,
        }

    if not isinstance(proof_pack, Mapping):
        raise HardwareAdmissibilityProofPackValidationError(
            "proof_pack must be HardwareAdmissibilityProofPack or mapping"
        )

    verdict = _normalize_text(proof_pack.get("verdict"), field="proof_pack.verdict")
    if verdict not in _SUPPORTED_VERDICTS:
        raise HardwareAdmissibilityProofPackValidationError("proof_pack.verdict unsupported")

    receipt = proof_pack.get("receipt")
    if not isinstance(receipt, Mapping):
        raise HardwareAdmissibilityProofPackValidationError("proof_pack.receipt must be a mapping")

    return {
        "verdict": verdict,
        "proof_lineage_hash": _normalize_hash(proof_pack.get("proof_lineage_hash"), field="proof_pack.proof_lineage_hash"),
        "proof_pack_hash": _normalize_hash(receipt.get("proof_pack_hash"), field="proof_pack.receipt.proof_pack_hash"),
        "receipt_hash": _normalize_hash(receipt.get("receipt_hash"), field="proof_pack.receipt.receipt_hash"),
    }
