# SPDX-License-Identifier: MIT
"""v138.3.5 — deterministic proof-carrying runtime bridge."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Tuple

from qec.runtime.hardware_admissibility_proof_pack import (
    validate_hardware_admissibility_proof_pack,
)
from qec.runtime.constraint_bound_dispatch_firewall import (
    validate_constraint_bound_dispatch_firewall,
)

PROOF_CARRYING_RUNTIME_BRIDGE_VERSION = "v138.3.5"
_SUPPORTED_VERDICTS = ("allow", "deny", "recover_only")


class ProofCarryingRuntimeBridgeValidationError(ValueError):
    """Raised when proof-carrying runtime bridge inputs violate deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _normalize_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ProofCarryingRuntimeBridgeValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ProofCarryingRuntimeBridgeValidationError(f"{field} must be non-empty")
    return normalized


def _normalize_hash(value: Any, *, field: str) -> str:
    text = _normalize_text(value, field=field)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ProofCarryingRuntimeBridgeValidationError(f"{field} must be a lowercase 64-char sha256 hex string")
    return text


def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ProofCarryingRuntimeBridgeValidationError(f"{field} contains non-finite float")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda item: str(item)):
            key = str(raw_key)
            if key in normalized:
                raise ProofCarryingRuntimeBridgeValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise ProofCarryingRuntimeBridgeValidationError(f"{field} contains unsupported type: {type(value).__name__}")


def _extract_mapping(payload: Any, *, field: str) -> Dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        mapped = payload.to_dict()
        if isinstance(mapped, Mapping):
            return dict(mapped)
    raise ProofCarryingRuntimeBridgeValidationError(f"{field} must be mapping-compatible")


def _extract_firewall_verdict(firewall_map: Mapping[str, Any]) -> str:
    verdict = firewall_map.get("verdict")
    if isinstance(verdict, Mapping):
        return _normalize_text(verdict.get("verdict"), field="firewall.verdict.verdict")
    return _normalize_text(verdict, field="firewall.verdict")


def _authorization_hash_payload(*, state_id: str, proof_pack_hash: str, verdict: str, authorized: bool) -> Dict[str, Any]:
    return {
        "state_id": state_id,
        "proof_pack_hash": proof_pack_hash,
        "verdict": verdict,
        "authorized": bool(authorized),
    }


def _token_hash_payload(token: "RuntimeBridgeToken") -> Dict[str, Any]:
    return {
        "state_id": token.state_id,
        "verdict": token.verdict,
        "proof_pack_hash": token.proof_pack_hash,
        "authorization_hash": token.authorization_hash,
        "metadata": _canonicalize_value(dict(token.metadata), field="bridge_token.metadata"),
    }


def _bridge_hash_payload(bridge: "ProofCarryingRuntimeBridge") -> Dict[str, Any]:
    return {
        "proof_carrying_runtime_bridge_version": PROOF_CARRYING_RUNTIME_BRIDGE_VERSION,
        "state_id": bridge.state_id,
        "proof_pack_hash": bridge.proof_pack_hash,
        "verdict": bridge.verdict,
        "authorized": bool(bridge.authorized),
        "bridge_token": bridge.bridge_token.to_dict(),
    }


def _receipt_hash_payload(receipt: "RuntimeBridgeReceipt") -> Dict[str, Any]:
    return {
        "bridge_hash": receipt.bridge_hash,
        "validation_passed": bool(receipt.validation_passed),
    }


@dataclass(frozen=True)
class RuntimeBridgeToken:
    token_id: str
    state_id: str
    verdict: str
    proof_pack_hash: str
    authorization_hash: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token_id,
            "state_id": self.state_id,
            "verdict": self.verdict,
            "proof_pack_hash": self.proof_pack_hash,
            "authorization_hash": self.authorization_hash,
            "metadata": _canonicalize_value(dict(self.metadata), field="bridge_token.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimeBridgeReceipt:
    bridge_hash: str
    receipt_hash: str
    validation_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bridge_hash": self.bridge_hash,
            "receipt_hash": self.receipt_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(_receipt_hash_payload(self))


@dataclass(frozen=True)
class RuntimeBridgeValidationReport:
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
class ProofCarryingRuntimeBridge:
    state_id: str
    proof_pack_hash: str
    verdict: str
    authorized: bool
    bridge_token: RuntimeBridgeToken
    receipt: RuntimeBridgeReceipt
    validation: RuntimeBridgeValidationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proof_carrying_runtime_bridge_version": PROOF_CARRYING_RUNTIME_BRIDGE_VERSION,
            "state_id": self.state_id,
            "proof_pack_hash": self.proof_pack_hash,
            "verdict": self.verdict,
            "authorized": bool(self.authorized),
            "bridge_token": self.bridge_token.to_dict(),
            "receipt": self.receipt.to_dict(),
            "validation": self.validation.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def build_proof_carrying_runtime_bridge(
    hardware_proof_pack: Mapping[str, Any] | Any,
    firewall_artifact: Mapping[str, Any] | Any,
) -> ProofCarryingRuntimeBridge:
    proof_pack_map = _extract_mapping(hardware_proof_pack, field="hardware_proof_pack")
    firewall_map = _extract_mapping(firewall_artifact, field="firewall_artifact")

    pack_report = validate_hardware_admissibility_proof_pack(proof_pack_map)
    if not pack_report.valid:
        raise ProofCarryingRuntimeBridgeValidationError(
            f"hardware_proof_pack failed upstream validation: {pack_report.errors}"
        )
    fw_report = validate_constraint_bound_dispatch_firewall(firewall_map)
    if not fw_report.valid:
        raise ProofCarryingRuntimeBridgeValidationError(
            f"firewall_artifact failed upstream validation: {fw_report.errors}"
        )

    state_id = _normalize_text(proof_pack_map.get("state_id"), field="hardware_proof_pack.state_id")
    if _normalize_text(firewall_map.get("state_id"), field="firewall_artifact.state_id") != state_id:
        raise ProofCarryingRuntimeBridgeValidationError("lineage mismatch: firewall_artifact.state_id")

    proof_pack_receipt = _extract_mapping(proof_pack_map.get("receipt"), field="hardware_proof_pack.receipt")
    proof_pack_hash = _normalize_hash(
        proof_pack_receipt.get("proof_pack_hash"),
        field="hardware_proof_pack.receipt.proof_pack_hash",
    )

    proof_pack_verdict = _normalize_text(proof_pack_map.get("verdict"), field="hardware_proof_pack.verdict")
    firewall_verdict = _extract_firewall_verdict(firewall_map)
    if proof_pack_verdict != firewall_verdict:
        raise ProofCarryingRuntimeBridgeValidationError("lineage mismatch: verdict")
    if firewall_verdict not in _SUPPORTED_VERDICTS:
        raise ProofCarryingRuntimeBridgeValidationError(f"unsupported verdict: {firewall_verdict}")

    authorized = firewall_verdict == "allow"
    authorization_hash = _stable_hash(
        _authorization_hash_payload(
            state_id=state_id,
            proof_pack_hash=proof_pack_hash,
            verdict=firewall_verdict,
            authorized=authorized,
        )
    )

    firewall_receipt = firewall_map.get("receipt")
    if not isinstance(firewall_receipt, Mapping):
        raise ProofCarryingRuntimeBridgeValidationError("firewall_artifact.receipt must be a mapping")

    token_provisional = RuntimeBridgeToken(
        token_id="",
        state_id=state_id,
        verdict=firewall_verdict,
        proof_pack_hash=proof_pack_hash,
        authorization_hash=authorization_hash,
        metadata={
            "firewall_hash": _normalize_hash(
                firewall_receipt.get("firewall_hash"),
                field="firewall_artifact.receipt.firewall_hash",
            ),
            "proof_lineage_hash": _normalize_hash(
                proof_pack_map.get("proof_lineage_hash"),
                field="hardware_proof_pack.proof_lineage_hash",
            ),
        },
    )
    bridge_token = RuntimeBridgeToken(
        token_id=_stable_hash(_token_hash_payload(token_provisional)),
        state_id=token_provisional.state_id,
        verdict=token_provisional.verdict,
        proof_pack_hash=token_provisional.proof_pack_hash,
        authorization_hash=token_provisional.authorization_hash,
        metadata=token_provisional.metadata,
    )

    provisional_validation = RuntimeBridgeValidationReport(valid=True, errors=(), error_count=0)
    provisional_bridge = ProofCarryingRuntimeBridge(
        state_id=state_id,
        proof_pack_hash=proof_pack_hash,
        verdict=firewall_verdict,
        authorized=authorized,
        bridge_token=bridge_token,
        receipt=RuntimeBridgeReceipt(bridge_hash="", receipt_hash="", validation_passed=True),
        validation=provisional_validation,
    )
    bridge_hash = _stable_hash(_bridge_hash_payload(provisional_bridge))

    # Finalize receipt before running validation so the validator sees a proper 64-char receipt_hash.
    # _receipt_hash_payload uses only bridge_hash and validation_passed, not receipt_hash itself.
    receipt_base = RuntimeBridgeReceipt(bridge_hash=bridge_hash, receipt_hash="", validation_passed=True)
    receipt = RuntimeBridgeReceipt(
        bridge_hash=bridge_hash,
        receipt_hash=receipt_base.stable_hash(),
        validation_passed=True,
    )
    complete_bridge = ProofCarryingRuntimeBridge(
        state_id=state_id,
        proof_pack_hash=proof_pack_hash,
        verdict=firewall_verdict,
        authorized=authorized,
        bridge_token=bridge_token,
        receipt=receipt,
        validation=provisional_validation,
    )
    validation = validate_proof_carrying_runtime_bridge(complete_bridge)

    return ProofCarryingRuntimeBridge(
        state_id=state_id,
        proof_pack_hash=proof_pack_hash,
        verdict=firewall_verdict,
        authorized=authorized,
        bridge_token=bridge_token,
        receipt=receipt,
        validation=validation,
    )


def validate_proof_carrying_runtime_bridge(
    bridge: ProofCarryingRuntimeBridge | Mapping[str, Any],
) -> RuntimeBridgeValidationReport:
    errors: list[str] = []

    payload: Dict[str, Any]
    if isinstance(bridge, ProofCarryingRuntimeBridge):
        payload = bridge.to_dict()
    elif isinstance(bridge, Mapping):
        payload = dict(bridge)
    else:
        return RuntimeBridgeValidationReport(
            valid=False,
            errors=("bridge must be ProofCarryingRuntimeBridge or Mapping",),
            error_count=1,
        )

    version = payload.get("proof_carrying_runtime_bridge_version")
    if version != PROOF_CARRYING_RUNTIME_BRIDGE_VERSION:
        errors.append(f"proof_carrying_runtime_bridge_version must be {PROOF_CARRYING_RUNTIME_BRIDGE_VERSION!r}")

    try:
        state_id = _normalize_text(payload.get("state_id"), field="state_id")
    except ProofCarryingRuntimeBridgeValidationError as exc:
        errors.append(str(exc))
        state_id = ""

    try:
        proof_pack_hash = _normalize_hash(payload.get("proof_pack_hash"), field="proof_pack_hash")
    except ProofCarryingRuntimeBridgeValidationError as exc:
        errors.append(str(exc))
        proof_pack_hash = ""

    try:
        verdict = _normalize_text(payload.get("verdict"), field="verdict")
        if verdict not in _SUPPORTED_VERDICTS:
            errors.append(f"verdict unsupported: {verdict}")
    except ProofCarryingRuntimeBridgeValidationError as exc:
        errors.append(str(exc))
        verdict = ""

    authorized = bool(payload.get("authorized"))
    if verdict in _SUPPORTED_VERDICTS and authorized != (verdict == "allow"):
        errors.append("authorization consistency violation")

    token_map = payload.get("bridge_token")
    token: RuntimeBridgeToken | None = None
    if not isinstance(token_map, Mapping):
        errors.append("bridge_token must be a mapping")
    else:
        try:
            raw_metadata = token_map.get("metadata")
            if raw_metadata is None:
                raw_metadata = {}
            if not isinstance(raw_metadata, Mapping):
                raise ProofCarryingRuntimeBridgeValidationError("bridge_token.metadata must be a mapping")
            token = RuntimeBridgeToken(
                token_id=_normalize_hash(token_map.get("token_id"), field="bridge_token.token_id"),
                state_id=_normalize_text(token_map.get("state_id"), field="bridge_token.state_id"),
                verdict=_normalize_text(token_map.get("verdict"), field="bridge_token.verdict"),
                proof_pack_hash=_normalize_hash(token_map.get("proof_pack_hash"), field="bridge_token.proof_pack_hash"),
                authorization_hash=_normalize_hash(
                    token_map.get("authorization_hash"),
                    field="bridge_token.authorization_hash",
                ),
                metadata=dict(_canonicalize_value(raw_metadata, field="bridge_token.metadata")),
            )
            if token.token_id != _stable_hash(_token_hash_payload(token)):
                errors.append("bridge_token.token_id mismatch")
            if token.state_id != state_id:
                errors.append("bridge_token.state_id lineage mismatch")
            if token.proof_pack_hash != proof_pack_hash:
                errors.append("proof_pack_hash lineage mismatch")
            if token.verdict != verdict:
                errors.append("bridge_token.verdict mismatch")

            expected_auth_hash = _stable_hash(
                _authorization_hash_payload(
                    state_id=state_id,
                    proof_pack_hash=proof_pack_hash,
                    verdict=verdict,
                    authorized=authorized,
                )
            )
            if token.authorization_hash != expected_auth_hash:
                errors.append("authorization_hash mismatch")
        except (ProofCarryingRuntimeBridgeValidationError, TypeError, ValueError) as exc:
            errors.append(str(exc))

    validation_map = payload.get("validation")
    validation_obj: RuntimeBridgeValidationReport | None = None
    if not isinstance(validation_map, Mapping):
        errors.append("validation must be a mapping")
    else:
        raw_validation_errors = validation_map.get("errors", ())
        if isinstance(raw_validation_errors, (str, bytes)) or not isinstance(raw_validation_errors, (list, tuple)):
            errors.append("validation.errors must be a sequence")
            validation_errors: Tuple[str, ...] = ()
        else:
            validation_errors = tuple(str(item) for item in raw_validation_errors)
        try:
            validation_error_count = int(validation_map.get("error_count", 0))
        except (TypeError, ValueError):
            errors.append("validation.error_count must be an integer")
            validation_error_count = 0
        validation_obj = RuntimeBridgeValidationReport(
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
            receipt = RuntimeBridgeReceipt(
                bridge_hash=_normalize_hash(receipt_map.get("bridge_hash"), field="receipt.bridge_hash"),
                receipt_hash=_normalize_hash(receipt_map.get("receipt_hash"), field="receipt.receipt_hash"),
                validation_passed=bool(receipt_map.get("validation_passed")),
            )
            if token is not None:
                reconstructed = ProofCarryingRuntimeBridge(
                    state_id=state_id,
                    proof_pack_hash=proof_pack_hash,
                    verdict=verdict,
                    authorized=authorized,
                    bridge_token=token,
                    receipt=RuntimeBridgeReceipt(bridge_hash="", receipt_hash="", validation_passed=True),
                    validation=RuntimeBridgeValidationReport(valid=True, errors=(), error_count=0),
                )
                expected_bridge_hash = _stable_hash(_bridge_hash_payload(reconstructed))
                if receipt.bridge_hash != expected_bridge_hash:
                    errors.append("receipt.bridge_hash mismatch")
            if receipt.receipt_hash != receipt.stable_hash():
                errors.append("receipt.receipt_hash lineage mismatch")
            if validation_obj is not None and receipt.validation_passed != validation_obj.valid:
                errors.append("receipt.validation_passed mismatch")
        except ProofCarryingRuntimeBridgeValidationError as exc:
            errors.append(str(exc))

    valid = len(errors) == 0
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

    return RuntimeBridgeValidationReport(valid=valid, errors=tuple(errors), error_count=len(errors))


def runtime_bridge_projection(bridge: ProofCarryingRuntimeBridge | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(bridge, ProofCarryingRuntimeBridge):
        return {
            "authorized": bool(bridge.authorized),
            "verdict": bridge.verdict,
            "bridge_hash": bridge.receipt.bridge_hash,
            "receipt_hash": bridge.receipt.receipt_hash,
        }

    payload = _extract_mapping(bridge, field="bridge")
    verdict = _normalize_text(payload.get("verdict"), field="bridge.verdict")
    if verdict not in _SUPPORTED_VERDICTS:
        raise ProofCarryingRuntimeBridgeValidationError("bridge.verdict unsupported")
    receipt = payload.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ProofCarryingRuntimeBridgeValidationError("bridge.receipt must be a mapping")

    return {
        "authorized": bool(payload.get("authorized")),
        "verdict": verdict,
        "bridge_hash": _normalize_hash(receipt.get("bridge_hash"), field="bridge.receipt.bridge_hash"),
        "receipt_hash": _normalize_hash(receipt.get("receipt_hash"), field="bridge.receipt.receipt_hash"),
    }
