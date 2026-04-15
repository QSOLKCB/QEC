"""v137.21.0 — Proof-Bound API Layer.

Deterministic proof-binding contract for decoder-adjacent and orchestration-
adjacent API exports. This module is additive and does not alter decoder semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Iterable, Mapping, Tuple


ADVISORY_STATES: Tuple[str, ...] = (
    "proof_verified",
    "proof_partial",
    "proof_boundary_violation",
    "proof_contract_failure",
)

_METRIC_ORDER: Tuple[str, ...] = (
    "invariant_satisfaction_score",
    "replay_contract_score",
    "serialization_integrity_score",
    "interface_boundary_score",
    "proof_confidence_score",
)


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_hash(data: Any) -> str:
    return _sha256_hex(_canonical_json(data).encode("utf-8"))


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _safe_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for original_key in sorted(value.keys(), key=str):
            string_key = str(original_key)
            if string_key in normalized:
                raise ValueError(
                    f"Mapping contains non-unique canonical key: {string_key!r}"
                )
            normalized[string_key] = _canonicalize(value[original_key])
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return _safe_text(value)


def _normalize_mapping(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, Mapping):
        return _canonicalize(raw)
    return {}


def _normalize_string_tuple(raw: Any) -> Tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes, bytearray)):
        values: Iterable[Any] = [raw]
    elif isinstance(raw, Iterable):
        values = raw
    else:
        values = [raw]
    normalized = {_safe_text(entry) for entry in values if _safe_text(entry)}
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class ProofContract:
    contract_id: str
    invariant_requirements: Tuple[str, ...]
    interface_boundary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "invariant_requirements": list(self.invariant_requirements),
            "interface_boundary": self.interface_boundary,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class ProofBoundApiRequest:
    api_request: Dict[str, Any]
    proof_contract: ProofContract
    invariant_requirements: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_request": _canonicalize(self.api_request),
            "proof_contract": self.proof_contract.to_dict(),
            "invariant_requirements": list(self.invariant_requirements),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class ProofReceipt:
    receipt_id: str
    advisory_state: str
    invariant_satisfaction_score: float
    replay_contract_score: float
    serialization_integrity_score: float
    interface_boundary_score: float
    proof_confidence_score: float
    request_hash: str
    response_hash: str
    contract_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "advisory_state": self.advisory_state,
            "invariant_satisfaction_score": self.invariant_satisfaction_score,
            "replay_contract_score": self.replay_contract_score,
            "serialization_integrity_score": self.serialization_integrity_score,
            "interface_boundary_score": self.interface_boundary_score,
            "proof_confidence_score": self.proof_confidence_score,
            "request_hash": self.request_hash,
            "response_hash": self.response_hash,
            "contract_hash": self.contract_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class ProofBoundApiLayer:
    api_request: ProofBoundApiRequest
    api_response: Dict[str, Any]
    proof_receipt: ProofReceipt
    contract_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "api_request": self.api_request.to_dict(),
            "api_response": _canonicalize(self.api_response),
            "proof_receipt": self.proof_receipt.to_dict(),
            "contract_hash": self.contract_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def build_proof_bound_api_request(
    api_request: Any,
    proof_contract: Any,
    invariant_requirements: Any = (),
) -> ProofBoundApiRequest:
    mapping = _normalize_mapping(api_request)
    if isinstance(proof_contract, ProofContract):
        contract = ProofContract(
            contract_id=_safe_text(proof_contract.contract_id, "contract"),
            invariant_requirements=_normalize_string_tuple(proof_contract.invariant_requirements),
            interface_boundary=_safe_text(proof_contract.interface_boundary, "orchestration"),
        )
    else:
        proof_mapping = _normalize_mapping(proof_contract)
        contract = ProofContract(
            contract_id=_safe_text(proof_mapping.get("contract_id"), "contract"),
            invariant_requirements=_normalize_string_tuple(
                proof_mapping.get("invariant_requirements", invariant_requirements)
            ),
            interface_boundary=_safe_text(
                proof_mapping.get("interface_boundary"),
                "orchestration",
            ),
        )
    requirements = _normalize_string_tuple(invariant_requirements)
    if not requirements:
        requirements = contract.invariant_requirements
    return ProofBoundApiRequest(
        api_request=mapping,
        proof_contract=contract,
        invariant_requirements=requirements,
    )


def build_proof_receipt(
    request: ProofBoundApiRequest,
    api_response: Any,
    contract_hash: str,
) -> ProofReceipt:
    normalized_response = _normalize_mapping(api_response)
    request_hash = request.stable_hash()
    response_hash = _stable_hash(normalized_response)
    canonical_response = _canonical_json(normalized_response)
    serialization_integrity_score = 1.0
    try:
        round_trip = json.loads(canonical_response)
        if _canonical_json(round_trip) != canonical_response:
            serialization_integrity_score = 0.0
    except (TypeError, ValueError, json.JSONDecodeError):
        serialization_integrity_score = 0.0

    invariant_satisfaction_score = 1.0 if request.invariant_requirements else 0.5
    replay_contract_score = 1.0 if contract_hash == request.proof_contract.stable_hash() else 0.0
    interface_boundary_score = 1.0 if request.proof_contract.interface_boundary == "orchestration" else 0.0

    proof_confidence_score = _clamp01(
        (
            invariant_satisfaction_score
            + replay_contract_score
            + serialization_integrity_score
            + interface_boundary_score
        )
        / 4.0,
        default=0.0,
    )

    if interface_boundary_score < 1.0:
        advisory_state = "proof_boundary_violation"
    elif replay_contract_score < 1.0:
        advisory_state = "proof_contract_failure"
    elif proof_confidence_score >= 0.999999:
        advisory_state = "proof_verified"
    else:
        advisory_state = "proof_partial"

    preimage = {
        "request_hash": request_hash,
        "response_hash": response_hash,
        "contract_hash": contract_hash,
        "advisory_state": advisory_state,
    }
    receipt_id = f"proof-receipt::{_stable_hash(preimage)[:16]}"

    return ProofReceipt(
        receipt_id=receipt_id,
        advisory_state=advisory_state,
        invariant_satisfaction_score=_clamp01(invariant_satisfaction_score),
        replay_contract_score=_clamp01(replay_contract_score),
        serialization_integrity_score=_clamp01(serialization_integrity_score),
        interface_boundary_score=_clamp01(interface_boundary_score),
        proof_confidence_score=_clamp01(proof_confidence_score),
        request_hash=request_hash,
        response_hash=response_hash,
        contract_hash=contract_hash,
    )


def run_proof_bound_api_layer(
    api_request: Any,
    proof_contract: Any,
    invariant_requirements: Any = (),
    api_response: Any = None,
) -> ProofBoundApiLayer:
    request = build_proof_bound_api_request(
        api_request=api_request,
        proof_contract=proof_contract,
        invariant_requirements=invariant_requirements,
    )
    contract_hash = request.proof_contract.stable_hash()
    normalized_response = _normalize_mapping(api_response)
    receipt = build_proof_receipt(request=request, api_response=normalized_response, contract_hash=contract_hash)
    return ProofBoundApiLayer(
        api_request=request,
        api_response=normalized_response,
        proof_receipt=receipt,
        contract_hash=contract_hash,
    )


def validate_proof_bound_api_layer(layer: Any) -> Dict[str, Any]:
    try:
        non_layer_input = False
        if isinstance(layer, ProofBoundApiLayer):
            candidate = layer
        elif isinstance(layer, Mapping):
            api_request_raw = _normalize_mapping(layer.get("api_request", {}))
            proof_contract_raw = _normalize_mapping(
                layer.get("proof_contract", api_request_raw.get("proof_contract", {}))
            )
            invariant_requirements_raw = layer.get(
                "invariant_requirements",
                api_request_raw.get("invariant_requirements", ()),
            )
            api_response_raw = _normalize_mapping(layer.get("api_response", {}))

            api_request = build_proof_bound_api_request(
                api_request=api_request_raw.get("api_request", api_request_raw),
                proof_contract=proof_contract_raw,
                invariant_requirements=invariant_requirements_raw,
            )
            contract_hash = _safe_text(layer.get("contract_hash"), "")
            receipt_raw = _normalize_mapping(layer.get("proof_receipt", {}))
            candidate = ProofBoundApiLayer(
                api_request=api_request,
                api_response=api_response_raw,
                proof_receipt=ProofReceipt(
                    receipt_id=_safe_text(receipt_raw.get("receipt_id"), ""),
                    advisory_state=_safe_text(receipt_raw.get("advisory_state"), ""),
                    invariant_satisfaction_score=_clamp01(receipt_raw.get("invariant_satisfaction_score")),
                    replay_contract_score=_clamp01(receipt_raw.get("replay_contract_score")),
                    serialization_integrity_score=_clamp01(receipt_raw.get("serialization_integrity_score")),
                    interface_boundary_score=_clamp01(receipt_raw.get("interface_boundary_score")),
                    proof_confidence_score=_clamp01(receipt_raw.get("proof_confidence_score")),
                    request_hash=_safe_text(receipt_raw.get("request_hash"), ""),
                    response_hash=_safe_text(receipt_raw.get("response_hash"), ""),
                    contract_hash=_safe_text(receipt_raw.get("contract_hash"), ""),
                ),
                contract_hash=contract_hash,
            )
        else:
            non_layer_input = True
            candidate = run_proof_bound_api_layer({}, {}, (), {})

        expected_contract_hash = candidate.api_request.proof_contract.stable_hash()
        expected_receipt = build_proof_receipt(
            request=candidate.api_request,
            api_response=candidate.api_response,
            contract_hash=expected_contract_hash,
        )

        violations = []
        if candidate.contract_hash != expected_contract_hash:
            violations.append("contract_hash_mismatch")
        if candidate.proof_receipt.stable_hash() != expected_receipt.stable_hash():
            violations.append("proof_receipt_mismatch")
        if candidate.proof_receipt.advisory_state not in ADVISORY_STATES:
            violations.append("invalid_advisory_state")
        if non_layer_input:
            violations.append("malformed_layer_input")

        return {
            "valid": len(violations) == 0,
            "violations": tuple(sorted(violations)),
            "contract_hash": expected_contract_hash,
        }
    except Exception:
        return {
            "valid": False,
            "violations": ("validator_internal_error",),
            "contract_hash": "",
        }


def compare_proof_api_replay(
    baseline: Any,
    replay: Any,
) -> Dict[str, Any]:
    base_layer = baseline if isinstance(baseline, ProofBoundApiLayer) else run_proof_bound_api_layer(
        api_request=getattr(baseline, "api_request", {}),
        proof_contract=getattr(baseline, "proof_contract", {}),
        invariant_requirements=getattr(baseline, "invariant_requirements", ()),
        api_response=getattr(baseline, "api_response", {}),
    )
    replay_layer = replay if isinstance(replay, ProofBoundApiLayer) else run_proof_bound_api_layer(
        api_request=getattr(replay, "api_request", {}),
        proof_contract=getattr(replay, "proof_contract", {}),
        invariant_requirements=getattr(replay, "invariant_requirements", ()),
        api_response=getattr(replay, "api_response", {}),
    )

    mismatches = []
    if base_layer.contract_hash != replay_layer.contract_hash:
        mismatches.append("contract_hash")
    if base_layer.proof_receipt.stable_hash() != replay_layer.proof_receipt.stable_hash():
        mismatches.append("proof_receipt")
    if base_layer.stable_hash() != replay_layer.stable_hash():
        mismatches.append("layer_hash")

    return {
        "is_stable_replay": len(mismatches) == 0,
        "mismatches": tuple(sorted(mismatches)),
        "baseline_hash": base_layer.stable_hash(),
        "replay_hash": replay_layer.stable_hash(),
    }


def summarize_proof_bound_api(layer: Any) -> str:
    if not isinstance(layer, ProofBoundApiLayer):
        layer = run_proof_bound_api_layer({}, {}, (), {})
    metrics = {
        "invariant_satisfaction_score": layer.proof_receipt.invariant_satisfaction_score,
        "replay_contract_score": layer.proof_receipt.replay_contract_score,
        "serialization_integrity_score": layer.proof_receipt.serialization_integrity_score,
        "interface_boundary_score": layer.proof_receipt.interface_boundary_score,
        "proof_confidence_score": layer.proof_receipt.proof_confidence_score,
    }
    metric_text = ", ".join(f"{name}={metrics[name]:.6f}" for name in _METRIC_ORDER)
    return (
        "ProofBoundApiLayer("
        f"contract_hash={layer.contract_hash}, "
        f"advisory_state={layer.proof_receipt.advisory_state}, "
        f"metrics=[{metric_text}]"
        ")"
    )


__all__ = [
    "ADVISORY_STATES",
    "ProofBoundApiRequest",
    "ProofBoundApiLayer",
    "ProofContract",
    "ProofReceipt",
    "build_proof_bound_api_request",
    "run_proof_bound_api_layer",
    "validate_proof_bound_api_layer",
    "build_proof_receipt",
    "compare_proof_api_replay",
    "summarize_proof_bound_api",
]
