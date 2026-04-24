"""v148.0 — Governance Validation Kernel.

Deterministic analysis-layer validation that recomputes governance recommendation
from policy memory and verifies replay-stability against an expected governance
artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from qec.analysis.canonical_hashing import canonical_bytes, canonical_json, sha256_hex

GOVERNANCE_VALIDATION_SCHEMA_VERSION = "v148.0"
GOVERNANCE_VALIDATION_MODULE_VERSION = "v148.0"

_VALID_STATUSES = ("VALIDATED", "MISMATCH", "INVALID_INPUT")
_VALID_RECOMMENDATIONS = (
    "MAINTAIN_POLICY",
    "TIGHTEN_POLICY",
    "RELAX_POLICY",
    "ESCALATE_GOVERNANCE",
    "REVIEW_MEMORY",
)
_CANONICAL_SIGNAL_NAMES = (
    "adjustment_pressure",
    "escalation_pressure",
    "forecast_instability_memory",
    "governance_confidence",
    "hold_stability",
    "policy_volatility",
)


_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if hasattr(value, "to_dict"):
        mapped = value.to_dict()
    else:
        mapped = value
    if not isinstance(mapped, Mapping):
        raise ValueError(f"{name} must be a mapping or support to_dict")
    return mapped


def _validated_hash(payload: Mapping[str, Any], *, name: str) -> str:
    if "stable_hash" not in payload:
        raise ValueError(f"{name} must include stable_hash")
    observed = payload["stable_hash"]
    if isinstance(observed, bool) or not isinstance(observed, str) or len(observed) != 64:
        raise ValueError(f"{name}.stable_hash must be 64-char lowercase hex")
    check_payload = {key: payload[key] for key in payload if key != "stable_hash"}
    computed = sha256_hex(check_payload)
    if observed != computed:
        raise ValueError(f"{name} stable_hash mismatch")
    return observed


def _recommendation_from_signals(signals: Mapping[str, float]) -> str:
    escalation_pressure = float(signals["escalation_pressure"])
    governance_confidence = float(signals["governance_confidence"])
    adjustment_pressure = float(signals["adjustment_pressure"])
    hold_stability = float(signals["hold_stability"])
    policy_volatility = float(signals["policy_volatility"])

    if escalation_pressure >= 0.66 and governance_confidence >= 0.5:
        return "ESCALATE_GOVERNANCE"
    if adjustment_pressure >= 0.5:
        return "TIGHTEN_POLICY"
    if hold_stability >= 0.75 and policy_volatility < 0.25:
        return "MAINTAIN_POLICY"
    if governance_confidence < 0.33:
        return "REVIEW_MEMORY"
    return "RELAX_POLICY"


def _extract_signal_map(policy_memory_payload: Mapping[str, Any]) -> dict[str, float]:
    raw_signals = policy_memory_payload.get("signals")
    if not isinstance(raw_signals, (tuple, list)):
        raise ValueError("policy_memory.signals must be list/tuple")
    signal_map: dict[str, float] = {}
    for idx, entry in enumerate(raw_signals):
        if not isinstance(entry, Mapping):
            raise ValueError(f"policy_memory.signals[{idx}] must be mapping")
        name = entry.get("name")
        value = entry.get("value")
        if not isinstance(name, str) or not name:
            raise ValueError(f"policy_memory.signals[{idx}].name must be non-empty str")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"policy_memory.signals[{idx}].value must be finite float")
        signal_value = float(value)
        if signal_value < 0.0 or signal_value > 1.0:
            raise ValueError(f"policy_memory.signals[{idx}].value must be in [0,1]")
        signal_map[name] = signal_value

    if tuple(sorted(signal_map.keys())) != _CANONICAL_SIGNAL_NAMES:
        raise ValueError("policy_memory.signals must include canonical governance signals")
    return signal_map


def _extract_expected_recommendation_label(expected_payload: Mapping[str, Any]) -> str:
    if "recommendation" in expected_payload:
        recommendation = expected_payload["recommendation"]
        if not isinstance(recommendation, Mapping):
            raise ValueError("expected_governance_receipt.recommendation must be mapping")
        label = recommendation.get("label")
    else:
        label = expected_payload.get("label")
    if not isinstance(label, str) or label not in _VALID_RECOMMENDATIONS:
        raise ValueError("invalid recommendation labels")
    return label


@dataclass(frozen=True)
class GovernanceValidationReceipt:
    schema_version: str
    module_version: str
    validation_status: str
    expected_recommendation: str
    recomputed_recommendation: str
    recommendation_stable: bool
    memory_hash: str
    expected_governance_hash: str
    recomputed_governance_hash: str
    validation_hash: str
    stable_hash: str
    validation_score: float
    hash_match: bool

    def __post_init__(self) -> None:
        if self.schema_version != GOVERNANCE_VALIDATION_SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        if self.module_version != GOVERNANCE_VALIDATION_MODULE_VERSION:
            raise ValueError("unsupported module_version")
        if self.validation_status not in _VALID_STATUSES:
            raise ValueError("invalid validation_status")
        if self.expected_recommendation not in _VALID_RECOMMENDATIONS:
            raise ValueError("invalid recommendation labels")
        if self.recomputed_recommendation not in _VALID_RECOMMENDATIONS:
            raise ValueError("invalid recommendation labels")
        if not isinstance(self.recommendation_stable, bool):
            raise ValueError("recommendation_stable must be bool")
        if not isinstance(self.hash_match, bool):
            raise ValueError("hash_match must be bool")
        if isinstance(self.validation_score, bool) or not isinstance(self.validation_score, (int, float)):
            raise ValueError("validation_score must be float")
        if float(self.validation_score) < 0.0 or float(self.validation_score) > 1.0:
            raise ValueError("validation_score must be in [0,1]")

        for field_name in (
            "memory_hash",
            "expected_governance_hash",
            "recomputed_governance_hash",
            "validation_hash",
            "stable_hash",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{field_name} must be 64-char lowercase hex")

        if self.validation_status == "VALIDATED" and (not self.recommendation_stable or not self.hash_match):
            raise ValueError("VALIDATED requires stable recommendation and hash match")

        payload = self._payload_without_hash()
        if self.stable_hash != sha256_hex(payload):
            raise ValueError("stable_hash mismatch")

    def _payload_without_hash(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "module_version": self.module_version,
            "validation_status": self.validation_status,
            "expected_recommendation": self.expected_recommendation,
            "recomputed_recommendation": self.recomputed_recommendation,
            "recommendation_stable": self.recommendation_stable,
            "memory_hash": self.memory_hash,
            "expected_governance_hash": self.expected_governance_hash,
            "recomputed_governance_hash": self.recomputed_governance_hash,
            "validation_hash": self.validation_hash,
            "validation_score": float(self.validation_score),
            "hash_match": self.hash_match,
        }

    def to_dict(self) -> dict[str, _JSONValue]:
        return {**self._payload_without_hash(), "stable_hash": self.stable_hash}

    def to_canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def stable_hash_value(self) -> str:
        return self.stable_hash


def validate_governance_recommendation(
    policy_memory: Any,
    expected_governance_receipt: Any,
) -> GovernanceValidationReceipt:
    policy_memory_payload = _require_mapping(policy_memory, name="policy_memory")
    expected_payload = _require_mapping(expected_governance_receipt, name="expected_governance_receipt")

    memory_hash = _validated_hash(policy_memory_payload, name="policy_memory")
    _validated_hash(expected_payload, name="expected_governance_receipt")

    expected_recommendation = _extract_expected_recommendation_label(expected_payload)
    signal_map = _extract_signal_map(policy_memory_payload)
    recomputed_recommendation = _recommendation_from_signals(signal_map)

    expected_governance_hash = sha256_hex(
        {
            "memory_hash": memory_hash,
            "recommendation": expected_recommendation,
        }
    )
    recomputed_governance_hash = sha256_hex(
        {
            "memory_hash": memory_hash,
            "recommendation": recomputed_recommendation,
        }
    )

    recommendation_stable = expected_recommendation == recomputed_recommendation
    hash_match = expected_governance_hash == recomputed_governance_hash
    status = "VALIDATED" if recommendation_stable and hash_match else "MISMATCH"
    validation_score = 1.0 if status == "VALIDATED" else 0.0

    validation_hash = sha256_hex(
        {
            "memory_hash": memory_hash,
            "expected_recommendation": expected_recommendation,
            "recomputed_recommendation": recomputed_recommendation,
            "expected_governance_hash": expected_governance_hash,
            "recomputed_governance_hash": recomputed_governance_hash,
            "status": status,
        }
    )

    payload_without_hash = {
        "schema_version": GOVERNANCE_VALIDATION_SCHEMA_VERSION,
        "module_version": GOVERNANCE_VALIDATION_MODULE_VERSION,
        "validation_status": status,
        "expected_recommendation": expected_recommendation,
        "recomputed_recommendation": recomputed_recommendation,
        "recommendation_stable": recommendation_stable,
        "memory_hash": memory_hash,
        "expected_governance_hash": expected_governance_hash,
        "recomputed_governance_hash": recomputed_governance_hash,
        "validation_hash": validation_hash,
        "validation_score": validation_score,
        "hash_match": hash_match,
    }

    return GovernanceValidationReceipt(
        schema_version=GOVERNANCE_VALIDATION_SCHEMA_VERSION,
        module_version=GOVERNANCE_VALIDATION_MODULE_VERSION,
        validation_status=status,
        expected_recommendation=expected_recommendation,
        recomputed_recommendation=recomputed_recommendation,
        recommendation_stable=recommendation_stable,
        memory_hash=memory_hash,
        expected_governance_hash=expected_governance_hash,
        recomputed_governance_hash=recomputed_governance_hash,
        validation_hash=validation_hash,
        stable_hash=sha256_hex(payload_without_hash),
        validation_score=validation_score,
        hash_match=hash_match,
    )


__all__ = [
    "GOVERNANCE_VALIDATION_MODULE_VERSION",
    "GOVERNANCE_VALIDATION_SCHEMA_VERSION",
    "GovernanceValidationReceipt",
    "validate_governance_recommendation",
]
