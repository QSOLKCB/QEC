# SPDX-License-Identifier: MIT
"""v138.2.1 — deterministic latency-budget enforcement hardware layer.

This module is additive to v138.2.0 hardware control dispatch and provides
mathematical policy enforcement (no wall-clock timing, no async, no threads).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Tuple

from qec.runtime.fpga_asic_control_module import HardwareControlDispatch

LATENCY_BUDGET_ENFORCEMENT_VERSION = "v138.2.1"

_SUPPORTED_VIOLATION_ACTIONS: Tuple[str, ...] = ("allow", "throttle", "reroute", "reject")
_SUPPORTED_DECISIONS: Tuple[str, ...] = ("allow", "throttle", "reroute", "reject")
_SUPPORTED_VIOLATION_CLASSES: Tuple[str, ...] = ("nominal", "warning", "violation", "hard_breach")
_SHA256_HEX_CHARS: frozenset = frozenset("0123456789abcdef")


class LatencyBudgetValidationError(ValueError):
    """Raised when latency budget enforcement data violates deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _normalize_text(value: Any, *, field: str) -> str:
    if value is None or not isinstance(value, str):
        raise LatencyBudgetValidationError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise LatencyBudgetValidationError(f"{field} must be non-empty")
    return text


def _normalize_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LatencyBudgetValidationError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise LatencyBudgetValidationError(f"{field} must be >= {minimum}")
    return result


def _normalize_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise LatencyBudgetValidationError(f"{field} must be a boolean")
    return value


def _normalize_sha256_hex(value: Any, *, field: str) -> str:
    text = _normalize_text(value, field=field).lower()
    if len(text) != 64:
        raise LatencyBudgetValidationError(f"{field} must be a 64-character SHA-256 hex string")
    if not frozenset(text) <= _SHA256_HEX_CHARS:
        raise LatencyBudgetValidationError(f"{field} must be lowercase SHA-256 hex")
    return text


def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise LatencyBudgetValidationError(f"{field} contains non-canonical numeric value")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda x: str(x)):
            key = str(raw_key)
            if key in normalized:
                raise LatencyBudgetValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise LatencyBudgetValidationError(f"{field} contains unsupported type: {type(value).__name__}")


@dataclass(frozen=True)
class LatencyBudgetPolicy:
    policy_id: str
    max_latency_ns: int
    hard_limit_ns: int
    violation_action: str
    recovery_mode: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "max_latency_ns": int(self.max_latency_ns),
            "hard_limit_ns": int(self.hard_limit_ns),
            "violation_action": self.violation_action,
            "recovery_mode": self.recovery_mode,
            "metadata": _canonicalize_value(dict(self.metadata), field="policy.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class LatencyEnforcementDecision:
    dispatch_id: str
    projected_latency_ns: int
    within_budget: bool
    hard_limit_breached: bool
    decision: str
    enforcement_reason: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dispatch_id": self.dispatch_id,
            "projected_latency_ns": int(self.projected_latency_ns),
            "within_budget": bool(self.within_budget),
            "hard_limit_breached": bool(self.hard_limit_breached),
            "decision": self.decision,
            "enforcement_reason": self.enforcement_reason,
            "metadata": _canonicalize_value(dict(self.metadata), field="decision.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class LatencyBudgetReceipt:
    policy_hash: str
    dispatch_hash: str
    decision_hash: str
    within_budget: bool
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_hash": self.policy_hash,
            "dispatch_hash": self.dispatch_hash,
            "decision_hash": self.decision_hash,
            "within_budget": bool(self.within_budget),
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "policy_hash": self.policy_hash,
            "dispatch_hash": self.dispatch_hash,
            "decision_hash": self.decision_hash,
            "within_budget": bool(self.within_budget),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class LatencyBudgetValidationReport:
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
class LatencyBudgetEnforcement:
    policy: LatencyBudgetPolicy
    decision: LatencyEnforcementDecision
    receipt: LatencyBudgetReceipt
    validation: LatencyBudgetValidationReport
    target_family: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "decision": self.decision.to_dict(),
            "receipt": self.receipt.to_dict(),
            "validation": self.validation.to_dict(),
            "target_family": self.target_family,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def _normalize_policy(raw: LatencyBudgetPolicy | Mapping[str, Any]) -> LatencyBudgetPolicy:
    if isinstance(raw, LatencyBudgetPolicy):
        return raw
    if not isinstance(raw, Mapping):
        raise LatencyBudgetValidationError("policy must be mapping or LatencyBudgetPolicy")
    _meta = raw.get("metadata")
    return LatencyBudgetPolicy(
        policy_id=_normalize_text(raw.get("policy_id"), field="policy.policy_id"),
        max_latency_ns=_normalize_int(raw.get("max_latency_ns"), field="policy.max_latency_ns"),
        hard_limit_ns=_normalize_int(raw.get("hard_limit_ns"), field="policy.hard_limit_ns"),
        violation_action=_normalize_text(raw.get("violation_action"), field="policy.violation_action"),
        recovery_mode=_normalize_text(raw.get("recovery_mode"), field="policy.recovery_mode"),
        metadata=_canonicalize_value(_meta if isinstance(_meta, Mapping) else {}, field="policy.metadata"),
    )


def _normalize_dispatch_context(
    raw: HardwareControlDispatch | Mapping[str, Any],
) -> Dict[str, str]:
    if isinstance(raw, HardwareControlDispatch):
        return {
            "dispatch_id": raw.dispatch.dispatch_id,
            "dispatch_hash": raw.dispatch.stable_hash(),
            "target_family": raw.target.target_family,
            "latency_hash": raw.latency_receipt.latency_hash,
        }
    if not isinstance(raw, Mapping):
        raise LatencyBudgetValidationError("dispatch_receipt must be mapping or HardwareControlDispatch")

    dispatch_map = raw.get("dispatch", {})
    target_map = raw.get("target", {})
    control_map = raw.get("control_receipt", {})
    latency_map = raw.get("latency_receipt", {})
    return {
        "dispatch_id": _normalize_sha256_hex(dispatch_map.get("dispatch_id"), field="dispatch.dispatch_id"),
        "dispatch_hash": _normalize_sha256_hex(control_map.get("dispatch_hash"), field="control_receipt.dispatch_hash"),
        "target_family": _normalize_text(target_map.get("target_family"), field="target.target_family"),
        "latency_hash": _normalize_sha256_hex(latency_map.get("latency_hash"), field="latency_receipt.latency_hash"),
    }


def compute_latency_violation_class(*, projected_latency_ns: int, policy: LatencyBudgetPolicy | Mapping[str, Any]) -> str:
    normalized_policy = _normalize_policy(policy)
    projected = _normalize_int(projected_latency_ns, field="projected_latency_ns", minimum=0)

    if projected <= normalized_policy.max_latency_ns:
        return "nominal"
    if projected > normalized_policy.hard_limit_ns:
        return "hard_breach"

    soft_band = normalized_policy.hard_limit_ns - normalized_policy.max_latency_ns
    warning_ceiling = normalized_policy.max_latency_ns + (soft_band // 2)
    if projected <= warning_ceiling:
        return "warning"
    return "violation"


def _resolve_decision(*, violation_class: str, policy: LatencyBudgetPolicy, target_family: str) -> str:
    if violation_class == "nominal":
        return "allow"
    if violation_class == "hard_breach":
        return "reject"

    # For warning/violation: honour the policy's violation_action.
    # "reroute" is restricted to simulation_shadow targets; fall back to "throttle" otherwise.
    action = policy.violation_action
    if action == "reroute" and target_family != "simulation_shadow":
        return "throttle"
    return action


def _extract_target_family(decision: LatencyEnforcementDecision) -> str:
    metadata_target = decision.metadata.get("target_family")
    if isinstance(metadata_target, str):
        candidate = metadata_target.strip()
        if candidate:
            return candidate

    parts = decision.enforcement_reason.split(":", 2)
    if len(parts) == 3:
        candidate = parts[2].strip()
        if candidate:
            return candidate
    return "unknown"


def enforce_latency_budget(
    *,
    dispatch_receipt: HardwareControlDispatch | Mapping[str, Any],
    projected_latency_ns: int,
    budget_policy: LatencyBudgetPolicy | Mapping[str, Any],
) -> LatencyBudgetEnforcement:
    """Deterministically enforce latency budget for a dispatch intent."""
    policy = _normalize_policy(budget_policy)
    context = _normalize_dispatch_context(dispatch_receipt)
    projected = _normalize_int(projected_latency_ns, field="projected_latency_ns", minimum=0)

    violation_class = compute_latency_violation_class(projected_latency_ns=projected, policy=policy)
    within_budget = projected <= policy.max_latency_ns
    hard_limit_breached = projected > policy.hard_limit_ns
    decision_str = _resolve_decision(
        violation_class=violation_class,
        policy=policy,
        target_family=context["target_family"],
    )

    decision = LatencyEnforcementDecision(
        dispatch_id=context["dispatch_id"],
        projected_latency_ns=projected,
        within_budget=within_budget,
        hard_limit_breached=hard_limit_breached,
        decision=decision_str,
        enforcement_reason=f"{violation_class}:{policy.violation_action}:{context['target_family']}",
        metadata={
            "violation_class": violation_class,
            "recovery_mode": policy.recovery_mode,
            "source_latency_hash": context["latency_hash"],
            "target_family": context["target_family"],
            "enforcement_version": LATENCY_BUDGET_ENFORCEMENT_VERSION,
        },
    )

    receipt = LatencyBudgetReceipt(
        policy_hash=policy.stable_hash(),
        dispatch_hash=context["dispatch_hash"],
        decision_hash=decision.stable_hash(),
        within_budget=within_budget,
        receipt_hash="",
    )
    receipt = LatencyBudgetReceipt(
        policy_hash=receipt.policy_hash,
        dispatch_hash=receipt.dispatch_hash,
        decision_hash=receipt.decision_hash,
        within_budget=receipt.within_budget,
        receipt_hash=receipt.stable_hash(),
    )

    provisional = LatencyBudgetEnforcement(
        policy=policy,
        decision=decision,
        receipt=receipt,
        validation=LatencyBudgetValidationReport(valid=True, errors=(), error_count=0),
        target_family=context["target_family"],
    )
    validation = validate_latency_budget(provisional)
    return LatencyBudgetEnforcement(
        policy=policy,
        decision=decision,
        receipt=receipt,
        validation=validation,
        target_family=context["target_family"],
    )


def validate_latency_budget(
    enforcement_obj: LatencyBudgetEnforcement | Mapping[str, Any],
) -> LatencyBudgetValidationReport:
    """Validate deterministic latency-budget enforcement semantics and receipt consistency."""
    errors = []

    try:
        if isinstance(enforcement_obj, LatencyBudgetEnforcement):
            enforcement = enforcement_obj
        elif isinstance(enforcement_obj, Mapping):
            policy = _normalize_policy(enforcement_obj["policy"])
            decision_map = enforcement_obj["decision"]
            receipt_map = enforcement_obj["receipt"]
            validation_map = enforcement_obj.get("validation", {"valid": True, "errors": [], "error_count": 0})
            decision_meta = decision_map.get("metadata")
            decision = LatencyEnforcementDecision(
                dispatch_id=_normalize_sha256_hex(decision_map["dispatch_id"], field="decision.dispatch_id"),
                projected_latency_ns=_normalize_int(
                    decision_map["projected_latency_ns"], field="decision.projected_latency_ns", minimum=0
                ),
                within_budget=_normalize_bool(decision_map["within_budget"], field="decision.within_budget"),
                hard_limit_breached=_normalize_bool(
                    decision_map["hard_limit_breached"], field="decision.hard_limit_breached"
                ),
                decision=_normalize_text(decision_map["decision"], field="decision.decision"),
                enforcement_reason=_normalize_text(
                    decision_map["enforcement_reason"], field="decision.enforcement_reason"
                ),
                metadata=_canonicalize_value(
                    decision_meta if isinstance(decision_meta, Mapping) else {},
                    field="decision.metadata",
                ),
            )
            tf_raw = enforcement_obj.get("target_family")
            resolved_target_family = (
                tf_raw.strip()
                if isinstance(tf_raw, str) and tf_raw.strip()
                else _extract_target_family(decision)
            )
            enforcement = LatencyBudgetEnforcement(
                policy=policy,
                decision=decision,
                receipt=LatencyBudgetReceipt(
                    policy_hash=_normalize_sha256_hex(receipt_map["policy_hash"], field="receipt.policy_hash"),
                    dispatch_hash=_normalize_sha256_hex(receipt_map["dispatch_hash"], field="receipt.dispatch_hash"),
                    decision_hash=_normalize_sha256_hex(receipt_map["decision_hash"], field="receipt.decision_hash"),
                    within_budget=_normalize_bool(receipt_map["within_budget"], field="receipt.within_budget"),
                    receipt_hash=_normalize_sha256_hex(receipt_map["receipt_hash"], field="receipt.receipt_hash"),
                ),
                validation=LatencyBudgetValidationReport(
                    valid=_normalize_bool(validation_map.get("valid", True), field="validation.valid"),
                    errors=tuple(str(v) for v in tuple(validation_map.get("errors", ()))),
                    error_count=_normalize_int(validation_map.get("error_count", 0), field="validation.error_count", minimum=0),
                ),
                target_family=resolved_target_family,
            )
        else:
            raise LatencyBudgetValidationError("enforcement_obj must be mapping or LatencyBudgetEnforcement")
    except Exception as exc:
        return LatencyBudgetValidationReport(valid=False, errors=(f"normalization_failed: {exc}",), error_count=1)

    if enforcement.policy.max_latency_ns <= 0:
        errors.append("policy.max_latency_ns must be > 0")
    if enforcement.policy.hard_limit_ns < enforcement.policy.max_latency_ns:
        errors.append("policy.hard_limit_ns must be >= policy.max_latency_ns")
    if enforcement.decision.projected_latency_ns < 0:
        errors.append("decision.projected_latency_ns must be >= 0")

    if enforcement.policy.violation_action not in _SUPPORTED_VIOLATION_ACTIONS:
        errors.append(f"unsupported policy.violation_action: {enforcement.policy.violation_action!r}")
    if enforcement.decision.decision not in _SUPPORTED_DECISIONS:
        errors.append(f"unsupported decision.decision: {enforcement.decision.decision!r}")

    recomputed_class = compute_latency_violation_class(
        projected_latency_ns=enforcement.decision.projected_latency_ns,
        policy=enforcement.policy,
    )
    if recomputed_class not in _SUPPORTED_VIOLATION_CLASSES:
        errors.append(f"unsupported violation class: {recomputed_class!r}")

    expected_within_budget = enforcement.decision.projected_latency_ns <= enforcement.policy.max_latency_ns
    if enforcement.decision.within_budget != expected_within_budget:
        errors.append("decision.within_budget invariant violated")
    if enforcement.receipt.within_budget != enforcement.decision.within_budget:
        errors.append("receipt.within_budget does not match decision.within_budget")

    expected_hard_breach = enforcement.decision.projected_latency_ns > enforcement.policy.hard_limit_ns
    if enforcement.decision.hard_limit_breached != expected_hard_breach:
        errors.append("decision.hard_limit_breached invariant violated")
    if expected_hard_breach and enforcement.decision.decision != "reject":
        errors.append("hard limit breach must force reject decision")

    expected_decision = _resolve_decision(
        violation_class=recomputed_class,
        policy=enforcement.policy,
        target_family=enforcement.target_family,
    )
    if enforcement.decision.decision != expected_decision:
        errors.append("decision.decision is inconsistent with policy semantics")

    if enforcement.receipt.policy_hash != enforcement.policy.stable_hash():
        errors.append("receipt.policy_hash mismatch")
    if enforcement.receipt.decision_hash != enforcement.decision.stable_hash():
        errors.append("receipt.decision_hash mismatch")
    if enforcement.receipt.receipt_hash != enforcement.receipt.stable_hash():
        errors.append("receipt.receipt_hash mismatch")

    if _canonical_json(enforcement.to_dict()) != enforcement.to_canonical_json():
        errors.append("canonical ordering mismatch")

    return LatencyBudgetValidationReport(valid=not errors, errors=tuple(errors), error_count=len(errors))


def replay_timing_projection(
    enforcement_obj: LatencyBudgetEnforcement,
) -> Dict[str, Any]:
    """Emit deterministic replay-safe timing lineage for throughput studies."""
    violation_class = compute_latency_violation_class(
        projected_latency_ns=enforcement_obj.decision.projected_latency_ns,
        policy=enforcement_obj.policy,
    )
    payload = {
        "module_version": LATENCY_BUDGET_ENFORCEMENT_VERSION,
        "dispatch_id": enforcement_obj.decision.dispatch_id,
        "projected_latency_ns": enforcement_obj.decision.projected_latency_ns,
        "decision": enforcement_obj.decision.decision,
        "within_budget": enforcement_obj.decision.within_budget,
        "hard_limit_breached": enforcement_obj.decision.hard_limit_breached,
        "violation_class": violation_class,
        "policy_hash": enforcement_obj.receipt.policy_hash,
        "dispatch_hash": enforcement_obj.receipt.dispatch_hash,
        "decision_hash": enforcement_obj.receipt.decision_hash,
        "receipt_hash": enforcement_obj.receipt.receipt_hash,
        "enforcement_hash": enforcement_obj.stable_hash(),
    }
    payload["projection_hash"] = _stable_hash(payload)
    return payload
