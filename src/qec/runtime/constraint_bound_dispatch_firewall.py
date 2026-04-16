# SPDX-License-Identifier: MIT
"""v138.3.3 — deterministic constraint-bound dispatch firewall."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

CONSTRAINT_BOUND_DISPATCH_FIREWALL_VERSION = "v138.3.3"
_DISPATCH_TOLERANCE = 1e-12
_SUPPORTED_OPERATORS = ("<=", "<", ">=", ">", "==")
_SUPPORTED_VERDICTS = ("allow", "deny", "recover_only")


class ConstraintBoundDispatchFirewallValidationError(ValueError):
    """Raised when dispatch firewall input violates deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _normalize_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ConstraintBoundDispatchFirewallValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ConstraintBoundDispatchFirewallValidationError(f"{field} must be non-empty")
    return normalized


def _normalize_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConstraintBoundDispatchFirewallValidationError(f"{field} must be numeric")
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        raise ConstraintBoundDispatchFirewallValidationError(f"{field} must be finite")
    return numeric


def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ConstraintBoundDispatchFirewallValidationError(f"{field} contains non-finite float")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda item: str(item)):
            key = str(raw_key)
            if key in normalized:
                raise ConstraintBoundDispatchFirewallValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise ConstraintBoundDispatchFirewallValidationError(f"{field} contains unsupported type: {type(value).__name__}")


def _extract_mapping(payload: Any, *, field: str) -> Dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        mapped = payload.to_dict()
        if isinstance(mapped, Mapping):
            return dict(mapped)
    raise ConstraintBoundDispatchFirewallValidationError(f"{field} must be mapping-compatible")


def _is_constraint_non_negative_type(constraint_type: str) -> bool:
    return constraint_type in {"tension_value", "recovery_magnitude"}


@dataclass(frozen=True)
class DispatchConstraint:
    constraint_id: str
    constraint_type: str
    threshold: float
    operator: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type,
            "threshold": float(self.threshold),
            "operator": self.operator,
            "metadata": _canonicalize_value(dict(self.metadata), field="constraint.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class DispatchVerdict:
    verdict: str
    admissible: bool
    recovery_required: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "admissible": bool(self.admissible),
            "recovery_required": bool(self.recovery_required),
            "reason": self.reason,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class DispatchReceipt:
    input_state_hash: str
    recovery_hash: str
    firewall_hash: str
    receipt_hash: str
    validation_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_state_hash": self.input_state_hash,
            "recovery_hash": self.recovery_hash,
            "firewall_hash": self.firewall_hash,
            "receipt_hash": self.receipt_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "input_state_hash": self.input_state_hash,
            "recovery_hash": self.recovery_hash,
            "firewall_hash": self.firewall_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class DispatchValidationReport:
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
class ConstraintBoundDispatchFirewall:
    state_id: str
    tension_value: float
    recovery_magnitude: float
    upstream_admissible: bool
    constraints: Tuple[DispatchConstraint, ...]
    verdict: DispatchVerdict
    receipt: DispatchReceipt
    validation: DispatchValidationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "tension_value": float(self.tension_value),
            "recovery_magnitude": float(self.recovery_magnitude),
            "upstream_admissible": bool(self.upstream_admissible),
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "verdict": self.verdict.to_dict(),
            "receipt": self.receipt.to_dict(),
            "validation": self.validation.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def _normalize_metadata(value: Any, *, field: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConstraintBoundDispatchFirewallValidationError(f"{field} must be a mapping")
    return dict(_canonicalize_value(dict(value), field=field))


def _normalize_constraint(raw: Any, *, field: str) -> DispatchConstraint:
    if isinstance(raw, DispatchConstraint):
        candidate = raw
    elif isinstance(raw, Mapping):
        candidate = DispatchConstraint(
            constraint_id=_normalize_text(raw.get("constraint_id"), field=f"{field}.constraint_id"),
            constraint_type=_normalize_text(raw.get("constraint_type"), field=f"{field}.constraint_type"),
            threshold=_normalize_float(raw.get("threshold"), field=f"{field}.threshold"),
            operator=_normalize_text(raw.get("operator"), field=f"{field}.operator"),
            metadata=_normalize_metadata(raw.get("metadata", {}), field=f"{field}.metadata"),
        )
    else:
        raise ConstraintBoundDispatchFirewallValidationError(f"{field} must be DispatchConstraint or mapping")

    if candidate.operator not in _SUPPORTED_OPERATORS:
        raise ConstraintBoundDispatchFirewallValidationError(f"{field}.operator unsupported: {candidate.operator}")
    if _is_constraint_non_negative_type(candidate.constraint_type) and candidate.threshold < 0.0:
        raise ConstraintBoundDispatchFirewallValidationError(f"{field}.threshold must be >= 0")
    return DispatchConstraint(
        constraint_id=_normalize_text(candidate.constraint_id, field=f"{field}.constraint_id"),
        constraint_type=_normalize_text(candidate.constraint_type, field=f"{field}.constraint_type"),
        threshold=_normalize_float(candidate.threshold, field=f"{field}.threshold"),
        operator=_normalize_text(candidate.operator, field=f"{field}.operator"),
        metadata=_normalize_metadata(candidate.metadata, field=f"{field}.metadata"),
    )


def _sorted_constraints(constraints: Sequence[DispatchConstraint]) -> Tuple[DispatchConstraint, ...]:
    return tuple(sorted(constraints, key=lambda c: (c.constraint_type, c.constraint_id)))


def _compare(actual: float, threshold: float, operator: str, *, tolerance: float = _DISPATCH_TOLERANCE) -> bool:
    if operator == "<=":
        return bool(actual <= threshold)
    if operator == "<":
        return bool(actual < threshold)
    if operator == ">=":
        return bool(actual >= threshold)
    if operator == ">":
        return bool(actual > threshold)
    if operator == "==":
        return bool(math.isclose(actual, threshold, rel_tol=0.0, abs_tol=tolerance))
    raise ConstraintBoundDispatchFirewallValidationError(f"unsupported operator: {operator}")


def _constraint_value(*, constraint_type: str, tension_value: float, recovery_magnitude: float) -> float:
    if constraint_type == "tension_value":
        return tension_value
    if constraint_type == "recovery_magnitude":
        return recovery_magnitude
    raise ConstraintBoundDispatchFirewallValidationError(f"unsupported constraint_type: {constraint_type}")


def evaluate_dispatch_constraints(
    tension_value: float,
    recovery_magnitude: float,
    constraints: Sequence[DispatchConstraint | Mapping[str, Any]],
    *,
    tolerance: float = _DISPATCH_TOLERANCE,
    upstream_admissible: bool = True,
) -> DispatchVerdict:
    normalized_tension = _normalize_float(tension_value, field="tension_value")
    normalized_recovery = _normalize_float(recovery_magnitude, field="recovery_magnitude")
    if normalized_tension < 0.0:
        raise ConstraintBoundDispatchFirewallValidationError("tension_value must be >= 0")
    if normalized_recovery < 0.0:
        raise ConstraintBoundDispatchFirewallValidationError("recovery_magnitude must be >= 0")
    normalized_tolerance = _normalize_float(tolerance, field="tolerance")
    if normalized_tolerance < 0.0:
        raise ConstraintBoundDispatchFirewallValidationError("tolerance must be >= 0")

    if not bool(upstream_admissible):
        return DispatchVerdict(
            verdict="deny",
            admissible=False,
            recovery_required=False,
            reason="upstream_admissibility_failure",
        )

    normalized_constraints = _sorted_constraints(
        [_normalize_constraint(raw, field=f"constraints[{index}]") for index, raw in enumerate(constraints)]
    )

    has_soft_failure = False
    for constraint in normalized_constraints:
        actual = _constraint_value(
            constraint_type=constraint.constraint_type,
            tension_value=normalized_tension,
            recovery_magnitude=normalized_recovery,
        )
        if not _compare(actual, constraint.threshold, constraint.operator, tolerance=normalized_tolerance):
            is_hard = bool(constraint.metadata.get("hard", True))
            if is_hard:
                return DispatchVerdict(
                    verdict="deny",
                    admissible=False,
                    recovery_required=False,
                    reason=f"constraint_failed:{constraint.constraint_type}:{constraint.constraint_id}",
                )
            has_soft_failure = True

    if has_soft_failure or normalized_recovery > normalized_tolerance:
        if has_soft_failure and normalized_recovery > normalized_tolerance:
            reason = "soft_constraint_failure_and_recovery_above_tolerance"
        elif has_soft_failure:
            reason = "soft_constraint_failure"
        else:
            reason = "recovery_magnitude_above_tolerance"
        return DispatchVerdict(
            verdict="recover_only",
            admissible=True,
            recovery_required=True,
            reason=reason,
        )

    return DispatchVerdict(
        verdict="allow",
        admissible=True,
        recovery_required=False,
        reason="all_constraints_satisfied",
    )


def _build_firewall_hash(
    *,
    state_id: str,
    tension_value: float,
    recovery_magnitude: float,
    upstream_admissible: bool,
    constraints: Sequence[DispatchConstraint],
    verdict: DispatchVerdict,
) -> str:
    return _stable_hash(
        {
            "state_id": state_id,
            "tension_value": float(tension_value),
            "recovery_magnitude": float(recovery_magnitude),
            "upstream_admissible": bool(upstream_admissible),
            "constraints": [constraint.to_dict() for constraint in constraints],
            "verdict": verdict.to_dict(),
        }
    )


def _build_receipt(
    *,
    input_state_hash: str,
    recovery_hash: str,
    firewall_hash: str,
    validation_passed: bool,
) -> DispatchReceipt:
    provisional = DispatchReceipt(
        input_state_hash=input_state_hash,
        recovery_hash=recovery_hash,
        firewall_hash=firewall_hash,
        receipt_hash="",
        validation_passed=bool(validation_passed),
    )
    return DispatchReceipt(
        input_state_hash=provisional.input_state_hash,
        recovery_hash=provisional.recovery_hash,
        firewall_hash=provisional.firewall_hash,
        receipt_hash=provisional.stable_hash(),
        validation_passed=provisional.validation_passed,
    )


def build_constraint_bound_dispatch_firewall(
    recovery: Any,
    tension: Any,
    constraints: Sequence[DispatchConstraint | Mapping[str, Any]],
) -> ConstraintBoundDispatchFirewall:
    recovery_map = _extract_mapping(recovery, field="recovery")
    tension_map = _extract_mapping(tension, field="tension")

    state_id = _normalize_text(recovery_map.get("state_id"), field="recovery.state_id")
    tension_state_id = _normalize_text(tension_map.get("state_id"), field="tension.state_id")
    if state_id != tension_state_id:
        raise ConstraintBoundDispatchFirewallValidationError("state lineage mismatch: state_id")

    recovery_tension_value = _normalize_float(recovery_map.get("tension_value"), field="recovery.tension_value")
    tension_value = _normalize_float(tension_map.get("tension_value"), field="tension.tension_value")
    if recovery_tension_value < 0.0 or tension_value < 0.0:
        raise ConstraintBoundDispatchFirewallValidationError("tension_value must be >= 0")
    if not math.isclose(recovery_tension_value, tension_value, rel_tol=0.0, abs_tol=_DISPATCH_TOLERANCE):
        raise ConstraintBoundDispatchFirewallValidationError("state lineage mismatch: tension_value")

    recovery_magnitude = _normalize_float(recovery_map.get("recovery_magnitude"), field="recovery.recovery_magnitude")
    if recovery_magnitude < 0.0:
        raise ConstraintBoundDispatchFirewallValidationError("recovery.recovery_magnitude must be >= 0")

    recovery_receipt = recovery_map.get("receipt")
    if not isinstance(recovery_receipt, Mapping):
        raise ConstraintBoundDispatchFirewallValidationError("recovery.receipt must be a mapping")

    input_state_hash = _normalize_text(recovery_receipt.get("input_state_hash"), field="recovery.receipt.input_state_hash")
    recovery_hash = _normalize_text(recovery_receipt.get("recovery_hash"), field="recovery.receipt.recovery_hash")

    normalized_constraints = _sorted_constraints(
        tuple(_normalize_constraint(constraint, field=f"constraints[{index}]") for index, constraint in enumerate(constraints))
    )
    upstream_admissible = bool(tension_map.get("admissible", True))
    verdict = evaluate_dispatch_constraints(
        tension_value, recovery_magnitude, normalized_constraints,
        upstream_admissible=upstream_admissible,
    )

    firewall_hash = _build_firewall_hash(
        state_id=state_id,
        tension_value=tension_value,
        recovery_magnitude=recovery_magnitude,
        upstream_admissible=upstream_admissible,
        constraints=normalized_constraints,
        verdict=verdict,
    )

    provisional_receipt = _build_receipt(
        input_state_hash=input_state_hash,
        recovery_hash=recovery_hash,
        firewall_hash=firewall_hash,
        validation_passed=True,
    )

    provisional_firewall = ConstraintBoundDispatchFirewall(
        state_id=state_id,
        tension_value=tension_value,
        recovery_magnitude=recovery_magnitude,
        upstream_admissible=upstream_admissible,
        constraints=normalized_constraints,
        verdict=verdict,
        receipt=provisional_receipt,
        validation=DispatchValidationReport(valid=True, errors=(), error_count=0),
    )

    validation = validate_constraint_bound_dispatch_firewall(provisional_firewall)
    final_receipt = _build_receipt(
        input_state_hash=input_state_hash,
        recovery_hash=recovery_hash,
        firewall_hash=firewall_hash,
        validation_passed=validation.valid,
    )

    return ConstraintBoundDispatchFirewall(
        state_id=state_id,
        tension_value=tension_value,
        recovery_magnitude=recovery_magnitude,
        upstream_admissible=upstream_admissible,
        constraints=normalized_constraints,
        verdict=verdict,
        receipt=final_receipt,
        validation=validation,
    )


def validate_constraint_bound_dispatch_firewall(
    firewall: ConstraintBoundDispatchFirewall | Mapping[str, Any],
) -> DispatchValidationReport:
    errors: list[str] = []

    if isinstance(firewall, ConstraintBoundDispatchFirewall):
        payload = firewall.to_dict()
    elif isinstance(firewall, Mapping):
        payload = dict(firewall)
    else:
        return DispatchValidationReport(
            valid=False,
            errors=("firewall must be ConstraintBoundDispatchFirewall or Mapping",),
            error_count=1,
        )

    try:
        state_id = _normalize_text(payload.get("state_id"), field="state_id")
    except ConstraintBoundDispatchFirewallValidationError as exc:
        errors.append(str(exc))
        state_id = ""

    try:
        tension_value = _normalize_float(payload.get("tension_value"), field="tension_value")
        if tension_value < 0.0:
            errors.append("tension_value must be >= 0")
    except ConstraintBoundDispatchFirewallValidationError as exc:
        errors.append(str(exc))
        tension_value = None

    try:
        recovery_magnitude = _normalize_float(payload.get("recovery_magnitude"), field="recovery_magnitude")
        if recovery_magnitude < 0.0:
            errors.append("recovery_magnitude must be >= 0")
    except ConstraintBoundDispatchFirewallValidationError as exc:
        errors.append(str(exc))
        recovery_magnitude = None

    raw_constraints = payload.get("constraints")
    normalized_constraints: list[DispatchConstraint] = []
    if not isinstance(raw_constraints, Sequence) or isinstance(raw_constraints, (str, bytes)):
        errors.append("constraints must be a sequence")
    else:
        for index, raw_constraint in enumerate(raw_constraints):
            try:
                normalized_constraints.append(_normalize_constraint(raw_constraint, field=f"constraints[{index}]") )
            except ConstraintBoundDispatchFirewallValidationError as exc:
                errors.append(str(exc))

    expected_constraints = _sorted_constraints(normalized_constraints)
    if [c.to_dict() for c in normalized_constraints] != [c.to_dict() for c in expected_constraints]:
        errors.append("constraints must be sorted by (constraint_type, constraint_id)")

    verdict_map = payload.get("verdict")
    verdict_obj: DispatchVerdict | None = None
    if not isinstance(verdict_map, Mapping):
        errors.append("verdict must be a mapping")
    else:
        try:
            verdict_obj = DispatchVerdict(
                verdict=_normalize_text(verdict_map.get("verdict"), field="verdict.verdict"),
                admissible=bool(verdict_map.get("admissible")),
                recovery_required=bool(verdict_map.get("recovery_required")),
                reason=_normalize_text(verdict_map.get("reason"), field="verdict.reason"),
            )
            if verdict_obj.verdict not in _SUPPORTED_VERDICTS:
                errors.append(f"verdict.verdict unsupported: {verdict_obj.verdict}")
        except ConstraintBoundDispatchFirewallValidationError as exc:
            errors.append(str(exc))

    upstream_admissible = bool(payload.get("upstream_admissible", True))

    if verdict_obj is not None and tension_value is not None and recovery_magnitude is not None:
        try:
            expected_verdict = evaluate_dispatch_constraints(
                tension_value, recovery_magnitude, expected_constraints,
                upstream_admissible=upstream_admissible,
            )
            if verdict_obj.to_dict() != expected_verdict.to_dict():
                errors.append("verdict mismatch")
        except ConstraintBoundDispatchFirewallValidationError as exc:
            errors.append(str(exc))

    receipt_map = payload.get("receipt")
    if not isinstance(receipt_map, Mapping):
        errors.append("receipt must be a mapping")
    else:
        try:
            receipt_obj = DispatchReceipt(
                input_state_hash=_normalize_text(receipt_map.get("input_state_hash"), field="receipt.input_state_hash"),
                recovery_hash=_normalize_text(receipt_map.get("recovery_hash"), field="receipt.recovery_hash"),
                firewall_hash=_normalize_text(receipt_map.get("firewall_hash"), field="receipt.firewall_hash"),
                receipt_hash=_normalize_text(receipt_map.get("receipt_hash"), field="receipt.receipt_hash"),
                validation_passed=bool(receipt_map.get("validation_passed")),
            )
            if receipt_obj.receipt_hash != receipt_obj.stable_hash():
                errors.append("receipt.receipt_hash lineage mismatch")
            if verdict_obj is not None and tension_value is not None and recovery_magnitude is not None:
                recomputed_firewall_hash = _build_firewall_hash(
                    state_id=state_id,
                    tension_value=tension_value,
                    recovery_magnitude=recovery_magnitude,
                    upstream_admissible=upstream_admissible,
                    constraints=expected_constraints,
                    verdict=verdict_obj,
                )
                if receipt_obj.firewall_hash != recomputed_firewall_hash:
                    errors.append("receipt.firewall_hash mismatch")
        except ConstraintBoundDispatchFirewallValidationError as exc:
            errors.append(str(exc))

    valid = len(errors) == 0
    if isinstance(receipt_map, Mapping) and bool(receipt_map.get("validation_passed")) != valid:
        errors.append("receipt.validation_passed mismatch")
        valid = False

    try:
        canonical = _canonical_json(payload)
        if _canonical_json(json.loads(canonical)) != canonical:
            errors.append("canonical JSON stability violation")
            valid = False
    except (TypeError, ValueError):
        errors.append("canonical JSON stability violation")
        valid = False

    return DispatchValidationReport(valid=valid, errors=tuple(errors), error_count=len(errors))


def dispatch_firewall_projection(
    firewall: ConstraintBoundDispatchFirewall | Mapping[str, Any],
) -> Dict[str, Any]:
    if isinstance(firewall, ConstraintBoundDispatchFirewall):
        return {
            "verdict": firewall.verdict.verdict,
            "recovery_required": bool(firewall.verdict.recovery_required),
            "firewall_hash": firewall.receipt.firewall_hash,
            "receipt_hash": firewall.receipt.receipt_hash,
        }

    if not isinstance(firewall, Mapping):
        raise ConstraintBoundDispatchFirewallValidationError(
            "firewall must be ConstraintBoundDispatchFirewall or mapping"
        )

    verdict = firewall.get("verdict")
    receipt = firewall.get("receipt")
    if not isinstance(verdict, Mapping):
        raise ConstraintBoundDispatchFirewallValidationError("firewall.verdict must be a mapping")
    if not isinstance(receipt, Mapping):
        raise ConstraintBoundDispatchFirewallValidationError("firewall.receipt must be a mapping")

    verdict_value = _normalize_text(verdict.get("verdict"), field="firewall.verdict.verdict")
    if verdict_value not in _SUPPORTED_VERDICTS:
        raise ConstraintBoundDispatchFirewallValidationError("firewall.verdict.verdict unsupported")

    return {
        "verdict": verdict_value,
        "recovery_required": bool(verdict.get("recovery_required")),
        "firewall_hash": _normalize_text(receipt.get("firewall_hash"), field="firewall.receipt.firewall_hash"),
        "receipt_hash": _normalize_text(receipt.get("receipt_hash"), field="firewall.receipt.receipt_hash"),
    }
