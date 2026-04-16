# SPDX-License-Identifier: MIT
"""v138.3.2 — deterministic recovery operator.

Recovery is always projection-aligned: the recovered state equals the projected
admissible state regardless of tension magnitude.  Tension is cross-validated
against state geometry (quadratic squared-L2 norm) in the builder to enforce
lineage and prevent forged tension signals.  Receipt and hash integrity are
enforced at both construction and validation time.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

DETERMINISTIC_RECOVERY_OPERATOR_VERSION = "v138.3.2"
_RECOVERY_EQUALITY_TOLERANCE = 1e-12


class DeterministicRecoveryOperatorValidationError(ValueError):
    """Raised when deterministic recovery operator inputs violate deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _normalize_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise DeterministicRecoveryOperatorValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise DeterministicRecoveryOperatorValidationError(f"{field} must be non-empty")
    return normalized


def _normalize_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeterministicRecoveryOperatorValidationError(f"{field} must be numeric")
    normalized = float(value)
    if math.isnan(normalized) or math.isinf(normalized):
        raise DeterministicRecoveryOperatorValidationError(f"{field} must be finite")
    return normalized


def _normalize_vector(value: Any, *, field: str, expected_dimension: int | None = None) -> Tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DeterministicRecoveryOperatorValidationError(f"{field} must be a sequence")
    values: list[float] = []
    for index, item in enumerate(value):
        values.append(_normalize_float(item, field=f"{field}[{index}]") )
    if expected_dimension is not None and len(values) != expected_dimension:
        raise DeterministicRecoveryOperatorValidationError(
            f"{field} dimension mismatch: expected {expected_dimension}, got {len(values)}"
        )
    return tuple(values)


def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise DeterministicRecoveryOperatorValidationError(f"{field} contains non-finite float")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda item: str(item)):
            key = str(raw_key)
            if key in normalized:
                raise DeterministicRecoveryOperatorValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise DeterministicRecoveryOperatorValidationError(f"{field} contains unsupported type: {type(value).__name__}")


@dataclass(frozen=True)
class RecoveryStep:
    step_id: str
    coordinate_index: int
    original_value: float
    recovered_value: float
    delta: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "coordinate_index": int(self.coordinate_index),
            "original_value": float(self.original_value),
            "recovered_value": float(self.recovered_value),
            "delta": float(self.delta),
            "metadata": _canonicalize_value(dict(self.metadata), field="step.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class RecoveryReceipt:
    input_state_hash: str
    projected_state_hash: str
    tension_hash: str
    recovered_state_hash: str
    recovery_hash: str
    receipt_hash: str
    validation_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_state_hash": self.input_state_hash,
            "projected_state_hash": self.projected_state_hash,
            "tension_hash": self.tension_hash,
            "recovered_state_hash": self.recovered_state_hash,
            "recovery_hash": self.recovery_hash,
            "receipt_hash": self.receipt_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "input_state_hash": self.input_state_hash,
            "projected_state_hash": self.projected_state_hash,
            "tension_hash": self.tension_hash,
            "recovered_state_hash": self.recovered_state_hash,
            "recovery_hash": self.recovery_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class RecoveryValidationReport:
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
class DeterministicRecoveryState:
    state_id: str
    input_state: Tuple[float, ...]
    projected_state: Tuple[float, ...]
    recovered_state: Tuple[float, ...]
    recovery_steps: Tuple[RecoveryStep, ...]
    tension_value: float
    recovery_magnitude: float
    receipt: RecoveryReceipt
    validation: RecoveryValidationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "input_state": list(self.input_state),
            "projected_state": list(self.projected_state),
            "recovered_state": list(self.recovered_state),
            "recovery_steps": [step.to_dict() for step in self.recovery_steps],
            "tension_value": float(self.tension_value),
            "recovery_magnitude": float(self.recovery_magnitude),
            "receipt": self.receipt.to_dict(),
            "validation": self.validation.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def compute_recovery_state(
    input_state: Sequence[float],
    projected_state: Sequence[float],
    tension_value: float,
    *,
    tolerance: float = _RECOVERY_EQUALITY_TOLERANCE,
) -> Tuple[float, ...]:
    normalized_input = _normalize_vector(input_state, field="input_state")
    normalized_projected = _normalize_vector(
        projected_state,
        field="projected_state",
        expected_dimension=len(normalized_input),
    )
    normalized_tension = _normalize_float(tension_value, field="tension_value")
    if normalized_tension < 0.0:
        raise DeterministicRecoveryOperatorValidationError("tension_value must be >= 0")
    normalized_tolerance = _normalize_float(tolerance, field="tolerance")
    if normalized_tolerance < 0.0:
        raise DeterministicRecoveryOperatorValidationError("tolerance must be >= 0")

    # Recovery is always projection-aligned: the recovered state equals the
    # projected admissible state regardless of tension magnitude.  The tension
    # signal influences metadata and receipts but does not alter geometry.
    _ = normalized_tolerance  # retained for API stability; tolerance check is in builder
    return normalized_projected


def _extract_mapping(payload: Any, *, field: str) -> Dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        mapped = payload.to_dict()
        if isinstance(mapped, Mapping):
            return dict(mapped)
    raise DeterministicRecoveryOperatorValidationError(f"{field} must be mapping-compatible")


def _build_recovery_hash(
    *,
    state_id: str,
    input_state: Tuple[float, ...],
    projected_state: Tuple[float, ...],
    recovered_state: Tuple[float, ...],
    steps: Sequence[RecoveryStep],
    tension_value: float,
    recovery_magnitude: float,
    tension_hash: str,
) -> str:
    return _stable_hash(
        {
            "state_id": state_id,
            "input_state": list(input_state),
            "projected_state": list(projected_state),
            "recovered_state": list(recovered_state),
            "recovery_steps": [step.to_dict() for step in steps],
            "tension_value": float(tension_value),
            "recovery_magnitude": float(recovery_magnitude),
            "tension_hash": tension_hash,
        }
    )


def _build_receipt(
    *,
    input_state: Tuple[float, ...],
    projected_state: Tuple[float, ...],
    recovered_state: Tuple[float, ...],
    tension_hash: str,
    recovery_hash: str,
    validation_passed: bool,
) -> RecoveryReceipt:
    provisional = RecoveryReceipt(
        input_state_hash=_stable_hash({"state": list(input_state)}),
        projected_state_hash=_stable_hash({"state": list(projected_state)}),
        tension_hash=tension_hash,
        recovered_state_hash=_stable_hash({"state": list(recovered_state)}),
        recovery_hash=recovery_hash,
        receipt_hash="",
        validation_passed=bool(validation_passed),
    )
    return RecoveryReceipt(
        input_state_hash=provisional.input_state_hash,
        projected_state_hash=provisional.projected_state_hash,
        tension_hash=provisional.tension_hash,
        recovered_state_hash=provisional.recovered_state_hash,
        recovery_hash=provisional.recovery_hash,
        receipt_hash=provisional.stable_hash(),
        validation_passed=provisional.validation_passed,
    )


def _build_recovery_steps(
    *,
    input_state: Tuple[float, ...],
    recovered_state: Tuple[float, ...],
    tension_value: float,
) -> Tuple[RecoveryStep, ...]:
    steps: list[RecoveryStep] = []
    for index, (original, recovered) in enumerate(zip(input_state, recovered_state)):
        delta = float(recovered - original)
        steps.append(
            RecoveryStep(
                step_id=f"recovery-step-{index:06d}",
                coordinate_index=index,
                original_value=float(original),
                recovered_value=float(recovered),
                delta=delta,
                metadata={
                    "source": "projection-aligned",
                    "tension_sign": "positive" if tension_value > _RECOVERY_EQUALITY_TOLERANCE else "zero",
                },
            )
        )
    return tuple(sorted(steps, key=lambda step: (step.coordinate_index, step.step_id)))


def build_deterministic_recovery_state(
    projection: Any,
    tension: Any,
) -> DeterministicRecoveryState:
    projection_map = _extract_mapping(projection, field="projection")
    tension_map = _extract_mapping(tension, field="tension")

    state_id = _normalize_text(projection_map.get("state_id"), field="projection.state_id")
    tension_state_id = _normalize_text(tension_map.get("state_id"), field="tension.state_id")
    if state_id != tension_state_id:
        raise DeterministicRecoveryOperatorValidationError("state lineage mismatch: state_id")

    input_state = _normalize_vector(projection_map.get("input_state"), field="projection.input_state")
    projected_state = _normalize_vector(
        projection_map.get("projected_state"),
        field="projection.projected_state",
        expected_dimension=len(input_state),
    )

    tension_input_state = _normalize_vector(
        tension_map.get("input_state"),
        field="tension.input_state",
        expected_dimension=len(input_state),
    )
    tension_projected_state = _normalize_vector(
        tension_map.get("projected_state"),
        field="tension.projected_state",
        expected_dimension=len(input_state),
    )
    if tension_input_state != input_state:
        raise DeterministicRecoveryOperatorValidationError("state lineage mismatch: input_state")
    if tension_projected_state != projected_state:
        raise DeterministicRecoveryOperatorValidationError("state lineage mismatch: projected_state")

    tension_value = _normalize_float(tension_map.get("tension_value"), field="tension.tension_value")
    if tension_value < 0.0:
        raise DeterministicRecoveryOperatorValidationError("tension.tension_value must be >= 0")

    # Enforce tension-value lineage: recompute the expected quadratic tension
    # (squared L2 norm of the projection residual) from the already-verified
    # state geometry and reject any mapping whose tension_value does not match.
    expected_tension = float(sum((p - i) ** 2 for p, i in zip(projected_state, input_state)))
    if not math.isclose(tension_value, expected_tension, rel_tol=0.0, abs_tol=_RECOVERY_EQUALITY_TOLERANCE):
        raise DeterministicRecoveryOperatorValidationError(
            "tension.tension_value does not match state geometry"
        )

    tension_receipt = tension_map.get("receipt")
    if not isinstance(tension_receipt, Mapping):
        raise DeterministicRecoveryOperatorValidationError("tension.receipt must be a mapping")
    tension_hash = _normalize_text(tension_receipt.get("tension_hash"), field="tension.receipt.tension_hash")

    recovered_state = compute_recovery_state(input_state, projected_state, tension_value)
    recovery_steps = _build_recovery_steps(input_state=input_state, recovered_state=recovered_state, tension_value=tension_value)

    recovery_magnitude = float(math.sqrt(sum(step.delta * step.delta for step in recovery_steps)))
    if not math.isfinite(recovery_magnitude):
        raise DeterministicRecoveryOperatorValidationError("recovery_magnitude must be finite")
    if recovery_magnitude < 0.0:
        raise DeterministicRecoveryOperatorValidationError("recovery_magnitude must be >= 0")

    recovery_hash = _build_recovery_hash(
        state_id=state_id,
        input_state=input_state,
        projected_state=projected_state,
        recovered_state=recovered_state,
        steps=recovery_steps,
        tension_value=tension_value,
        recovery_magnitude=recovery_magnitude,
        tension_hash=tension_hash,
    )

    provisional_receipt = _build_receipt(
        input_state=input_state,
        projected_state=projected_state,
        recovered_state=recovered_state,
        tension_hash=tension_hash,
        recovery_hash=recovery_hash,
        validation_passed=True,
    )

    provisional_state = DeterministicRecoveryState(
        state_id=state_id,
        input_state=input_state,
        projected_state=projected_state,
        recovered_state=recovered_state,
        recovery_steps=recovery_steps,
        tension_value=tension_value,
        recovery_magnitude=recovery_magnitude,
        receipt=provisional_receipt,
        validation=RecoveryValidationReport(valid=True, errors=(), error_count=0),
    )

    validation = validate_deterministic_recovery_state(provisional_state)
    final_receipt = _build_receipt(
        input_state=input_state,
        projected_state=projected_state,
        recovered_state=recovered_state,
        tension_hash=tension_hash,
        recovery_hash=recovery_hash,
        validation_passed=validation.valid,
    )

    return DeterministicRecoveryState(
        state_id=state_id,
        input_state=input_state,
        projected_state=projected_state,
        recovered_state=recovered_state,
        recovery_steps=recovery_steps,
        tension_value=tension_value,
        recovery_magnitude=recovery_magnitude,
        receipt=final_receipt,
        validation=validation,
    )


def validate_deterministic_recovery_state(
    recovery: DeterministicRecoveryState | Mapping[str, Any],
) -> RecoveryValidationReport:
    errors: list[str] = []

    if isinstance(recovery, DeterministicRecoveryState):
        payload = recovery.to_dict()
    elif isinstance(recovery, Mapping):
        payload = dict(recovery)
    else:
        return RecoveryValidationReport(
            valid=False,
            errors=(
                f"recovery must be DeterministicRecoveryState or Mapping, got {type(recovery).__name__}",
            ),
            error_count=1,
        )

    if not isinstance(payload.get("state_id"), str) or not str(payload.get("state_id")).strip():
        errors.append("state_id must be non-empty")

    input_state: Tuple[float, ...] | None = None
    projected_state: Tuple[float, ...] | None = None
    recovered_state: Tuple[float, ...] | None = None
    tension_value: float | None = None
    recovery_magnitude: float | None = None
    normalized_steps: list[RecoveryStep] = []

    try:
        input_state = _normalize_vector(payload.get("input_state"), field="input_state")
    except DeterministicRecoveryOperatorValidationError as exc:
        errors.append(str(exc))

    try:
        projected_state = _normalize_vector(
            payload.get("projected_state"),
            field="projected_state",
            expected_dimension=len(input_state) if input_state is not None else None,
        )
    except DeterministicRecoveryOperatorValidationError as exc:
        errors.append(str(exc))

    try:
        recovered_state = _normalize_vector(
            payload.get("recovered_state"),
            field="recovered_state",
            expected_dimension=len(input_state) if input_state is not None else None,
        )
    except DeterministicRecoveryOperatorValidationError as exc:
        errors.append(str(exc))

    try:
        tension_value = _normalize_float(payload.get("tension_value"), field="tension_value")
        if tension_value < 0.0:
            errors.append("tension_value must be >= 0")
    except DeterministicRecoveryOperatorValidationError as exc:
        errors.append(str(exc))

    try:
        recovery_magnitude = _normalize_float(payload.get("recovery_magnitude"), field="recovery_magnitude")
        if recovery_magnitude < 0.0:
            errors.append("recovery_magnitude must be >= 0")
    except DeterministicRecoveryOperatorValidationError as exc:
        errors.append(str(exc))

    steps = payload.get("recovery_steps")
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        errors.append("recovery_steps must be a sequence")
    else:
        for index, raw_step in enumerate(steps):
            if not isinstance(raw_step, Mapping):
                errors.append(f"recovery_steps[{index}] must be a mapping")
                continue
            try:
                coord = raw_step.get("coordinate_index")
                if isinstance(coord, bool) or not isinstance(coord, int):
                    raise DeterministicRecoveryOperatorValidationError(
                        f"recovery_steps[{index}].coordinate_index must be an integer"
                    )
                step = RecoveryStep(
                    step_id=_normalize_text(raw_step.get("step_id"), field=f"recovery_steps[{index}].step_id"),
                    coordinate_index=coord,
                    original_value=_normalize_float(
                        raw_step.get("original_value"), field=f"recovery_steps[{index}].original_value"
                    ),
                    recovered_value=_normalize_float(
                        raw_step.get("recovered_value"), field=f"recovery_steps[{index}].recovered_value"
                    ),
                    delta=_normalize_float(raw_step.get("delta"), field=f"recovery_steps[{index}].delta"),
                    metadata=_canonicalize_value(
                        dict(raw_step.get("metadata", {})),
                        field=f"recovery_steps[{index}].metadata",
                    ),
                )
                if step.coordinate_index < 0:
                    errors.append(f"recovery_steps[{index}].coordinate_index must be >= 0")
                normalized_steps.append(step)
            except (DeterministicRecoveryOperatorValidationError, TypeError, ValueError) as exc:
                errors.append(str(exc))

    expected_order = sorted(normalized_steps, key=lambda step: (step.coordinate_index, step.step_id))
    if [step.to_dict() for step in normalized_steps] != [step.to_dict() for step in expected_order]:
        errors.append("recovery_steps must be sorted by (coordinate_index, step_id)")

    if input_state is not None and recovered_state is not None and len(normalized_steps) != len(recovered_state):
        errors.append("recovery_steps length must match recovered_state dimension")

    # Enforce that coordinate indices are unique and cover the full range [0..dim-1].
    if input_state is not None and recovered_state is not None and normalized_steps:
        expected_indices = set(range(len(recovered_state)))
        actual_indices = {step.coordinate_index for step in normalized_steps}
        if actual_indices != expected_indices:
            errors.append(
                f"recovery_steps coordinate_index values must be unique and cover [0..{len(recovered_state) - 1}]"
            )

    for idx, step in enumerate(normalized_steps):
        if recovered_state is not None and input_state is not None:
            coord = step.coordinate_index
            if not (0 <= coord < len(recovered_state)):
                errors.append(f"recovery_steps[{idx}].coordinate_index out of bounds")
                continue
            if not math.isclose(step.original_value, input_state[coord], rel_tol=0.0, abs_tol=_RECOVERY_EQUALITY_TOLERANCE):
                errors.append(f"recovery_steps[{idx}].original_value mismatch")
            if not math.isclose(step.recovered_value, recovered_state[coord], rel_tol=0.0, abs_tol=_RECOVERY_EQUALITY_TOLERANCE):
                errors.append(f"recovery_steps[{idx}].recovered_value mismatch")
            expected_delta = recovered_state[coord] - input_state[coord]
            if not math.isclose(step.delta, expected_delta, rel_tol=0.0, abs_tol=_RECOVERY_EQUALITY_TOLERANCE):
                errors.append(f"recovery_steps[{idx}].delta mismatch")

    recomputed_magnitude = float(math.sqrt(sum(step.delta * step.delta for step in normalized_steps)))
    if not math.isfinite(recomputed_magnitude):
        errors.append("recomputed recovery_magnitude must be finite")
    if recomputed_magnitude < 0.0:
        errors.append("recomputed recovery_magnitude must be >= 0")
    if recovery_magnitude is not None and not math.isclose(
        recovery_magnitude,
        recomputed_magnitude,
        rel_tol=0.0,
        abs_tol=_RECOVERY_EQUALITY_TOLERANCE,
    ):
        errors.append("recovery_magnitude mismatch")

    receipt = payload.get("receipt")
    if not isinstance(receipt, Mapping):
        errors.append("receipt must be a mapping")
    else:
        try:
            receipt_obj = RecoveryReceipt(
                input_state_hash=_normalize_text(receipt.get("input_state_hash"), field="receipt.input_state_hash"),
                projected_state_hash=_normalize_text(
                    receipt.get("projected_state_hash"), field="receipt.projected_state_hash"
                ),
                tension_hash=_normalize_text(receipt.get("tension_hash"), field="receipt.tension_hash"),
                recovered_state_hash=_normalize_text(
                    receipt.get("recovered_state_hash"), field="receipt.recovered_state_hash"
                ),
                recovery_hash=_normalize_text(receipt.get("recovery_hash"), field="receipt.recovery_hash"),
                receipt_hash=_normalize_text(receipt.get("receipt_hash"), field="receipt.receipt_hash"),
                validation_passed=bool(receipt.get("validation_passed")),
            )
            if receipt_obj.receipt_hash != receipt_obj.stable_hash():
                errors.append("receipt.receipt_hash lineage mismatch")
            if input_state is not None and receipt_obj.input_state_hash != _stable_hash({"state": list(input_state)}):
                errors.append("receipt.input_state_hash mismatch")
            if projected_state is not None and receipt_obj.projected_state_hash != _stable_hash({"state": list(projected_state)}):
                errors.append("receipt.projected_state_hash mismatch")
            if recovered_state is not None and receipt_obj.recovered_state_hash != _stable_hash({"state": list(recovered_state)}):
                errors.append("receipt.recovered_state_hash mismatch")

            if input_state is not None and projected_state is not None and recovered_state is not None and tension_value is not None:
                recomputed_recovery_hash = _build_recovery_hash(
                    state_id=str(payload.get("state_id", "")).strip(),
                    input_state=input_state,
                    projected_state=projected_state,
                    recovered_state=recovered_state,
                    steps=expected_order,
                    tension_value=tension_value,
                    recovery_magnitude=recomputed_magnitude,
                    tension_hash=receipt_obj.tension_hash,
                )
                if receipt_obj.recovery_hash != recomputed_recovery_hash:
                    errors.append("receipt.recovery_hash mismatch")
        except DeterministicRecoveryOperatorValidationError as exc:
            errors.append(str(exc))

    valid = len(errors) == 0
    if isinstance(receipt, Mapping):
        receipt_validation_passed = bool(receipt.get("validation_passed"))
        if receipt_validation_passed != valid:
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

    return RecoveryValidationReport(valid=valid, errors=tuple(errors), error_count=len(errors))


def recovery_projection(recovery: DeterministicRecoveryState | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(recovery, DeterministicRecoveryState):
        return {
            "recovery_magnitude": float(recovery.recovery_magnitude),
            "recovered_state_hash": recovery.receipt.recovered_state_hash,
            "recovery_hash": recovery.receipt.recovery_hash,
            "receipt_hash": recovery.receipt.receipt_hash,
        }

    if not isinstance(recovery, Mapping):
        raise DeterministicRecoveryOperatorValidationError("recovery must be DeterministicRecoveryState or mapping")

    receipt = recovery.get("receipt", {})
    if not isinstance(receipt, Mapping):
        raise DeterministicRecoveryOperatorValidationError("recovery.receipt must be a mapping")

    return {
        "recovery_magnitude": _normalize_float(recovery.get("recovery_magnitude"), field="recovery.recovery_magnitude"),
        "recovered_state_hash": _normalize_text(receipt.get("recovered_state_hash"), field="recovery.receipt.recovered_state_hash"),
        "recovery_hash": _normalize_text(receipt.get("recovery_hash"), field="recovery.receipt.recovery_hash"),
        "receipt_hash": _normalize_text(receipt.get("receipt_hash"), field="recovery.receipt.receipt_hash"),
    }
