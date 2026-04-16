# SPDX-License-Identifier: MIT
"""v138.3.1 — deterministic quadratic tension functional kernel."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

QUADRATIC_TENSION_FUNCTIONAL_KERNEL_VERSION = "v138.3.1"
_QUADRATIC_TENSION_EQUALITY_TOLERANCE = 1e-12


class QuadraticTensionFunctionalValidationError(ValueError):
    """Raised when quadratic tension functional input violates deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise QuadraticTensionFunctionalValidationError(f"{field} contains non-finite float")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda item: str(item)):
            key = str(raw_key)
            if key in normalized:
                raise QuadraticTensionFunctionalValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise QuadraticTensionFunctionalValidationError(f"{field} contains unsupported type: {type(value).__name__}")


def _normalize_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise QuadraticTensionFunctionalValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise QuadraticTensionFunctionalValidationError(f"{field} must be non-empty")
    return normalized


def _normalize_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QuadraticTensionFunctionalValidationError(f"{field} must be numeric")
    normalized = float(value)
    if math.isnan(normalized) or math.isinf(normalized):
        raise QuadraticTensionFunctionalValidationError(f"{field} must be finite")
    return normalized


def _normalize_vector(value: Any, *, field: str, expected_dimension: int | None = None) -> Tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise QuadraticTensionFunctionalValidationError(f"{field} must be a sequence")
    values: list[float] = []
    for index, item in enumerate(value):
        values.append(_normalize_float(item, field=f"{field}[{index}]"))
    if expected_dimension is not None and len(values) != expected_dimension:
        raise QuadraticTensionFunctionalValidationError(
            f"{field} dimension mismatch: expected {expected_dimension}, got {len(values)}"
        )
    return tuple(values)


def _receipt_payload(
    *,
    input_state_hash: str,
    projected_state_hash: str,
    projection_receipt_hash: str,
    tension_hash: str,
    validation_passed: bool,
) -> Dict[str, Any]:
    return {
        "input_state_hash": input_state_hash,
        "projected_state_hash": projected_state_hash,
        "projection_receipt_hash": projection_receipt_hash,
        "tension_hash": tension_hash,
        "validation_passed": bool(validation_passed),
    }


@dataclass(frozen=True)
class QuadraticTensionTerm:
    term_id: str
    coordinate_index: int
    residual_component: float
    squared_component: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "term_id": self.term_id,
            "coordinate_index": int(self.coordinate_index),
            "residual_component": float(self.residual_component),
            "squared_component": float(self.squared_component),
            "metadata": _canonicalize_value(dict(self.metadata), field="term.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class QuadraticTensionReceipt:
    input_state_hash: str
    projected_state_hash: str
    projection_receipt_hash: str
    tension_hash: str
    receipt_hash: str
    validation_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_state_hash": self.input_state_hash,
            "projected_state_hash": self.projected_state_hash,
            "projection_receipt_hash": self.projection_receipt_hash,
            "tension_hash": self.tension_hash,
            "receipt_hash": self.receipt_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return _receipt_payload(
            input_state_hash=self.input_state_hash,
            projected_state_hash=self.projected_state_hash,
            projection_receipt_hash=self.projection_receipt_hash,
            tension_hash=self.tension_hash,
            validation_passed=self.validation_passed,
        )

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class QuadraticTensionValidationReport:
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
class QuadraticTensionFunctional:
    state_id: str
    input_state: Tuple[float, ...]
    projected_state: Tuple[float, ...]
    residual_vector: Tuple[float, ...]
    terms: Tuple[QuadraticTensionTerm, ...]
    tension_value: float
    admissible: bool
    receipt: QuadraticTensionReceipt
    validation: QuadraticTensionValidationReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "input_state": list(self.input_state),
            "projected_state": list(self.projected_state),
            "residual_vector": list(self.residual_vector),
            "terms": [term.to_dict() for term in self.terms],
            "tension_value": float(self.tension_value),
            "admissible": bool(self.admissible),
            "receipt": self.receipt.to_dict(),
            "validation": self.validation.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def compute_quadratic_tension(residual_vector: Sequence[float]) -> Tuple[Tuple[QuadraticTensionTerm, ...], float]:
    normalized_residual = _normalize_vector(residual_vector, field="residual_vector")
    terms: list[QuadraticTensionTerm] = []
    tension_value = 0.0
    for index, component in enumerate(normalized_residual):
        squared = float(component * component)
        if squared < 0.0:
            raise QuadraticTensionFunctionalValidationError(f"terms[{index}].squared_component must be >= 0")
        terms.append(
            QuadraticTensionTerm(
                term_id=f"term-{index:06d}",
                coordinate_index=index,
                residual_component=float(component),
                squared_component=squared,
                metadata={"source": "residual_component"},
            )
        )
        tension_value += squared
    if not math.isfinite(tension_value):
        raise QuadraticTensionFunctionalValidationError("tension_value must be finite")
    if tension_value < 0.0:
        raise QuadraticTensionFunctionalValidationError("tension_value must be >= 0")

    ordered_terms = tuple(sorted(terms, key=lambda term: (term.coordinate_index, term.term_id)))
    return ordered_terms, float(tension_value)


def _extract_projection_mapping(projection: Any) -> Dict[str, Any]:
    if isinstance(projection, Mapping):
        return dict(projection)
    if hasattr(projection, "to_dict") and callable(projection.to_dict):
        data = projection.to_dict()
        if isinstance(data, Mapping):
            return dict(data)
    raise QuadraticTensionFunctionalValidationError(
        "projection must be RuntimeAdmissibilityProjection-compatible object or mapping"
    )


def _build_tension_hash(
    *,
    state_id: str,
    input_state: Tuple[float, ...],
    projected_state: Tuple[float, ...],
    residual_vector: Tuple[float, ...],
    terms: Sequence[QuadraticTensionTerm],
    tension_value: float,
    admissible: bool,
) -> str:
    return _stable_hash(
        {
            "state_id": state_id,
            "input_state": list(input_state),
            "projected_state": list(projected_state),
            "residual_vector": list(residual_vector),
            "terms": [term.to_dict() for term in terms],
            "tension_value": float(tension_value),
            "admissible": bool(admissible),
        }
    )


def _build_receipt(
    *,
    input_state: Tuple[float, ...],
    projected_state: Tuple[float, ...],
    projection_receipt_hash: str,
    tension_hash: str,
    validation_passed: bool,
) -> QuadraticTensionReceipt:
    input_state_hash = _stable_hash({"state": list(input_state)})
    projected_state_hash = _stable_hash({"state": list(projected_state)})
    provisional = QuadraticTensionReceipt(
        input_state_hash=input_state_hash,
        projected_state_hash=projected_state_hash,
        projection_receipt_hash=projection_receipt_hash,
        tension_hash=tension_hash,
        receipt_hash="",
        validation_passed=bool(validation_passed),
    )
    return QuadraticTensionReceipt(
        input_state_hash=provisional.input_state_hash,
        projected_state_hash=provisional.projected_state_hash,
        projection_receipt_hash=provisional.projection_receipt_hash,
        tension_hash=provisional.tension_hash,
        receipt_hash=provisional.stable_hash(),
        validation_passed=provisional.validation_passed,
    )


def build_quadratic_tension_functional(projection: Any) -> QuadraticTensionFunctional:
    payload = _extract_projection_mapping(projection)

    state_id = _normalize_text(payload.get("state_id", "runtime-state"), field="projection.state_id")
    input_state = _normalize_vector(payload.get("input_state"), field="projection.input_state")
    projected_state = _normalize_vector(
        payload.get("projected_state"),
        field="projection.projected_state",
        expected_dimension=len(input_state),
    )

    residual_source = payload.get("residual_vector")
    if residual_source is None:
        residual = payload.get("residual")
        if isinstance(residual, Mapping):
            residual_source = residual.get("residual_vector")
    residual_vector = _normalize_vector(
        residual_source,
        field="projection.residual_vector",
        expected_dimension=len(input_state),
    )

    residual_mapping = payload.get("residual") if isinstance(payload.get("residual"), Mapping) else {}
    admissible = bool(payload.get("admissible", residual_mapping.get("admissible", False)))

    receipt = payload.get("receipt")
    if not isinstance(receipt, Mapping):
        raise QuadraticTensionFunctionalValidationError("projection.receipt must be a mapping")
    projection_receipt_hash = _normalize_text(
        receipt.get("receipt_hash"),
        field="projection.receipt.receipt_hash",
    )

    terms, tension_value = compute_quadratic_tension(residual_vector)
    tension_hash = _build_tension_hash(
        state_id=state_id,
        input_state=input_state,
        projected_state=projected_state,
        residual_vector=residual_vector,
        terms=terms,
        tension_value=tension_value,
        admissible=admissible,
    )

    provisional_receipt = _build_receipt(
        input_state=input_state,
        projected_state=projected_state,
        projection_receipt_hash=projection_receipt_hash,
        tension_hash=tension_hash,
        validation_passed=True,
    )

    provisional = QuadraticTensionFunctional(
        state_id=state_id,
        input_state=input_state,
        projected_state=projected_state,
        residual_vector=residual_vector,
        terms=terms,
        tension_value=tension_value,
        admissible=admissible,
        receipt=provisional_receipt,
        validation=QuadraticTensionValidationReport(valid=True, errors=(), error_count=0),
    )

    validation = validate_quadratic_tension_functional(provisional)
    final_receipt = _build_receipt(
        input_state=input_state,
        projected_state=projected_state,
        projection_receipt_hash=projection_receipt_hash,
        tension_hash=tension_hash,
        validation_passed=validation.valid,
    )

    final = QuadraticTensionFunctional(
        state_id=state_id,
        input_state=input_state,
        projected_state=projected_state,
        residual_vector=residual_vector,
        terms=terms,
        tension_value=tension_value,
        admissible=admissible,
        receipt=final_receipt,
        validation=validation,
    )
    return final


def validate_quadratic_tension_functional(
    functional: QuadraticTensionFunctional | Mapping[str, Any],
) -> QuadraticTensionValidationReport:
    errors: list[str] = []

    if isinstance(functional, QuadraticTensionFunctional):
        payload = functional.to_dict()
    elif isinstance(functional, Mapping):
        payload = dict(functional)
    else:
        return QuadraticTensionValidationReport(
            valid=False,
            errors=("functional must be QuadraticTensionFunctional or mapping",),
            error_count=1,
        )

    input_state = payload.get("input_state")
    projected_state = payload.get("projected_state")
    residual_vector = payload.get("residual_vector")
    terms = payload.get("terms")
    tension_value_raw = payload.get("tension_value")
    receipt = payload.get("receipt")
    state_id = payload.get("state_id")
    admissible = bool(payload.get("admissible"))

    normalized_state: Tuple[float, ...] | None = None
    normalized_projected: Tuple[float, ...] | None = None
    normalized_residual: Tuple[float, ...] | None = None
    normalized_tension: float | None = None
    normalized_terms: list[QuadraticTensionTerm] = []

    if not isinstance(state_id, str) or not state_id.strip():
        errors.append("state_id must be non-empty")

    try:
        normalized_state = _normalize_vector(input_state, field="input_state")
    except QuadraticTensionFunctionalValidationError as exc:
        errors.append(str(exc))

    try:
        expected = len(normalized_state) if normalized_state is not None else None
        normalized_projected = _normalize_vector(projected_state, field="projected_state", expected_dimension=expected)
    except QuadraticTensionFunctionalValidationError as exc:
        errors.append(str(exc))

    try:
        expected = len(normalized_state) if normalized_state is not None else None
        normalized_residual = _normalize_vector(residual_vector, field="residual_vector", expected_dimension=expected)
    except QuadraticTensionFunctionalValidationError as exc:
        errors.append(str(exc))

    try:
        normalized_tension = _normalize_float(tension_value_raw, field="tension_value")
        if normalized_tension < 0.0:
            errors.append("tension_value must be >= 0")
    except QuadraticTensionFunctionalValidationError as exc:
        errors.append(str(exc))

    if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
        errors.append("terms must be a sequence")
    else:
        for index, raw_term in enumerate(terms):
            if not isinstance(raw_term, Mapping):
                errors.append(f"terms[{index}] must be a mapping")
                continue
            try:
                term = QuadraticTensionTerm(
                    term_id=_normalize_text(raw_term.get("term_id"), field=f"terms[{index}].term_id"),
                    coordinate_index=int(raw_term.get("coordinate_index")),
                    residual_component=_normalize_float(
                        raw_term.get("residual_component"), field=f"terms[{index}].residual_component"
                    ),
                    squared_component=_normalize_float(
                        raw_term.get("squared_component"), field=f"terms[{index}].squared_component"
                    ),
                    metadata=_canonicalize_value(dict(raw_term.get("metadata", {})), field=f"terms[{index}].metadata"),
                )
                if term.coordinate_index < 0:
                    errors.append(f"terms[{index}].coordinate_index must be >= 0")
                if term.squared_component < 0.0:
                    errors.append(f"terms[{index}].squared_component must be >= 0")
                normalized_terms.append(term)
            except (QuadraticTensionFunctionalValidationError, TypeError, ValueError) as exc:
                errors.append(str(exc))

    expected_order = sorted(normalized_terms, key=lambda term: (term.coordinate_index, term.term_id))
    if [term.to_dict() for term in normalized_terms] != [term.to_dict() for term in expected_order]:
        errors.append("terms must be sorted by (coordinate_index, term_id)")

    recomputed_tension = float(sum(term.squared_component for term in normalized_terms))
    if normalized_tension is not None and not math.isclose(
        normalized_tension,
        recomputed_tension,
        rel_tol=0.0,
        abs_tol=_QUADRATIC_TENSION_EQUALITY_TOLERANCE,
    ):
        errors.append("tension_value mismatch with sum(terms.squared_component)")

    if not math.isfinite(recomputed_tension) or recomputed_tension < 0.0:
        errors.append("recomputed tension must be finite and >= 0")

    if not isinstance(receipt, Mapping):
        errors.append("receipt must be a mapping")
    else:
        try:
            receipt_obj = QuadraticTensionReceipt(
                input_state_hash=_normalize_text(receipt.get("input_state_hash"), field="receipt.input_state_hash"),
                projected_state_hash=_normalize_text(
                    receipt.get("projected_state_hash"), field="receipt.projected_state_hash"
                ),
                projection_receipt_hash=_normalize_text(
                    receipt.get("projection_receipt_hash"), field="receipt.projection_receipt_hash"
                ),
                tension_hash=_normalize_text(receipt.get("tension_hash"), field="receipt.tension_hash"),
                receipt_hash=_normalize_text(receipt.get("receipt_hash"), field="receipt.receipt_hash"),
                validation_passed=bool(receipt.get("validation_passed")),
            )
            if receipt_obj.receipt_hash != receipt_obj.stable_hash():
                errors.append("receipt.receipt_hash lineage mismatch")
            if normalized_state is not None:
                if receipt_obj.input_state_hash != _stable_hash({"state": list(normalized_state)}):
                    errors.append("receipt.input_state_hash mismatch")
            if normalized_projected is not None:
                if receipt_obj.projected_state_hash != _stable_hash({"state": list(normalized_projected)}):
                    errors.append("receipt.projected_state_hash mismatch")

            if normalized_state is not None and normalized_projected is not None and normalized_residual is not None:
                recomputed_tension_hash = _build_tension_hash(
                    state_id=state_id.strip() if isinstance(state_id, str) else "",
                    input_state=normalized_state,
                    projected_state=normalized_projected,
                    residual_vector=normalized_residual,
                    terms=expected_order,
                    tension_value=float(recomputed_tension),
                    admissible=admissible,
                )
                if receipt_obj.tension_hash != recomputed_tension_hash:
                    errors.append("receipt.tension_hash mismatch")
        except QuadraticTensionFunctionalValidationError as exc:
            errors.append(str(exc))

    valid = len(errors) == 0
    if isinstance(receipt, Mapping):
        receipt_validation_flag = bool(receipt.get("validation_passed"))
        if receipt_validation_flag != valid:
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

    return QuadraticTensionValidationReport(valid=valid, errors=tuple(errors), error_count=len(errors))


def quadratic_tension_projection(functional: QuadraticTensionFunctional | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(functional, QuadraticTensionFunctional):
        return {
            "admissible": bool(functional.admissible),
            "tension_value": float(functional.tension_value),
            "term_count": int(len(functional.terms)),
            "tension_hash": functional.receipt.tension_hash,
            "receipt_hash": functional.receipt.receipt_hash,
        }

    if not isinstance(functional, Mapping):
        raise QuadraticTensionFunctionalValidationError("functional must be QuadraticTensionFunctional or mapping")

    terms = functional.get("terms", ())
    if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
        raise QuadraticTensionFunctionalValidationError("functional.terms must be a sequence")

    receipt = functional.get("receipt", {})
    if not isinstance(receipt, Mapping):
        raise QuadraticTensionFunctionalValidationError("functional.receipt must be a mapping")

    return {
        "admissible": bool(functional.get("admissible")),
        "tension_value": _normalize_float(functional.get("tension_value"), field="functional.tension_value"),
        "term_count": int(len(terms)),
        "tension_hash": _normalize_text(receipt.get("tension_hash"), field="functional.receipt.tension_hash"),
        "receipt_hash": _normalize_text(receipt.get("receipt_hash"), field="functional.receipt.receipt_hash"),
    }
