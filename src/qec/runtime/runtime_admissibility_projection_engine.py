# SPDX-License-Identifier: MIT
"""v138.3.0 — deterministic runtime admissibility projection engine."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

RUNTIME_ADMISSIBILITY_PROJECTION_ENGINE_VERSION = "v138.3.0"
_ADMISSIBILITY_EPSILON = 1e-12


class RuntimeAdmissibilityProjectionValidationError(ValueError):
    """Raised when runtime admissibility projection input violates deterministic schema."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise RuntimeAdmissibilityProjectionValidationError(f"{field} contains non-finite float")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda item: str(item)):
            key = str(raw_key)
            if key in normalized:
                raise RuntimeAdmissibilityProjectionValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise RuntimeAdmissibilityProjectionValidationError(f"{field} contains unsupported type: {type(value).__name__}")


def _normalize_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise RuntimeAdmissibilityProjectionValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise RuntimeAdmissibilityProjectionValidationError(f"{field} must be non-empty")
    return normalized


def _normalize_dimension(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeAdmissibilityProjectionValidationError(f"{field} must be an integer")
    result = int(value)
    if result <= 0:
        raise RuntimeAdmissibilityProjectionValidationError(f"{field} must be > 0")
    return result


def _normalize_vector(value: Any, *, field: str, expected_dimension: int | None = None) -> Tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeAdmissibilityProjectionValidationError(f"{field} must be a sequence")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise RuntimeAdmissibilityProjectionValidationError(f"{field}[{index}] must be a finite float")
        numeric = float(item)
        if math.isnan(numeric) or math.isinf(numeric):
            raise RuntimeAdmissibilityProjectionValidationError(f"{field}[{index}] must be a finite float")
        result.append(numeric)
    if expected_dimension is not None and len(result) != expected_dimension:
        raise RuntimeAdmissibilityProjectionValidationError(
            f"{field} dimension mismatch: expected {expected_dimension}, got {len(result)}"
        )
    return tuple(result)


def _normalize_basis_vectors(value: Any, *, dimension: int, field: str) -> Tuple[Tuple[float, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeAdmissibilityProjectionValidationError(f"{field} must be a sequence")
    if len(value) == 0:
        raise RuntimeAdmissibilityProjectionValidationError("basis_vectors must be non-empty")
    basis: list[Tuple[float, ...]] = []
    for index, vector in enumerate(value):
        basis.append(_normalize_vector(vector, field=f"{field}[{index}]", expected_dimension=dimension))
    return tuple(basis)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _subtract(a: Sequence[float], b: Sequence[float]) -> Tuple[float, ...]:
    return tuple(float(x) - float(y) for x, y in zip(a, b))


def _add_scaled(acc: Sequence[float], vec: Sequence[float], scale: float) -> Tuple[float, ...]:
    return tuple(float(current) + float(scale) * float(component) for current, component in zip(acc, vec))


def _gram_schmidt(basis_vectors: Sequence[Tuple[float, ...]]) -> Tuple[Tuple[float, ...], ...]:
    orthonormal: list[Tuple[float, ...]] = []
    for vector in basis_vectors:
        candidate = tuple(vector)
        for unit in orthonormal:
            candidate = _subtract(candidate, tuple(_dot(candidate, unit) * component for component in unit))
        norm_sq = _dot(candidate, candidate)
        if norm_sq <= _ADMISSIBILITY_EPSILON:
            continue
        norm = math.sqrt(norm_sq)
        orthonormal.append(tuple(component / norm for component in candidate))
    return tuple(orthonormal)


def _project_onto_subspace(state: Tuple[float, ...], basis_vectors: Tuple[Tuple[float, ...], ...]) -> Tuple[float, ...]:
    orthonormal_basis = _gram_schmidt(basis_vectors)
    projection = tuple(0.0 for _ in state)
    for unit in orthonormal_basis:
        projection = _add_scaled(projection, unit, _dot(state, unit))
    return projection


def _build_receipt(
    *,
    input_state: Tuple[float, ...],
    projected_state: Tuple[float, ...],
    subspace: "AdmissibleSubspace",
    admissible: bool,
) -> "ProjectionProofReceipt":
    input_state_hash = _stable_hash({"state": list(input_state)})
    projected_state_hash = _stable_hash({"state": list(projected_state)})
    subspace_hash = subspace.stable_hash()
    proof_hash = _stable_hash(
        {
            "input_state_hash": input_state_hash,
            "projected_state_hash": projected_state_hash,
            "subspace_hash": subspace_hash,
            "admissible": bool(admissible),
        }
    )
    provisional = ProjectionProofReceipt(
        input_state_hash=input_state_hash,
        projected_state_hash=projected_state_hash,
        subspace_hash=subspace_hash,
        admissible=bool(admissible),
        proof_hash=proof_hash,
        receipt_hash="",
    )
    return ProjectionProofReceipt(
        input_state_hash=provisional.input_state_hash,
        projected_state_hash=provisional.projected_state_hash,
        subspace_hash=provisional.subspace_hash,
        admissible=provisional.admissible,
        proof_hash=provisional.proof_hash,
        receipt_hash=provisional.stable_hash(),
    )


@dataclass(frozen=True)
class AdmissibleSubspace:
    subspace_id: str
    dimension: int
    basis_vectors: Tuple[Tuple[float, ...], ...]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subspace_id": self.subspace_id,
            "dimension": int(self.dimension),
            "basis_vectors": [list(vector) for vector in self.basis_vectors],
            "metadata": _canonicalize_value(dict(self.metadata), field="subspace.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimeResidual:
    residual_norm: float
    residual_vector: Tuple[float, ...]
    admissible: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "residual_norm": float(self.residual_norm),
            "residual_vector": list(self.residual_vector),
            "admissible": bool(self.admissible),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class ProjectionProofReceipt:
    input_state_hash: str
    projected_state_hash: str
    subspace_hash: str
    admissible: bool
    proof_hash: str
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_state_hash": self.input_state_hash,
            "projected_state_hash": self.projected_state_hash,
            "subspace_hash": self.subspace_hash,
            "admissible": bool(self.admissible),
            "proof_hash": self.proof_hash,
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "input_state_hash": self.input_state_hash,
            "projected_state_hash": self.projected_state_hash,
            "subspace_hash": self.subspace_hash,
            "admissible": bool(self.admissible),
            "proof_hash": self.proof_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class RuntimeAdmissibilityProjection:
    state_id: str
    input_state: Tuple[float, ...]
    projected_state: Tuple[float, ...]
    residual: RuntimeResidual
    receipt: ProjectionProofReceipt
    validation_errors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "input_state": list(self.input_state),
            "projected_state": list(self.projected_state),
            "residual": self.residual.to_dict(),
            "receipt": self.receipt.to_dict(),
            "validation_errors": list(self.validation_errors),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


def _normalize_subspace(subspace: AdmissibleSubspace | Mapping[str, Any]) -> AdmissibleSubspace:
    if isinstance(subspace, AdmissibleSubspace):
        return AdmissibleSubspace(
            subspace_id=_normalize_text(subspace.subspace_id, field="subspace.subspace_id"),
            dimension=_normalize_dimension(subspace.dimension, field="subspace.dimension"),
            basis_vectors=_normalize_basis_vectors(
                subspace.basis_vectors,
                dimension=_normalize_dimension(subspace.dimension, field="subspace.dimension"),
                field="subspace.basis_vectors",
            ),
            metadata=_canonicalize_value(dict(subspace.metadata), field="subspace.metadata"),
        )
    if not isinstance(subspace, Mapping):
        raise RuntimeAdmissibilityProjectionValidationError("subspace must be AdmissibleSubspace or mapping")
    dimension = _normalize_dimension(subspace.get("dimension"), field="subspace.dimension")
    return AdmissibleSubspace(
        subspace_id=_normalize_text(subspace.get("subspace_id"), field="subspace.subspace_id"),
        dimension=dimension,
        basis_vectors=_normalize_basis_vectors(subspace.get("basis_vectors"), dimension=dimension, field="subspace.basis_vectors"),
        metadata=_canonicalize_value(dict(subspace.get("metadata", {})), field="subspace.metadata"),
    )


def compute_runtime_residual(
    state_vector: Sequence[float],
    admissible_basis: Sequence[Sequence[float]],
) -> RuntimeResidual:
    state = _normalize_vector(state_vector, field="state_vector")
    if len(state) == 0:
        raise RuntimeAdmissibilityProjectionValidationError("state_vector must be non-empty")
    basis = _normalize_basis_vectors(admissible_basis, dimension=len(state), field="admissible_basis")
    projected_state = _project_onto_subspace(state, basis)
    residual_vector = _subtract(state, projected_state)
    residual_norm = float(math.sqrt(_dot(residual_vector, residual_vector)))
    if residual_norm < 0.0:
        raise RuntimeAdmissibilityProjectionValidationError("residual_norm must be >= 0")
    return RuntimeResidual(
        residual_norm=residual_norm,
        residual_vector=residual_vector,
        admissible=bool(residual_norm <= _ADMISSIBILITY_EPSILON),
    )


def project_runtime_state(
    runtime_state: Mapping[str, Any] | Sequence[float],
    admissible_subspace: AdmissibleSubspace | Mapping[str, Any],
) -> RuntimeAdmissibilityProjection:
    subspace = _normalize_subspace(admissible_subspace)
    if isinstance(runtime_state, Mapping):
        state_id = _normalize_text(runtime_state.get("state_id"), field="runtime_state.state_id")
        input_state = _normalize_vector(
            runtime_state.get("state"),
            field="runtime_state.state",
            expected_dimension=subspace.dimension,
        )
    else:
        state_id = "runtime-state"
        input_state = _normalize_vector(runtime_state, field="runtime_state", expected_dimension=subspace.dimension)

    projected_state = _project_onto_subspace(input_state, subspace.basis_vectors)
    residual = RuntimeResidual(
        residual_norm=float(math.sqrt(_dot(_subtract(input_state, projected_state), _subtract(input_state, projected_state)))),
        residual_vector=_subtract(input_state, projected_state),
        admissible=bool(compute_runtime_residual(input_state, subspace.basis_vectors).admissible),
    )
    if residual.residual_norm < 0.0:
        raise RuntimeAdmissibilityProjectionValidationError("residual_norm must be >= 0")

    receipt = _build_receipt(
        input_state=input_state,
        projected_state=projected_state,
        subspace=subspace,
        admissible=residual.admissible,
    )
    projection = RuntimeAdmissibilityProjection(
        state_id=state_id,
        input_state=input_state,
        projected_state=projected_state,
        residual=residual,
        receipt=receipt,
        validation_errors=(),
    )
    errors = validate_runtime_projection(projection)
    return RuntimeAdmissibilityProjection(
        state_id=projection.state_id,
        input_state=projection.input_state,
        projected_state=projection.projected_state,
        residual=projection.residual,
        receipt=projection.receipt,
        validation_errors=errors,
    )


def validate_runtime_projection(projection: RuntimeAdmissibilityProjection | Mapping[str, Any]) -> Tuple[str, ...]:
    errors: list[str] = []
    if isinstance(projection, RuntimeAdmissibilityProjection):
        payload = projection.to_dict()
    elif isinstance(projection, Mapping):
        payload = dict(projection)
    else:
        return ("projection must be RuntimeAdmissibilityProjection or mapping",)

    state = payload.get("input_state")
    projected = payload.get("projected_state")
    residual = payload.get("residual")
    receipt = payload.get("receipt")

    if not isinstance(state, Sequence) or isinstance(state, (str, bytes)):
        errors.append("input_state must be a sequence")
    if not isinstance(projected, Sequence) or isinstance(projected, (str, bytes)):
        errors.append("projected_state must be a sequence")

    if isinstance(state, Sequence) and isinstance(projected, Sequence):
        if len(state) != len(projected):
            errors.append("input_state/projected_state dimension mismatch")

    if isinstance(state, Sequence):
        for idx, value in enumerate(state):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(float(value)) or math.isinf(float(value)):
                errors.append(f"input_state[{idx}] must be finite")
                break

    if isinstance(projected, Sequence):
        for idx, value in enumerate(projected):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(float(value)) or math.isinf(float(value)):
                errors.append(f"projected_state[{idx}] must be finite")
                break

    if not isinstance(residual, Mapping):
        errors.append("residual must be a mapping")
    else:
        residual_norm = residual.get("residual_norm")
        if isinstance(residual_norm, bool) or not isinstance(residual_norm, (int, float)):
            errors.append("residual.residual_norm must be numeric")
        else:
            residual_norm_float = float(residual_norm)
            if math.isnan(residual_norm_float) or math.isinf(residual_norm_float):
                errors.append("residual.residual_norm must be finite")
            if residual_norm_float < 0.0:
                errors.append("residual.residual_norm must be >= 0")

    if not isinstance(receipt, Mapping):
        errors.append("receipt must be a mapping")
    else:
        try:
            receipt_obj = ProjectionProofReceipt(
                input_state_hash=_normalize_text(receipt.get("input_state_hash"), field="receipt.input_state_hash"),
                projected_state_hash=_normalize_text(receipt.get("projected_state_hash"), field="receipt.projected_state_hash"),
                subspace_hash=_normalize_text(receipt.get("subspace_hash"), field="receipt.subspace_hash"),
                admissible=bool(receipt.get("admissible")),
                proof_hash=_normalize_text(receipt.get("proof_hash"), field="receipt.proof_hash"),
                receipt_hash=_normalize_text(receipt.get("receipt_hash"), field="receipt.receipt_hash"),
            )
            if receipt_obj.receipt_hash != receipt_obj.stable_hash():
                errors.append("receipt.receipt_hash lineage mismatch")
            recomputed_proof = _stable_hash(
                {
                    "input_state_hash": receipt_obj.input_state_hash,
                    "projected_state_hash": receipt_obj.projected_state_hash,
                    "subspace_hash": receipt_obj.subspace_hash,
                    "admissible": bool(receipt_obj.admissible),
                }
            )
            if receipt_obj.proof_hash != recomputed_proof:
                errors.append("receipt.proof_hash lineage mismatch")
            if isinstance(state, Sequence):
                if receipt_obj.input_state_hash != _stable_hash({"state": [float(v) for v in state]}):
                    errors.append("receipt.input_state_hash mismatch")
            if isinstance(projected, Sequence):
                if receipt_obj.projected_state_hash != _stable_hash({"state": [float(v) for v in projected]}):
                    errors.append("receipt.projected_state_hash mismatch")
        except RuntimeAdmissibilityProjectionValidationError as exc:
            errors.append(str(exc))

    try:
        canonical = _canonical_json(payload)
        if _canonical_json(json.loads(canonical)) != canonical:
            errors.append("canonical JSON stability violation")
    except (TypeError, ValueError):
        errors.append("canonical JSON stability violation")

    return tuple(errors)


def runtime_projection_summary(
    projection: RuntimeAdmissibilityProjection | Mapping[str, Any],
) -> Dict[str, Any]:
    if isinstance(projection, RuntimeAdmissibilityProjection):
        return {
            "admissible": bool(projection.residual.admissible),
            "residual_norm": float(projection.residual.residual_norm),
            "projected_state_hash": projection.receipt.projected_state_hash,
            "receipt_hash": projection.receipt.receipt_hash,
        }
    if not isinstance(projection, Mapping):
        raise RuntimeAdmissibilityProjectionValidationError("projection must be RuntimeAdmissibilityProjection or mapping")
    residual = projection.get("residual", {})
    receipt = projection.get("receipt", {})
    return {
        "admissible": bool(residual.get("admissible")),
        "residual_norm": float(residual.get("residual_norm", 0.0)),
        "projected_state_hash": str(receipt.get("projected_state_hash", "")),
        "receipt_hash": str(receipt.get("receipt_hash", "")),
    }
