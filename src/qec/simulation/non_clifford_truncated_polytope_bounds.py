# SPDX-License-Identifier: MIT
"""v138.1.3 — Non-Clifford / Truncated Polytope Bounds Pack.

Deterministic approximation-bound analysis for non-Clifford simulation lanes.
This module is additive-only and does not alter decoder behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

NON_CLIFFORD_TRUNCATED_POLYTOPE_BOUNDS_VERSION = "v138.1.3"

SUPPORTED_GATE_FAMILIES: Tuple[str, ...] = (
    "t_gate",
    "ccz",
    "toffoli",
    "qutrit_phase",
    "bosonic_phase",
)

_DEFAULT_THRESHOLD = 0.35

_GATE_BASE_SPAN: Dict[str, float] = {
    "t_gate": 0.20,
    "ccz": 0.24,
    "toffoli": 0.22,
    "qutrit_phase": 0.19,
    "bosonic_phase": 0.26,
}

_GATE_CENTER_SCALE: Dict[str, float] = {
    "t_gate": 0.70,
    "ccz": 0.78,
    "toffoli": 0.75,
    "qutrit_phase": 0.68,
    "bosonic_phase": 0.80,
}


class NonCliffordBoundsValidationError(ValueError):
    """Raised when non-Clifford bounds payload violates deterministic schema."""



def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)



def _stable_hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()



def _normalize_text(value: Any, *, field: str) -> str:
    if value is None:
        raise NonCliffordBoundsValidationError(f"{field} must be non-empty")
    text = str(value).strip()
    if not text:
        raise NonCliffordBoundsValidationError(f"{field} must be non-empty")
    return text



def _normalize_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise NonCliffordBoundsValidationError(f"{field} must be a finite float")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise NonCliffordBoundsValidationError(f"{field} must be a finite float") from exc
    if math.isnan(result) or math.isinf(result):
        raise NonCliffordBoundsValidationError(f"{field} must be finite")
    return result



def _normalize_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise NonCliffordBoundsValidationError(f"{field} must be an integer")
    if not isinstance(value, int):
        raise NonCliffordBoundsValidationError(f"{field} must be an integer")
    return int(value)



def _canonicalize_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise NonCliffordBoundsValidationError(f"{field} contains non-canonical numeric value")
        return float(value)
    if isinstance(value, Mapping):
        normalized: Dict[str, Any] = {}
        for raw_key in sorted(value.keys(), key=lambda x: str(x)):
            key = str(raw_key)
            if key in normalized:
                raise NonCliffordBoundsValidationError(f"{field} contains duplicate canonical key: {key!r}")
            normalized[key] = _canonicalize_value(value[raw_key], field=f"{field}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonicalize_value(item, field=field) for item in value]
    raise NonCliffordBoundsValidationError(f"{field} contains unsupported type: {type(value).__name__}")



def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)



def _as_profile(raw: "GateProfile | Mapping[str, Any]") -> "GateProfile":
    if isinstance(raw, GateProfile):
        raw = raw.to_dict()
    if not isinstance(raw, Mapping):
        raise NonCliffordBoundsValidationError("profile must be mapping or GateProfile")

    gate_family = _normalize_text(raw.get("gate_family"), field="profile.gate_family")
    if gate_family not in SUPPORTED_GATE_FAMILIES:
        raise NonCliffordBoundsValidationError(f"unsupported gate family: {gate_family!r}")

    non_clifford_weight = _normalize_float(raw.get("non_clifford_weight"), field="profile.non_clifford_weight")
    if non_clifford_weight < 0.0:
        raise NonCliffordBoundsValidationError("profile.non_clifford_weight must be >= 0")

    gate_sequence = tuple(
        _normalize_text(item, field="profile.gate_sequence")
        for item in tuple(raw.get("gate_sequence", (gate_family,)))
    )
    if not gate_sequence:
        raise NonCliffordBoundsValidationError("profile.gate_sequence must be non-empty")

    return GateProfile(
        gate_family=gate_family,
        gate_sequence=gate_sequence,
        non_clifford_weight=non_clifford_weight,
        approximation_policy=_normalize_text(raw.get("approximation_policy", "deterministic_bounds"), field="profile.approximation_policy"),
        metadata=_canonicalize_value(dict(raw.get("metadata", {})), field="profile.metadata"),
    )


@dataclass(frozen=True)
class GateProfile:
    gate_family: str
    gate_sequence: Tuple[str, ...]
    non_clifford_weight: float
    approximation_policy: str
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_family": self.gate_family,
            "gate_sequence": list(self.gate_sequence),
            "non_clifford_weight": float(self.non_clifford_weight),
            "approximation_policy": self.approximation_policy,
            "metadata": _canonicalize_value(dict(self.metadata), field="profile.metadata"),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class PolytopeBound:
    region_id: str
    lower_bound: float
    upper_bound: float
    truncation_level: int
    admissible: bool
    pressure_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "lower_bound": float(self.lower_bound),
            "upper_bound": float(self.upper_bound),
            "truncation_level": int(self.truncation_level),
            "admissible": bool(self.admissible),
            "pressure_score": float(self.pressure_score),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())


@dataclass(frozen=True)
class NonCliffordBoundsReceipt:
    profile_hash: str
    bound_set_hash: str
    admissibility_hash: str
    validation_passed: bool
    receipt_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_hash": self.profile_hash,
            "bound_set_hash": self.bound_set_hash,
            "admissibility_hash": self.admissibility_hash,
            "validation_passed": bool(self.validation_passed),
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> Dict[str, Any]:
        return {
            "profile_hash": self.profile_hash,
            "bound_set_hash": self.bound_set_hash,
            "admissibility_hash": self.admissibility_hash,
            "validation_passed": bool(self.validation_passed),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_hash_payload_dict())


@dataclass(frozen=True)
class NonCliffordBoundsAnalysis:
    profile: GateProfile
    bounds: Tuple[PolytopeBound, ...]
    receipt: NonCliffordBoundsReceipt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "bounds": [bound.to_dict() for bound in self.bounds],
            "receipt": self.receipt.to_dict(),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def stable_hash(self) -> str:
        return _stable_hash(self.to_dict())



def compute_polytope_bounds(
    *,
    profile: GateProfile | Mapping[str, Any],
    truncation_level: int,
    admissibility_threshold: float = _DEFAULT_THRESHOLD,
    lane_metadata: Mapping[str, Any] | None = None,
) -> Tuple[PolytopeBound, ...]:
    """Compute deterministic bounded approximation intervals."""
    normalized_profile = _as_profile(profile)
    truncation = _normalize_int(truncation_level, field="truncation_level")
    if truncation < 0:
        raise NonCliffordBoundsValidationError("truncation_level must be >= 0")

    threshold = _normalize_float(admissibility_threshold, field="admissibility_threshold")
    if not 0.0 < threshold <= 1.0:
        raise NonCliffordBoundsValidationError("admissibility_threshold must be in (0,1]")

    _canonicalize_value(dict(lane_metadata or {}), field="lane_metadata")

    pressure = _clamp01(normalized_profile.non_clifford_weight / float(1 + truncation))
    base_span = _GATE_BASE_SPAN[normalized_profile.gate_family] * (1.0 + normalized_profile.non_clifford_weight)
    center_scale = _GATE_CENTER_SCALE[normalized_profile.gate_family]

    bounds: list[PolytopeBound] = []
    for index, gate_name in enumerate(normalized_profile.gate_sequence):
        half_width = base_span / float(2 * (1 + truncation + index))
        center = _clamp01(pressure * center_scale)
        lower = _clamp01(center - half_width)
        upper = _clamp01(center + half_width)
        bounds.append(
            PolytopeBound(
                region_id=f"{normalized_profile.gate_family}:{index:03d}:{gate_name}",
                lower_bound=lower,
                upper_bound=upper,
                truncation_level=truncation,
                admissible=bool(pressure < threshold),
                pressure_score=pressure,
            )
        )

    return tuple(bounds)



def build_non_clifford_bounds(
    *,
    gate_profile: GateProfile | Mapping[str, Any],
    truncation_level: int,
    lane_metadata: Mapping[str, Any] | None = None,
    policy_flags: Mapping[str, Any] | None = None,
) -> NonCliffordBoundsAnalysis:
    """Build deterministic admissibility + polytope bound analysis."""
    normalized_profile = _as_profile(gate_profile)
    normalized_lane_metadata = _canonicalize_value(dict(lane_metadata or {}), field="lane_metadata")
    normalized_policy_flags = _canonicalize_value(dict(policy_flags or {}), field="policy_flags")

    threshold = _normalize_float(normalized_policy_flags.get("admissibility_threshold", _DEFAULT_THRESHOLD), field="policy_flags.admissibility_threshold")
    bounds = compute_polytope_bounds(
        profile=normalized_profile,
        truncation_level=truncation_level,
        admissibility_threshold=threshold,
        lane_metadata=normalized_lane_metadata,
    )

    profile_hash = normalized_profile.stable_hash()
    bound_set_hash = _stable_hash([bound.to_dict() for bound in bounds])
    admissibility_hash = _stable_hash(
        {
            "admissible": [bound.admissible for bound in bounds],
            "pressure_scores": [bound.pressure_score for bound in bounds],
            "truncation_level": int(truncation_level),
            "schema": NON_CLIFFORD_TRUNCATED_POLYTOPE_BOUNDS_VERSION,
        }
    )

    receipt_base = NonCliffordBoundsReceipt(
        profile_hash=profile_hash,
        bound_set_hash=bound_set_hash,
        admissibility_hash=admissibility_hash,
        validation_passed=True,
        receipt_hash="",
    )
    receipt_hash = _stable_hash(receipt_base.to_hash_payload_dict())
    analysis = NonCliffordBoundsAnalysis(
        profile=normalized_profile,
        bounds=bounds,
        receipt=NonCliffordBoundsReceipt(
            profile_hash=profile_hash,
            bound_set_hash=bound_set_hash,
            admissibility_hash=admissibility_hash,
            validation_passed=True,
            receipt_hash=receipt_hash,
        ),
    )

    valid, errors = validate_non_clifford_bounds(analysis)
    if not valid:
        raise NonCliffordBoundsValidationError("; ".join(errors))
    return analysis



def validate_non_clifford_bounds(analysis: NonCliffordBoundsAnalysis) -> tuple[bool, tuple[str, ...]]:
    """Validate deterministic schema, bounds safety, and receipt consistency."""
    errors: list[str] = []

    if analysis.profile.gate_family not in SUPPORTED_GATE_FAMILIES:
        errors.append("profile.gate_family unsupported")

    for idx, bound in enumerate(analysis.bounds):
        if math.isnan(bound.lower_bound) or math.isinf(bound.lower_bound):
            errors.append(f"bounds[{idx}].lower_bound must be finite")
        if math.isnan(bound.upper_bound) or math.isinf(bound.upper_bound):
            errors.append(f"bounds[{idx}].upper_bound must be finite")
        if bound.lower_bound > bound.upper_bound:
            errors.append(f"bounds[{idx}].lower_bound must be <= upper_bound")
        if bound.truncation_level < 0:
            errors.append(f"bounds[{idx}].truncation_level must be >= 0")
        if math.isnan(bound.pressure_score) or math.isinf(bound.pressure_score):
            errors.append(f"bounds[{idx}].pressure_score must be finite")
        if not 0.0 <= bound.pressure_score <= 1.0:
            errors.append(f"bounds[{idx}].pressure_score must be in [0,1]")

    expected_profile_hash = analysis.profile.stable_hash()
    if analysis.receipt.profile_hash != expected_profile_hash:
        errors.append("receipt.profile_hash mismatch")

    expected_bound_set_hash = _stable_hash([bound.to_dict() for bound in analysis.bounds])
    if analysis.receipt.bound_set_hash != expected_bound_set_hash:
        errors.append("receipt.bound_set_hash mismatch")

    expected_admissibility_hash = _stable_hash(
        {
            "admissible": [bound.admissible for bound in analysis.bounds],
            "pressure_scores": [bound.pressure_score for bound in analysis.bounds],
            "truncation_level": int(analysis.bounds[0].truncation_level if analysis.bounds else 0),
            "schema": NON_CLIFFORD_TRUNCATED_POLYTOPE_BOUNDS_VERSION,
        }
    )
    if analysis.receipt.admissibility_hash != expected_admissibility_hash:
        errors.append("receipt.admissibility_hash mismatch")

    if analysis.receipt.receipt_hash != analysis.receipt.stable_hash():
        errors.append("receipt.receipt_hash mismatch")

    ordered = tuple(sorted(errors))
    return (not ordered, ordered)



def admissibility_projection(
    *,
    analysis: NonCliffordBoundsAnalysis,
    lane_id: str,
    orchestration_version: str = "v138.1.2",
) -> Dict[str, Any]:
    """Return deterministic lane-safe admissibility projection for orchestration."""
    lane = _normalize_text(lane_id, field="lane_id")
    orchestration = _normalize_text(orchestration_version, field="orchestration_version")
    admissible = all(bound.admissible for bound in analysis.bounds)
    pressure_max = max((bound.pressure_score for bound in analysis.bounds), default=0.0)
    payload = {
        "lane_id": lane,
        "orchestration_version": orchestration,
        "profile_hash": analysis.receipt.profile_hash,
        "bound_set_hash": analysis.receipt.bound_set_hash,
        "admissibility_hash": analysis.receipt.admissibility_hash,
        "admissible": admissible,
        "pressure_max": float(pressure_max),
        "schema": NON_CLIFFORD_TRUNCATED_POLYTOPE_BOUNDS_VERSION,
    }
    return {**payload, "projection_hash": _stable_hash(payload)}


__all__ = [
    "NON_CLIFFORD_TRUNCATED_POLYTOPE_BOUNDS_VERSION",
    "SUPPORTED_GATE_FAMILIES",
    "NonCliffordBoundsValidationError",
    "GateProfile",
    "PolytopeBound",
    "NonCliffordBoundsReceipt",
    "NonCliffordBoundsAnalysis",
    "build_non_clifford_bounds",
    "validate_non_clifford_bounds",
    "compute_polytope_bounds",
    "admissibility_projection",
]
