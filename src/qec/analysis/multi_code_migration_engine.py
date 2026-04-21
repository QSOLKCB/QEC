"""v138.9.1 — Deterministic multi-code migration planning engine."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, TypeAlias

_JSONPrimitive: TypeAlias = None | bool | int | float | str
_JSONValue: TypeAlias = _JSONPrimitive | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

ALLOWED_CODE_FAMILIES: tuple[str, ...] = ("ternary", "qldpc", "surface")
SUPPORTED_MIGRATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("qldpc", "surface"),
    ("surface", "surface"),
    ("ternary", "qldpc"),
    ("ternary", "surface"),
)

BOUNDED_PROFILE_FIELDS: tuple[str, ...] = (
    "logical_error_rate",
    "syndrome_density",
    "check_density",
    "hardware_alignment",
    "stability_score",
)
BOUNDED_ASSESSMENT_FIELDS: tuple[str, ...] = (
    "compatibility_score",
    "projected_loss",
    "distance_retention",
    "observable_overlap",
    "hardware_fit",
    "migration_confidence",
)

MIGRATION_STEP_NAMES: tuple[tuple[str, str, str, str], ...] = (
    ("normalize_source_profile", "source_profile", "canonical_profile", "Normalize and validate source profile deterministically."),
    ("map_observable_basis", "observables", "target_observables", "Compute deterministic observable overlap and mapping basis."),
    ("estimate_distance_retention", "distance", "projected_distance", "Estimate bounded distance retention under target family constraints."),
    ("project_density_metrics", "syndrome_density/check_density", "target_density_projection", "Project syndrome/check density transfer characteristics."),
    ("compute_target_alignment", "hardware_alignment", "target_alignment", "Estimate deterministic hardware fit for target family."),
    ("emit_migration_verdict", "assessment", "receipt", "Emit replay-safe admissibility verdict and receipt."),
)

FAMILY_STRUCTURAL_COMPATIBILITY: dict[tuple[str, str], float] = {
    ("ternary", "surface"): 0.760000000000,
    ("qldpc", "surface"): 0.820000000000,
    ("ternary", "qldpc"): 0.880000000000,
    ("surface", "surface"): 1.000000000000,
}

FAMILY_DISTANCE_RETENTION_CAP: dict[tuple[str, str], float] = {
    ("ternary", "surface"): 0.820000000000,
    ("qldpc", "surface"): 0.860000000000,
    ("ternary", "qldpc"): 0.900000000000,
    ("surface", "surface"): 1.000000000000,
}

FAMILY_HARDWARE_MULTIPLIER: dict[tuple[str, str], float] = {
    ("ternary", "surface"): 0.900000000000,
    ("qldpc", "surface"): 0.940000000000,
    ("ternary", "qldpc"): 0.920000000000,
    ("surface", "surface"): 1.000000000000,
}


def _require_non_empty_str(value: str, field: str) -> None:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field} must be a non-empty string")


def _validate_bounded_float(value: float, field: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a float in [0, 1]")
    bounded = float(value)
    if not math.isfinite(bounded) or bounded < 0.0 or bounded > 1.0:
        raise ValueError(f"{field} must be a finite float in [0, 1]")
    return bounded


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float in canonical payload")
        return value
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if not all(isinstance(k, str) for k in keys):
            raise ValueError("canonical payload mappings must use string keys")
        return {k: _canonicalize_json(value[k]) for k in sorted(keys)}
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(_canonicalize_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _normalize_metadata(metadata: Mapping[str, str] | None) -> Mapping[str, str] | None:
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping[str, str] if provided")
    normalized: dict[str, str] = {}
    for key in sorted(metadata.keys()):
        value = metadata[key]
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("metadata must contain only string keys and string values")
        normalized[key] = value
    return normalized


def _validate_observables(observables: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(observables, tuple):
        raise ValueError("observables must be a tuple[str, ...]")
    if len(observables) == 0:
        raise ValueError("observables must be non-empty")
    for observable in observables:
        _require_non_empty_str(observable, "observable")
    if len(set(observables)) != len(observables):
        raise ValueError("observables must be duplicate-free")
    return observables


def _validate_code_family(family: str, field: str = "code_family") -> None:
    if family not in ALLOWED_CODE_FAMILIES:
        raise ValueError(f"{field} must be one of {ALLOWED_CODE_FAMILIES}")


def _bounded(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass(frozen=True)
class CodeStateProfile:
    code_id: str
    code_family: str
    distance: int
    logical_error_rate: float
    syndrome_density: float
    check_density: float
    hardware_alignment: float
    stability_score: float
    observables: tuple[str, ...]
    metadata: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        _require_non_empty_str(self.code_id, "code_id")
        _validate_code_family(self.code_family)
        if not isinstance(self.distance, int) or self.distance < 1:
            raise ValueError("distance must be int >= 1")
        for field in BOUNDED_PROFILE_FIELDS:
            _validate_bounded_float(getattr(self, field), field)
        _validate_observables(self.observables)
        normalized_metadata = _normalize_metadata(self.metadata)
        object.__setattr__(self, "metadata", normalized_metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_id": self.code_id,
            "code_family": self.code_family,
            "distance": self.distance,
            "logical_error_rate": float(self.logical_error_rate),
            "syndrome_density": float(self.syndrome_density),
            "check_density": float(self.check_density),
            "hardware_alignment": float(self.hardware_alignment),
            "stability_score": float(self.stability_score),
            "observables": self.observables,
            "metadata": self.metadata,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class MigrationPolicy:
    target_family: str
    minimum_compatibility: float
    maximum_projected_loss: float
    require_observable_overlap: bool
    prefer_distance_preservation: bool
    prefer_hardware_alignment: bool

    def __post_init__(self) -> None:
        _validate_code_family(self.target_family, field="target_family")
        _validate_bounded_float(self.minimum_compatibility, "minimum_compatibility")
        _validate_bounded_float(self.maximum_projected_loss, "maximum_projected_loss")
        if not isinstance(self.require_observable_overlap, bool):
            raise ValueError("require_observable_overlap must be bool")
        if not isinstance(self.prefer_distance_preservation, bool):
            raise ValueError("prefer_distance_preservation must be bool")
        if not isinstance(self.prefer_hardware_alignment, bool):
            raise ValueError("prefer_hardware_alignment must be bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_family": self.target_family,
            "minimum_compatibility": float(self.minimum_compatibility),
            "maximum_projected_loss": float(self.maximum_projected_loss),
            "require_observable_overlap": self.require_observable_overlap,
            "prefer_distance_preservation": self.prefer_distance_preservation,
            "prefer_hardware_alignment": self.prefer_hardware_alignment,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class MigrationStep:
    step_index: int
    step_type: str
    source_field: str
    target_field: str
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.step_index, int) or self.step_index < 0:
            raise ValueError("step_index must be int >= 0")
        _require_non_empty_str(self.step_type, "step_type")
        _require_non_empty_str(self.source_field, "source_field")
        _require_non_empty_str(self.target_field, "target_field")
        _require_non_empty_str(self.detail, "detail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "step_type": self.step_type,
            "source_field": self.source_field,
            "target_field": self.target_field,
            "detail": self.detail,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class MigrationAssessment:
    compatibility_score: float
    projected_loss: float
    distance_retention: float
    observable_overlap: float
    hardware_fit: float
    migration_confidence: float
    admissible: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in BOUNDED_ASSESSMENT_FIELDS:
            _validate_bounded_float(getattr(self, field), field)
        if not isinstance(self.admissible, bool):
            raise ValueError("admissible must be bool")
        if not isinstance(self.reasons, tuple) or not all(isinstance(r, str) for r in self.reasons):
            raise ValueError("reasons must be tuple[str, ...]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatibility_score": float(self.compatibility_score),
            "projected_loss": float(self.projected_loss),
            "distance_retention": float(self.distance_retention),
            "observable_overlap": float(self.observable_overlap),
            "hardware_fit": float(self.hardware_fit),
            "migration_confidence": float(self.migration_confidence),
            "admissible": self.admissible,
            "reasons": self.reasons,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_dict())


@dataclass(frozen=True)
class MigrationReceipt:
    source_code_id: str
    source_code_family: str
    target_family: str
    selected_migration_path: str
    migration_steps: tuple[MigrationStep, ...]
    assessment: MigrationAssessment
    policy_snapshot: MigrationPolicy
    replay_identity: str
    stable_hash_value: str

    @property
    def stable_hash(self) -> str:
        return self.stable_hash_value

    def __post_init__(self) -> None:
        _require_non_empty_str(self.source_code_id, "source_code_id")
        _validate_code_family(self.source_code_family, "source_code_family")
        _validate_code_family(self.target_family, "target_family")
        _require_non_empty_str(self.selected_migration_path, "selected_migration_path")
        if not isinstance(self.migration_steps, tuple) or not all(isinstance(s, MigrationStep) for s in self.migration_steps):
            raise ValueError("migration_steps must be tuple[MigrationStep, ...]")
        if not isinstance(self.assessment, MigrationAssessment):
            raise ValueError("assessment must be MigrationAssessment")
        if not isinstance(self.policy_snapshot, MigrationPolicy):
            raise ValueError("policy_snapshot must be MigrationPolicy")
        _require_non_empty_str(self.replay_identity, "replay_identity")
        if not isinstance(self.stable_hash_value, str):
            raise ValueError("stable_hash must be a string")

    def to_hash_payload_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("stable_hash")
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_code_id": self.source_code_id,
            "source_code_family": self.source_code_family,
            "target_family": self.target_family,
            "selected_migration_path": self.selected_migration_path,
            "migration_steps": tuple(step.to_dict() for step in self.migration_steps),
            "assessment": self.assessment.to_dict(),
            "policy_snapshot": self.policy_snapshot.to_dict(),
            "replay_identity": self.replay_identity,
            "stable_hash": self.stable_hash_value,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def _compute_observable_overlap(source_observables: tuple[str, ...], target_family: str) -> float:
    target_observable_bases: dict[str, tuple[str, ...]] = {
        "surface": ("X", "Z", "XZ", "plaquette", "star"),
        "qldpc": ("X", "Z", "Y", "parity", "check"),
        "ternary": ("T0", "T1", "T2", "X", "Z"),
    }
    target_basis = target_observable_bases[target_family]
    overlap = len(set(source_observables).intersection(target_basis)) / float(len(set(source_observables)))
    return _bounded(overlap)


def _build_assessment(source_profile: CodeStateProfile, policy: MigrationPolicy) -> MigrationAssessment:
    pair = (source_profile.code_family, policy.target_family)
    base_compatibility = FAMILY_STRUCTURAL_COMPATIBILITY[pair]

    observable_overlap = _compute_observable_overlap(source_profile.observables, policy.target_family)
    density_gap = abs(source_profile.syndrome_density - source_profile.check_density)
    density_similarity = 1.0 - density_gap

    compatibility_score = _bounded(
        0.500000000000 * base_compatibility
        + 0.300000000000 * observable_overlap
        + 0.200000000000 * density_similarity
    )

    distance_retention = _bounded(
        FAMILY_DISTANCE_RETENTION_CAP[pair] * (0.500000000000 + 0.500000000000 * source_profile.stability_score)
    )
    if policy.prefer_distance_preservation:
        distance_retention = _bounded(distance_retention + 0.050000000000)

    hardware_fit = _bounded(FAMILY_HARDWARE_MULTIPLIER[pair] * source_profile.hardware_alignment)
    if policy.prefer_hardware_alignment:
        hardware_fit = _bounded(hardware_fit + 0.050000000000)

    projected_loss = _bounded(
        0.450000000000 * (1.0 - compatibility_score)
        + 0.350000000000 * (1.0 - distance_retention)
        + 0.200000000000 * source_profile.logical_error_rate
    )

    migration_confidence = _bounded(
        0.400000000000 * compatibility_score
        + 0.250000000000 * (1.0 - projected_loss)
        + 0.200000000000 * distance_retention
        + 0.150000000000 * hardware_fit
    )

    reasons: list[str] = []
    admissible = True
    if compatibility_score < policy.minimum_compatibility:
        admissible = False
        reasons.append("compatibility_below_policy_minimum")
    if projected_loss > policy.maximum_projected_loss:
        admissible = False
        reasons.append("projected_loss_above_policy_maximum")
    if policy.require_observable_overlap and observable_overlap <= 0.0:
        admissible = False
        reasons.append("observable_overlap_required")

    if admissible:
        reasons.append("admissible")

    return MigrationAssessment(
        compatibility_score=compatibility_score,
        projected_loss=projected_loss,
        distance_retention=distance_retention,
        observable_overlap=observable_overlap,
        hardware_fit=hardware_fit,
        migration_confidence=migration_confidence,
        admissible=admissible,
        reasons=tuple(reasons),
    )


def _build_steps() -> tuple[MigrationStep, ...]:
    steps = []
    for idx, (name, source_field, target_field, detail) in enumerate(MIGRATION_STEP_NAMES):
        steps.append(
            MigrationStep(
                step_index=idx,
                step_type=name,
                source_field=source_field,
                target_field=target_field,
                detail=detail,
            )
        )
    return tuple(steps)


def plan_code_migration(source_profile: CodeStateProfile, policy: MigrationPolicy) -> MigrationReceipt:
    if not isinstance(source_profile, CodeStateProfile):
        raise ValueError("source_profile must be CodeStateProfile")
    if not isinstance(policy, MigrationPolicy):
        raise ValueError("policy must be MigrationPolicy")

    pair = (source_profile.code_family, policy.target_family)
    if pair not in SUPPORTED_MIGRATION_PAIRS:
        raise ValueError(f"unsupported migration pair: {pair[0]} -> {pair[1]}")

    assessment = _build_assessment(source_profile=source_profile, policy=policy)
    steps = _build_steps()
    selected_migration_path = f"{source_profile.code_family}_to_{policy.target_family}"

    replay_payload = {
        "source_code_id": source_profile.code_id,
        "source_code_family": source_profile.code_family,
        "target_family": policy.target_family,
        "selected_migration_path": selected_migration_path,
        "source_profile_hash": source_profile.stable_hash(),
        "policy_hash": policy.stable_hash(),
        "assessment_hash": assessment.stable_hash(),
        "step_hashes": tuple(step.stable_hash() for step in steps),
    }
    replay_identity = _sha256_hex(replay_payload)

    provisional = MigrationReceipt(
        source_code_id=source_profile.code_id,
        source_code_family=source_profile.code_family,
        target_family=policy.target_family,
        selected_migration_path=selected_migration_path,
        migration_steps=steps,
        assessment=assessment,
        policy_snapshot=policy,
        replay_identity=replay_identity,
        stable_hash_value="",
    )
    stable_hash_value = _sha256_hex(provisional.to_hash_payload_dict())

    return MigrationReceipt(
        source_code_id=provisional.source_code_id,
        source_code_family=provisional.source_code_family,
        target_family=provisional.target_family,
        selected_migration_path=provisional.selected_migration_path,
        migration_steps=provisional.migration_steps,
        assessment=provisional.assessment,
        policy_snapshot=provisional.policy_snapshot,
        replay_identity=provisional.replay_identity,
        stable_hash_value=stable_hash_value,
    )
