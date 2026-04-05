"""Deterministic Layer-4 quantum noise/fidelity observatory.

Theory invariants preserved by this module:

- FIDELITY_STABILITY_OBSERVATION_LAW:
  Same normalized noise/fidelity trajectory yields byte-identical observatory
  artifacts.
- BOUNDED_NOISE_SCORE_INVARIANT:
  Noise score derives only from explicit bounded inputs and remains in [0, 1].
- DETERMINISTIC_COMPENSATION_METRIC_RULE:
  Compensation metrics derive only from explicit noise/drift/compensation values.
- REPLAY_SAFE_NOISE_AUDIT_CHAIN:
  Audit entries form a parent-linked SHA-256 lineage.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _round12(value: float) -> float:
    return round(float(value), 12)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite")
    return out


def _non_negative(value: float, field: str) -> float:
    if value < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _valid_hash(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be str")
    if len(value) != 64:
        raise ValueError(f"{field} must be 64 hex characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be lowercase hex") from exc
    if value.lower() != value:
        raise ValueError(f"{field} must be lowercase hex")
    return value


def _stable_average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return _round12(sum(values) / len(values))


_SNAPSHOT_REQUIRED_KEYS: tuple[str, ...] = (
    "snapshot_id",
    "noise_level",
    "fidelity_score",
    "stability_score",
    "error_drift",
    "compensation_factor",
)


def _as_snapshot(obj: NoiseFidelitySnapshot | Mapping[str, Any]) -> NoiseFidelitySnapshot:
    if isinstance(obj, NoiseFidelitySnapshot):
        return obj
    if isinstance(obj, Mapping):
        missing = [k for k in _SNAPSHOT_REQUIRED_KEYS if k not in obj]
        if missing:
            raise ValueError(
                f"snapshot mapping is missing required fields: {missing}"
            )
        return normalize_noise_fidelity_inputs(
            snapshot_id=obj["snapshot_id"],
            noise_level=obj["noise_level"],
            fidelity_score=obj["fidelity_score"],
            stability_score=obj["stability_score"],
            error_drift=obj["error_drift"],
            compensation_factor=obj["compensation_factor"],
        )
    raise ValueError("snapshot must be NoiseFidelitySnapshot or mapping")


@dataclass(frozen=True)
class NoiseFidelitySnapshot:
    snapshot_id: str
    noise_level: float
    fidelity_score: float
    stability_score: float
    error_drift: float
    compensation_factor: float
    bounded: bool
    snapshot_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "noise_level": self.noise_level,
            "fidelity_score": self.fidelity_score,
            "stability_score": self.stability_score,
            "error_drift": self.error_drift,
            "compensation_factor": self.compensation_factor,
            "bounded": self.bounded,
            "snapshot_hash": self.snapshot_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class FidelityTimelinePoint:
    sequence_id: int
    fidelity_score: float
    stability_score: float
    noise_score: float
    point_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "fidelity_score": self.fidelity_score,
            "stability_score": self.stability_score,
            "noise_score": self.noise_score,
            "point_hash": self.point_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class FidelityStabilityTimeline:
    points: tuple[FidelityTimelinePoint, ...]
    average_fidelity: float
    average_stability: float
    timeline_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "average_fidelity": self.average_fidelity,
            "average_stability": self.average_stability,
            "timeline_hash": self.timeline_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PerturbationCompensationMetrics:
    compensation_effectiveness: float
    drift_reduction_score: float
    balance_score: float
    metrics_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "compensation_effectiveness": self.compensation_effectiveness,
            "drift_reduction_score": self.drift_reduction_score,
            "balance_score": self.balance_score,
            "metrics_hash": self.metrics_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class NoiseAuditEntry:
    sequence_id: int
    snapshot_hash: str
    parent_hash: str
    noise_score: float
    fidelity_score: float
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "snapshot_hash": self.snapshot_hash,
            "parent_hash": self.parent_hash,
            "noise_score": self.noise_score,
            "fidelity_score": self.fidelity_score,
            "entry_hash": self.entry_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class NoiseAuditTrail:
    entries: tuple[NoiseAuditEntry, ...]
    head_hash: str
    chain_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "head_hash": self.head_hash,
            "chain_valid": self.chain_valid,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class NoiseObservatoryReport:
    bounded_noise_score: float
    fidelity_health: str
    compensation_stable: bool
    deterministic: bool
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bounded_noise_score": self.bounded_noise_score,
            "fidelity_health": self.fidelity_health,
            "compensation_stable": self.compensation_stable,
            "deterministic": self.deterministic,
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def normalize_noise_fidelity_inputs(
    *,
    snapshot_id: str,
    noise_level: Any,
    fidelity_score: Any,
    stability_score: Any,
    error_drift: Any,
    compensation_factor: Any,
) -> NoiseFidelitySnapshot:
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ValueError("snapshot_id must be non-empty str")

    noise_raw = _non_negative(_finite_float(noise_level, "noise_level"), "noise_level")
    fidelity_raw = _non_negative(_finite_float(fidelity_score, "fidelity_score"), "fidelity_score")
    stability_raw = _non_negative(_finite_float(stability_score, "stability_score"), "stability_score")
    drift_raw = _non_negative(_finite_float(error_drift, "error_drift"), "error_drift")
    compensation_raw = _non_negative(
        _finite_float(compensation_factor, "compensation_factor"),
        "compensation_factor",
    )

    bounded = (
        noise_raw <= 1.0
        and fidelity_raw <= 1.0
        and stability_raw <= 1.0
        and drift_raw <= 1.0
        and compensation_raw <= 1.0
    )

    payload = {
        "snapshot_id": snapshot_id.strip(),
        "noise_level": _round12(_clamp01(noise_raw)),
        "fidelity_score": _round12(_clamp01(fidelity_raw)),
        "stability_score": _round12(_clamp01(stability_raw)),
        "error_drift": _round12(_clamp01(drift_raw)),
        "compensation_factor": _round12(_clamp01(compensation_raw)),
        "bounded": bounded,
    }
    snapshot_hash = _hash_sha256(payload)
    return NoiseFidelitySnapshot(**payload, snapshot_hash=snapshot_hash)


def compute_bounded_noise_score(
    *,
    noise_level: Any,
    error_drift: Any,
    stability_score: Any,
    compensation_factor: Any,
) -> float:
    noise = _clamp01(_finite_float(noise_level, "noise_level"))
    drift = _clamp01(_finite_float(error_drift, "error_drift"))
    stability = _clamp01(_finite_float(stability_score, "stability_score"))
    compensation = _clamp01(_finite_float(compensation_factor, "compensation_factor"))

    score = (
        0.4 * noise
        + 0.3 * drift
        + 0.2 * (1.0 - stability)
        + 0.1 * (1.0 - compensation)
    )
    return _round12(_clamp01(score))


def build_fidelity_stability_timeline(
    snapshots: Iterable[NoiseFidelitySnapshot | Mapping[str, Any]],
) -> FidelityStabilityTimeline:
    normalized = tuple(_as_snapshot(s) for s in snapshots)
    ordered = tuple(sorted(normalized, key=lambda s: (s.snapshot_id, s.snapshot_hash)))

    points: list[FidelityTimelinePoint] = []
    for sequence_id, snapshot in enumerate(ordered):
        noise_score = compute_bounded_noise_score(
            noise_level=snapshot.noise_level,
            error_drift=snapshot.error_drift,
            stability_score=snapshot.stability_score,
            compensation_factor=snapshot.compensation_factor,
        )
        point_payload = {
            "sequence_id": sequence_id,
            "fidelity_score": snapshot.fidelity_score,
            "stability_score": snapshot.stability_score,
            "noise_score": noise_score,
        }
        points.append(FidelityTimelinePoint(**point_payload, point_hash=_hash_sha256(point_payload)))

    fidelity_values = [p.fidelity_score for p in points]
    stability_values = [p.stability_score for p in points]
    avg_fidelity = _stable_average(fidelity_values)
    avg_stability = _stable_average(stability_values)
    timeline_payload = {
        "points": [p.to_dict() for p in points],
        "average_fidelity": avg_fidelity,
        "average_stability": avg_stability,
    }
    return FidelityStabilityTimeline(
        points=tuple(points),
        average_fidelity=avg_fidelity,
        average_stability=avg_stability,
        timeline_hash=_hash_sha256(timeline_payload),
    )


def compute_perturbation_compensation_metrics(
    *,
    noise_level: Any,
    error_drift: Any,
    compensation_factor: Any,
) -> PerturbationCompensationMetrics:
    noise = _clamp01(_finite_float(noise_level, "noise_level"))
    drift = _clamp01(_finite_float(error_drift, "error_drift"))
    compensation = _clamp01(_finite_float(compensation_factor, "compensation_factor"))

    compensation_effectiveness = _round12(_clamp01(compensation * (1.0 - noise)))
    drift_reduction_score = _round12(_clamp01(1.0 - drift))
    balance_score = _round12(_clamp01(1.0 - abs(noise - compensation)))

    payload = {
        "compensation_effectiveness": compensation_effectiveness,
        "drift_reduction_score": drift_reduction_score,
        "balance_score": balance_score,
    }
    return PerturbationCompensationMetrics(**payload, metrics_hash=_hash_sha256(payload))


def derive_noise_observatory_report(
    *,
    bounded_noise_score: Any,
    average_fidelity: Any,
    average_stability: Any,
    compensation_metrics: PerturbationCompensationMetrics,
) -> NoiseObservatoryReport:
    noise_score = _round12(_clamp01(_finite_float(bounded_noise_score, "bounded_noise_score")))
    fidelity = _clamp01(_finite_float(average_fidelity, "average_fidelity"))
    stability = _clamp01(_finite_float(average_stability, "average_stability"))

    if fidelity >= 0.85 and stability >= 0.85:
        health = "strong"
    elif fidelity >= 0.65 and stability >= 0.65:
        health = "stable"
    elif fidelity >= 0.4:
        health = "fragile"
    else:
        health = "critical"

    compensation_stable = (
        compensation_metrics.compensation_effectiveness >= 0.5
        and compensation_metrics.drift_reduction_score >= 0.5
    )

    payload = {
        "bounded_noise_score": noise_score,
        "fidelity_health": health,
        "compensation_stable": compensation_stable,
        "deterministic": True,
    }
    return NoiseObservatoryReport(**payload, report_hash=_hash_sha256(payload))


def empty_noise_audit_trail() -> NoiseAuditTrail:
    return NoiseAuditTrail(entries=(), head_hash="0" * 64, chain_valid=True)


def validate_noise_audit_trail(trail: NoiseAuditTrail) -> bool:
    if not isinstance(trail, NoiseAuditTrail):
        return False
    if trail.chain_valid is not True:
        return False

    try:
        _valid_hash(trail.head_hash, "head_hash")
    except ValueError:
        return False

    expected_parent = "0" * 64
    for expected_sequence_id, entry in enumerate(trail.entries):
        if not isinstance(entry, NoiseAuditEntry):
            return False
        if entry.sequence_id != expected_sequence_id:
            return False
        if entry.parent_hash != expected_parent:
            return False
        try:
            _valid_hash(entry.snapshot_hash, "snapshot_hash")
            _valid_hash(entry.parent_hash, "parent_hash")
            _valid_hash(entry.entry_hash, "entry_hash")
            _finite_float(entry.noise_score, "noise_score")
            _finite_float(entry.fidelity_score, "fidelity_score")
        except ValueError:
            return False
        expected_hash = _hash_sha256(
            {
                "sequence_id": entry.sequence_id,
                "snapshot_hash": entry.snapshot_hash,
                "parent_hash": entry.parent_hash,
                "noise_score": _round12(float(entry.noise_score)),
                "fidelity_score": _round12(float(entry.fidelity_score)),
            }
        )
        if entry.entry_hash != expected_hash:
            return False
        expected_parent = entry.entry_hash

    return trail.head_hash == expected_parent


def append_noise_audit_entry(
    trail: NoiseAuditTrail,
    *,
    snapshot_hash: str,
    noise_score: Any,
    fidelity_score: Any,
) -> NoiseAuditTrail:
    if not validate_noise_audit_trail(trail):
        raise ValueError("malformed or corrupted noise audit trail")
    snapshot_h = _valid_hash(snapshot_hash, "snapshot_hash")

    entry_payload = {
        "sequence_id": len(trail.entries),
        "snapshot_hash": snapshot_h,
        "parent_hash": trail.head_hash,
        "noise_score": _round12(_clamp01(_finite_float(noise_score, "noise_score"))),
        "fidelity_score": _round12(_clamp01(_finite_float(fidelity_score, "fidelity_score"))),
    }
    entry_hash = _hash_sha256(entry_payload)
    entry = NoiseAuditEntry(**entry_payload, entry_hash=entry_hash)
    updated = NoiseAuditTrail(
        entries=trail.entries + (entry,),
        head_hash=entry_hash,
        chain_valid=True,
    )
    if not validate_noise_audit_trail(updated):
        raise ValueError("failed to append deterministic audit entry")
    return updated


def run_quantum_noise_fidelity_observatory(
    snapshots: NoiseFidelitySnapshot | Mapping[str, Any] | Iterable[NoiseFidelitySnapshot | Mapping[str, Any]],
    prior_audit_trail: NoiseAuditTrail | None = None,
) -> tuple[
    FidelityStabilityTimeline,
    PerturbationCompensationMetrics,
    NoiseObservatoryReport,
    NoiseAuditTrail,
]:
    if isinstance(snapshots, (NoiseFidelitySnapshot, Mapping)):
        snapshot_items = (_as_snapshot(snapshots),)
    else:
        snapshot_items = tuple(_as_snapshot(s) for s in snapshots)
    timeline = build_fidelity_stability_timeline(snapshot_items)

    if timeline.points:
        avg_noise = _stable_average([p.noise_score for p in timeline.points])
        source = tuple(sorted(snapshot_items, key=lambda s: (s.snapshot_id, s.snapshot_hash)))
        avg_drift = _stable_average([s.error_drift for s in source])
        avg_compensation = _stable_average([s.compensation_factor for s in source])
    else:
        avg_noise = 0.0
        avg_drift = 0.0
        avg_compensation = 0.0

    metrics = compute_perturbation_compensation_metrics(
        noise_level=avg_noise,
        error_drift=avg_drift,
        compensation_factor=avg_compensation,
    )
    report = derive_noise_observatory_report(
        bounded_noise_score=avg_noise,
        average_fidelity=timeline.average_fidelity,
        average_stability=timeline.average_stability,
        compensation_metrics=metrics,
    )

    trail = prior_audit_trail if prior_audit_trail is not None else empty_noise_audit_trail()
    if not validate_noise_audit_trail(trail):
        raise ValueError("prior noise audit trail is invalid")

    updated = trail
    for s in tuple(sorted(snapshot_items, key=lambda x: (x.snapshot_id, x.snapshot_hash))):
        noise_score = compute_bounded_noise_score(
            noise_level=s.noise_level,
            error_drift=s.error_drift,
            stability_score=s.stability_score,
            compensation_factor=s.compensation_factor,
        )
        updated = append_noise_audit_entry(
            updated,
            snapshot_hash=s.snapshot_hash,
            noise_score=noise_score,
            fidelity_score=s.fidelity_score,
        )

    return timeline, metrics, report, updated


__all__ = [
    "NoiseFidelitySnapshot",
    "FidelityTimelinePoint",
    "FidelityStabilityTimeline",
    "PerturbationCompensationMetrics",
    "NoiseAuditEntry",
    "NoiseAuditTrail",
    "NoiseObservatoryReport",
    "normalize_noise_fidelity_inputs",
    "compute_bounded_noise_score",
    "build_fidelity_stability_timeline",
    "compute_perturbation_compensation_metrics",
    "derive_noise_observatory_report",
    "empty_noise_audit_trail",
    "append_noise_audit_entry",
    "validate_noise_audit_trail",
    "run_quantum_noise_fidelity_observatory",
]
