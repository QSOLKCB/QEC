"""v138.5.3 — Fractal Field Invariant Mapper.

Deterministic additive analysis module for multi-scale invariant motif mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

_RELEASE_VERSION = "v138.5.3"
_EXPERIMENT_KIND = "fractal_field_invariant_mapper"
_SCALE_SIZES: tuple[int, ...] = (2, 4, 8)
_INVARIANT_EPSILON = 0.05


_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]
_Token = int | str
_Motif = tuple[_Token, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def _motif_signature(motif: _Motif) -> str:
    """Return a deterministic, collision-safe signature for a motif."""
    if not motif:
        raise ValueError("motif must be non-empty")
    canonical = _primitive_motif(motif)
    return _canonical_json(canonical)


def _primitive_motif(motif: _Motif) -> _Motif:
    length = len(motif)
    for period in range(1, length + 1):
        if length % period != 0:
            continue
        candidate = motif[:period]
        if candidate * (length // period) == motif:
            return candidate
    return motif


def _canonicalize_token(value: Any) -> _Token:
    if isinstance(value, bool):
        raise ValueError("trajectory tokens must not be bool")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value == "":
            raise ValueError("trajectory string tokens must be non-empty")
        return value
    raise ValueError("trajectory tokens must be int or str")


def _canonicalize_trajectory(trajectory: Sequence[Any]) -> tuple[_Token, ...]:
    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes, bytearray)):
        raise ValueError("trajectory must be a sequence of int/str tokens")
    normalized = tuple(_canonicalize_token(item) for item in trajectory)
    if not normalized:
        raise ValueError("trajectory must be non-empty")
    return normalized


def _canonicalize_motif(motif: _Motif) -> _Motif:
    if not isinstance(motif, tuple) or not motif:
        raise ValueError("motifs must be non-empty tuples")
    first_type = type(motif[0])
    if first_type not in (int, str):
        raise ValueError("motif entries must be int or str")
    for item in motif:
        if type(item) is not first_type:
            raise ValueError("motif entries must have homogeneous canonical type")
        if isinstance(item, str) and not item:
            raise ValueError("trajectory string tokens must be non-empty")
    return motif


@dataclass(frozen=True)
class _ScaleProfileInternal:
    scale_size: int
    window_count: int
    dominant_motif: _Motif
    motif_recurrence_ratio: float
    motif_diversity: float
    scale_stability_score: float
    motif_counts: tuple[tuple[_Motif, int], ...]


@dataclass(frozen=True)
class _MotifSignatureInternal:
    motif: _Motif
    signature: str
    scale_occurrences: tuple[tuple[int, float], ...]
    mean_recurrence: float
    recurrence_span: float


@dataclass(frozen=True)
class FractalFieldInvariantReceipt:
    release_version: str
    experiment_kind: str
    trajectory_length: int
    ordered_scale_profiles: tuple[dict[str, _JSONValue], ...]
    ordered_invariant_motifs: tuple[dict[str, _JSONValue], ...]
    classification: str
    recommendation: str
    decision: dict[str, _JSONValue]
    bounded_metric_bundle: dict[str, float]
    advisory_only: bool
    decoder_core_modified: bool
    input_content_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "release_version": self.release_version,
            "experiment_kind": self.experiment_kind,
            "trajectory_length": self.trajectory_length,
            "ordered_scale_profiles": self.ordered_scale_profiles,
            "ordered_invariant_motifs": self.ordered_invariant_motifs,
            "classification": self.classification,
            "recommendation": self.recommendation,
            "decision": self.decision,
            "bounded_metric_bundle": self.bounded_metric_bundle,
            "advisory_only": self.advisory_only,
            "decoder_core_modified": self.decoder_core_modified,
            "input_content_hash": self.input_content_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_canonical_bytes())


def _precompute_scale_profiles(trajectory: tuple[_Token, ...]) -> tuple[_ScaleProfileInternal, ...]:
    profiles: list[_ScaleProfileInternal] = []
    n = len(trajectory)

    for scale_size in _SCALE_SIZES:
        if n < scale_size:
            continue
        windows = tuple(
            _canonicalize_motif(tuple(trajectory[idx : idx + scale_size]))
            for idx in range(0, n - scale_size + 1)
        )
        window_count = len(windows)
        if window_count <= 0:
            raise ValueError("inconsistent scale decomposition")

        counts_map: dict[_Motif, int] = {}
        first_index: dict[_Motif, int] = {}
        for idx, motif in enumerate(windows):
            if motif not in counts_map:
                counts_map[motif] = 0
                first_index[motif] = idx
            counts_map[motif] += 1

        sig_cache: dict[_Motif, str] = {motif: _motif_signature(motif) for motif in counts_map}
        motif_counts = tuple(
            sorted(
                ((motif, count) for motif, count in counts_map.items()),
                key=lambda item: (-item[1], sig_cache[item[0]], first_index[item[0]]),
            )
        )
        dominant_motif = motif_counts[0][0]
        dominant_count = motif_counts[0][1]
        recurrence_ratio = dominant_count / float(window_count)
        diversity = len(motif_counts) / float(window_count)
        stability = 1.0 - diversity
        if motif_counts:
            stability = (stability + recurrence_ratio) / 2.0

        expected_window_count = n - scale_size + 1
        if expected_window_count != window_count:
            raise ValueError("inconsistent scale decomposition")

        profiles.append(
            _ScaleProfileInternal(
                scale_size=scale_size,
                window_count=window_count,
                dominant_motif=dominant_motif,
                motif_recurrence_ratio=_clamp01(recurrence_ratio),
                motif_diversity=_clamp01(diversity),
                scale_stability_score=_clamp01(stability),
                motif_counts=motif_counts,
            )
        )

    return tuple(profiles)


def _precompute_motif_signatures(
    scale_profiles: tuple[_ScaleProfileInternal, ...],
) -> tuple[_MotifSignatureInternal, ...]:
    signature_scales: dict[str, list[tuple[int, float]]] = {}
    signature_motifs: dict[str, list[_Motif]] = {}

    for profile in scale_profiles:
        for motif, count in profile.motif_counts:
            ratio = count / float(profile.window_count)
            signature = _motif_signature(motif)
            if signature not in signature_scales:
                signature_scales[signature] = []
                signature_motifs[signature] = []
            signature_scales[signature].append((profile.scale_size, ratio))
            signature_motifs[signature].append(motif)

    signatures: list[_MotifSignatureInternal] = []
    for signature, occurrences in signature_scales.items():
        ordered_occurrences = tuple(sorted(occurrences, key=lambda item: item[0]))
        representative_motif = sorted(
            signature_motifs[signature],
            key=lambda m: (len(m), m),
        )[0]
        ratios = tuple(item[1] for item in ordered_occurrences)
        signatures.append(
            _MotifSignatureInternal(
                motif=representative_motif,
                signature=signature,
                scale_occurrences=ordered_occurrences,
                mean_recurrence=sum(ratios) / float(len(ratios)),
                recurrence_span=max(ratios) - min(ratios),
            )
        )

    return tuple(
        sorted(
            signatures,
            key=lambda item: (
                -len(item.scale_occurrences),
                item.recurrence_span,
                -item.mean_recurrence,
                item.signature,
            ),
        )
    )


def _build_metrics(
    scale_profiles: tuple[_ScaleProfileInternal, ...],
    motif_signatures: tuple[_MotifSignatureInternal, ...],
    invariant_signatures: tuple[_MotifSignatureInternal, ...],
) -> dict[str, float]:
    valid_scales = len(scale_profiles)
    if valid_scales == 0:
        raise ValueError("trajectory does not support any valid scales")

    unique_motifs = max(1, len(motif_signatures))
    cross_scale_invariance_score = len(invariant_signatures) / float(unique_motifs)

    if invariant_signatures:
        motif_recurrence_score = sum(sig.mean_recurrence for sig in invariant_signatures) / float(
            len(invariant_signatures)
        )
    else:
        motif_recurrence_score = 0.0

    scale_stability_score = sum(p.scale_stability_score for p in scale_profiles) / float(valid_scales)

    aggregate_counts: dict[_Motif, int] = {}
    total_count = 0
    for profile in scale_profiles:
        for motif, count in profile.motif_counts:
            aggregate_counts[motif] = aggregate_counts.get(motif, 0) + count
            total_count += count

    if total_count <= 0:
        raise ValueError("inconsistent scale decomposition")

    concentration = 0.0
    for count in aggregate_counts.values():
        p = count / float(total_count)
        concentration += p * p
    max_concentration = 1.0
    min_concentration = 1.0 / float(max(1, len(aggregate_counts)))
    normalized_concentration = 1.0
    if max_concentration > min_concentration:
        normalized_concentration = (concentration - min_concentration) / (max_concentration - min_concentration)
    fragmentation_penalty = (len(aggregate_counts) - 1) / float(max(1, total_count - 1))
    fractal_concentration_score = normalized_concentration * (1.0 - _clamp01(fragmentation_penalty))

    if invariant_signatures:
        source_agreement_score = sum(
            len(sig.scale_occurrences) / float(valid_scales) for sig in invariant_signatures
        ) / float(len(invariant_signatures))
    else:
        source_agreement_score = 0.0

    bounded_invariant_confidence = (
        0.30 * _clamp01(cross_scale_invariance_score)
        + 0.20 * _clamp01(motif_recurrence_score)
        + 0.20 * _clamp01(scale_stability_score)
        + 0.15 * _clamp01(fractal_concentration_score)
        + 0.15 * _clamp01(source_agreement_score)
    )

    return {
        "cross_scale_invariance_score": _clamp01(cross_scale_invariance_score),
        "motif_recurrence_score": _clamp01(motif_recurrence_score),
        "scale_stability_score": _clamp01(scale_stability_score),
        "fractal_concentration_score": _clamp01(fractal_concentration_score),
        "source_agreement_score": _clamp01(source_agreement_score),
        "bounded_invariant_confidence": _clamp01(bounded_invariant_confidence),
    }


def _classify(metrics: Mapping[str, float], invariant_count: int) -> str:
    confidence = metrics["bounded_invariant_confidence"]
    concentration = metrics["fractal_concentration_score"]
    agreement = metrics["source_agreement_score"]
    cross = metrics["cross_scale_invariance_score"]

    if invariant_count == 0:
        if concentration < 0.25:
            return "dispersed_multiscale_field"
        return "weak_invariant_structure"
    if confidence >= 0.75 and concentration >= 0.70:
        return "stable_recursive_field"
    if concentration >= 0.65 and cross < 0.60:
        return "localized_invariant_cluster"
    if agreement >= 0.60 and cross >= 0.50:
        return "cross_scale_balanced_field"
    if confidence < 0.40:
        return "weak_invariant_structure"
    return "cross_scale_balanced_field"


def _recommend(classification: str) -> str:
    if classification == "stable_recursive_field":
        return "mapping_supports_recursive_structure"
    if classification == "localized_invariant_cluster":
        return "mapping_supports_localized_invariants"
    if classification == "cross_scale_balanced_field":
        return "mapping_supports_cross_scale_balance"
    return "mapping_is_weakly_informative"


def _validate_structural_invariants(
    scale_profiles: tuple[_ScaleProfileInternal, ...],
    invariant_signatures: tuple[_MotifSignatureInternal, ...],
    metrics: Mapping[str, float],
    classification: str,
) -> None:
    sizes = tuple(profile.scale_size for profile in scale_profiles)
    if tuple(sorted(sizes)) != sizes or len(set(sizes)) != len(sizes):
        raise ValueError("scale_profiles must be strictly ordered")

    allowed_empty = {"weak_invariant_structure", "dispersed_multiscale_field"}
    if not invariant_signatures and classification not in allowed_empty:
        raise ValueError("invariant motifs required for non-weak classifications")

    for name, value in metrics.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0,1]")

    scale_map = {profile.scale_size: profile for profile in scale_profiles}
    scale_sig_map: dict[int, set[str]] = {
        profile.scale_size: {_motif_signature(motif) for motif, _count in profile.motif_counts}
        for profile in scale_profiles
    }
    for signature in invariant_signatures:
        for scale_size, _ratio in signature.scale_occurrences:
            if scale_size not in scale_map:
                raise ValueError("motif references invalid scale")
            if signature.signature not in scale_sig_map[scale_size]:
                raise ValueError("motif references invalid profile content")


def map_fractal_field_invariants(trajectory: Sequence[Any]) -> FractalFieldInvariantReceipt:
    canonical_trajectory = _canonicalize_trajectory(trajectory)
    input_bytes = _canonical_json({"trajectory": canonical_trajectory}).encode("utf-8")
    input_content_hash = _sha256_hex(input_bytes)

    scale_profiles = _precompute_scale_profiles(canonical_trajectory)
    motif_signatures = _precompute_motif_signatures(scale_profiles)

    invariant_signatures = tuple(
        sig
        for sig in motif_signatures
        if len(sig.scale_occurrences) >= 2 and sig.recurrence_span <= _INVARIANT_EPSILON
    )

    valid_scale_count = len(scale_profiles)
    ordered_invariant_motifs: tuple[dict[str, _JSONValue], ...] = tuple(
        {
            "motif": sig.motif,
            "canonical_signature": sig.signature,
            "scale_sizes": tuple(scale_size for scale_size, _ratio in sig.scale_occurrences),
            "scale_recurrence_ratios": tuple(_clamp01(ratio) for _scale_size, ratio in sig.scale_occurrences),
            "invariance_score": _clamp01(
                sig.mean_recurrence * (len(sig.scale_occurrences) / float(max(1, valid_scale_count)))
            ),
        }
        for sig in sorted(
            invariant_signatures,
            key=lambda item: (
                -item.mean_recurrence * (len(item.scale_occurrences) / float(max(1, valid_scale_count))),
                item.signature,
            ),
        )
    )

    metrics = _build_metrics(scale_profiles, motif_signatures, invariant_signatures)
    classification = _classify(metrics, len(ordered_invariant_motifs))
    recommendation = _recommend(classification)

    dominant_invariant_motif: _Motif | None = None
    if ordered_invariant_motifs:
        dominant_invariant_motif = ordered_invariant_motifs[0]["motif"]  # type: ignore[assignment]

    ordered_scale_profiles: tuple[dict[str, _JSONValue], ...] = tuple(
        {
            "scale_size": profile.scale_size,
            "window_count": profile.window_count,
            "dominant_motif": profile.dominant_motif,
            "motif_recurrence_ratio": profile.motif_recurrence_ratio,
            "motif_diversity": profile.motif_diversity,
            "scale_stability_score": profile.scale_stability_score,
        }
        for profile in scale_profiles
    )

    decision: dict[str, _JSONValue] = {
        "classification": classification,
        "dominant_invariant_motif": dominant_invariant_motif,
        "invariant_motif_count": len(ordered_invariant_motifs),
        "cross_scale_persistence_summary": {
            "valid_scale_count": valid_scale_count,
            "invariant_coverage_ratio": _clamp01(
                len(ordered_invariant_motifs) / float(max(1, len(motif_signatures)))
            ),
            "max_invariant_scale_span": max(
                (len(item["scale_sizes"]) for item in ordered_invariant_motifs),
                default=0,
            ),
        },
        "recommendation": recommendation,
    }

    _validate_structural_invariants(scale_profiles, invariant_signatures, metrics, classification)

    return FractalFieldInvariantReceipt(
        release_version=_RELEASE_VERSION,
        experiment_kind=_EXPERIMENT_KIND,
        trajectory_length=len(canonical_trajectory),
        ordered_scale_profiles=ordered_scale_profiles,
        ordered_invariant_motifs=ordered_invariant_motifs,
        classification=classification,
        recommendation=recommendation,
        decision=decision,
        bounded_metric_bundle=dict(metrics),
        advisory_only=True,
        decoder_core_modified=False,
        input_content_hash=input_content_hash,
    )
