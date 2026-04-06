"""v137.9.1 — Spectral Reasoning Layer.

Deterministic Layer-4 spectral reasoning built from multimodal feature schema artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.multimodal_feature_schema import MultimodalFeatureSchemaResult

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SPECTRAL_REASONING_VERSION = 1
_BAND_NAMES: tuple[str, ...] = ("low_band", "mid_band", "high_band", "residual_band")

SPECTRAL_REASONING_LAYER_LAW = "SPECTRAL_REASONING_LAYER_LAW"
DETERMINISTIC_BAND_ORDERING_RULE = "DETERMINISTIC_BAND_ORDERING_RULE"
REPLAY_SAFE_SPECTRAL_IDENTITY_RULE = "REPLAY_SAFE_SPECTRAL_IDENTITY_RULE"
BOUNDED_SPECTRAL_SCORE_RULE = "BOUNDED_SPECTRAL_SCORE_RULE"


def _canonicalize_json(value: Any) -> _JSONValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    if isinstance(value, tuple):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, list):
        return tuple(_canonicalize_json(v) for v in value)
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(k, str) for k in keys):
            raise ValueError("payload keys must be strings")
        out: dict[str, _JSONValue] = {}
        for key in sorted(keys):
            out[key] = _canonicalize_json(value[key])
        return out
    raise ValueError(f"unsupported canonical payload type: {type(value)!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_bytes(value: Any) -> bytes:
    return _canonical_json(value).encode("utf-8")


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _clamp01(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("score must be finite")
    return min(1.0, max(0.0, score))


def _mean(values: tuple[float, ...], default: float = 1.0) -> float:
    if not values:
        return _clamp01(default)
    return _clamp01(float(sum(values) / len(values)))


def _validate_unit_interval(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite number in [0.0, 1.0]")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be finite number in [0.0, 1.0]")


def _validate_schema_artifact(schema_artifact: MultimodalFeatureSchemaResult) -> None:
    if not isinstance(schema_artifact, MultimodalFeatureSchemaResult):
        raise ValueError("schema_artifact must be a MultimodalFeatureSchemaResult")
    if schema_artifact.stable_hash() != schema_artifact.feature_schema_hash:
        raise ValueError("schema_artifact feature_schema_hash must match stable_hash")
    if schema_artifact.feature_count != len(schema_artifact.features):
        raise ValueError("schema_artifact feature_count must match len(features)")
    if schema_artifact.namespace_count != len(schema_artifact.namespaces):
        raise ValueError("schema_artifact namespace_count must match len(namespaces)")

    expected_order = tuple(sorted(schema_artifact.features, key=lambda f: (f.feature_index, f.feature_family, f.feature_name)))
    if schema_artifact.features != expected_order:
        raise ValueError("schema_artifact features must be in canonical deterministic order")

    expected_namespaces = tuple(sorted(schema_artifact.namespaces, key=lambda n: (n.namespace_index, n.feature_namespace)))
    if schema_artifact.namespaces != expected_namespaces:
        raise ValueError("schema_artifact namespaces must be in canonical deterministic order")


def _canonicalize_optional_signal_payload(
    signal_payload: tuple[float, ...] | tuple[int, ...] | tuple[complex, ...] | None,
) -> tuple[complex, ...]:
    if signal_payload is None:
        return ()
    if not isinstance(signal_payload, tuple):
        raise ValueError("signal_payload must be a tuple")
    normalized: list[complex] = []
    for value in signal_payload:
        if isinstance(value, bool):
            raise ValueError("signal_payload values must be numeric")
        if isinstance(value, int):
            normalized.append(complex(float(value), 0.0))
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("signal_payload float values must be finite")
            normalized.append(complex(float(value), 0.0))
        elif isinstance(value, complex):
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise ValueError("signal_payload complex values must be finite")
            normalized.append(complex(float(value.real), float(value.imag)))
        else:
            raise ValueError("signal_payload values must be int, float, or complex")

    return tuple(
        sorted(
            normalized,
            key=lambda z: (
                round(abs(z), 12),
                round(z.real, 12),
                round(z.imag, 12),
            ),
        )
    )


def _extract_schema_magnitudes(schema_artifact: MultimodalFeatureSchemaResult) -> tuple[float, ...]:
    magnitudes: list[float] = []
    for feature in schema_artifact.features:
        value = feature.feature_value
        if isinstance(value, bool):
            raise TypeError(
                f"schema feature bool values are not supported for magnitude extraction"
                f" (feature: {feature.feature_name!r})"
            )
        if isinstance(value, int):
            magnitudes.append(float(abs(value)))
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("schema feature float values must be finite")
            magnitudes.append(float(abs(value)))
        elif isinstance(value, str):
            magnitudes.append(float(len(value)))
        else:
            raise TypeError(
                f"unsupported schema feature_value type {type(value)!r} for magnitude extraction"
                f" (feature: {feature.feature_name!r})"
            )
    if not magnitudes:
        return (0.0,)
    return tuple(magnitudes)


def _normalize_magnitudes(magnitudes: tuple[float, ...]) -> tuple[float, ...]:
    for value in magnitudes:
        if not math.isfinite(value):
            raise ValueError("non-finite magnitudes are not allowed")
        if value < 0.0:
            raise ValueError("magnitudes must be non-negative")
    max_value = max(magnitudes) if magnitudes else 0.0
    if max_value <= 0.0:
        return tuple(0.0 for _ in magnitudes)
    return tuple(_clamp01(value / max_value) for value in magnitudes)


def _chunk_slices(length: int, chunks: int) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for idx in range(chunks):
        start = (idx * length) // chunks
        end = ((idx + 1) * length) // chunks
        out.append((start, end))
    return tuple(out)


@dataclass(frozen=True)
class SpectralFeature:
    feature_id: str
    source_feature_index: int
    source_feature_family: str
    source_feature_name: str
    source_feature_hash: str
    band_name: str
    band_index: int
    normalized_magnitude: float
    feature_projection_score: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "feature_id": self.feature_id,
            "source_feature_index": self.source_feature_index,
            "source_feature_family": self.source_feature_family,
            "source_feature_name": self.source_feature_name,
            "source_feature_hash": self.source_feature_hash,
            "band_name": self.band_name,
            "band_index": self.band_index,
            "normalized_magnitude": self.normalized_magnitude,
            "feature_projection_score": self.feature_projection_score,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("feature_id")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class SpectralBand:
    band_name: str
    band_index: int
    feature_count: int
    normalized_envelope_min: float
    normalized_envelope_max: float
    normalized_envelope_mean: float
    band_consistency_score: float
    features: tuple[SpectralFeature, ...]
    band_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "band_name": self.band_name,
            "band_index": self.band_index,
            "feature_count": self.feature_count,
            "normalized_envelope_min": self.normalized_envelope_min,
            "normalized_envelope_max": self.normalized_envelope_max,
            "normalized_envelope_mean": self.normalized_envelope_mean,
            "band_consistency_score": self.band_consistency_score,
            "features": tuple(feature.to_dict() for feature in self.features),
            "band_hash": self.band_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("band_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class SpectralReasoningResult:
    spectral_reasoning_version: int
    source_correspondence_hash: str
    source_feature_schema_hash: str
    spectral_feature_count: int
    spectral_band_count: int
    spectral_bands: tuple[SpectralBand, ...]
    spectral_coherence_score: float
    band_consistency_score: float
    normalization_integrity_score: float
    feature_projection_score: float
    overall_spectral_score: float
    law_invariants: tuple[str, ...]
    spectral_reasoning_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "spectral_reasoning_version": self.spectral_reasoning_version,
            "source_correspondence_hash": self.source_correspondence_hash,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "spectral_feature_count": self.spectral_feature_count,
            "spectral_band_count": self.spectral_band_count,
            "spectral_bands": tuple(band.to_dict() for band in self.spectral_bands),
            "spectral_coherence_score": self.spectral_coherence_score,
            "band_consistency_score": self.band_consistency_score,
            "normalization_integrity_score": self.normalization_integrity_score,
            "feature_projection_score": self.feature_projection_score,
            "overall_spectral_score": self.overall_spectral_score,
            "law_invariants": self.law_invariants,
            "spectral_reasoning_hash": self.spectral_reasoning_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("spectral_reasoning_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class SpectralReasoningReceipt:
    spectral_reasoning_version: int
    source_correspondence_hash: str
    source_feature_schema_hash: str
    spectral_reasoning_hash: str
    spectral_feature_count: int
    spectral_band_count: int
    overall_spectral_score: float
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "spectral_reasoning_version": self.spectral_reasoning_version,
            "source_correspondence_hash": self.source_correspondence_hash,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "spectral_reasoning_hash": self.spectral_reasoning_hash,
            "spectral_feature_count": self.spectral_feature_count,
            "spectral_band_count": self.spectral_band_count,
            "overall_spectral_score": self.overall_spectral_score,
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("receipt_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


def build_spectral_reasoning_layer(
    schema_artifact: MultimodalFeatureSchemaResult,
    *,
    signal_payload: tuple[float, ...] | tuple[int, ...] | tuple[complex, ...] | None = None,
) -> SpectralReasoningResult:
    _validate_schema_artifact(schema_artifact)
    # Optional payload is canonicalized and validated for deterministic readiness,
    # but spectral identity remains schema-derived per REPLAY_SAFE_SPECTRAL_IDENTITY_RULE.
    _canonicalize_optional_signal_payload(signal_payload)

    magnitudes = _extract_schema_magnitudes(schema_artifact)
    normalized = _normalize_magnitudes(magnitudes)

    slices = _chunk_slices(len(schema_artifact.features), len(_BAND_NAMES))
    projection_denominator = max(1, len(schema_artifact.features) - 1)

    spectral_bands: list[SpectralBand] = []
    all_features: list[SpectralFeature] = []
    for band_index, (band_name, (start, end)) in enumerate(zip(_BAND_NAMES, slices)):
        band_features: list[SpectralFeature] = []
        band_norms: list[float] = []
        for local_index, source_feature in enumerate(schema_artifact.features[start:end]):
            source_feature_index = start + local_index
            norm = normalized[source_feature_index]
            projection_score = _clamp01(1.0 - (source_feature_index / projection_denominator))
            spectral_feature = SpectralFeature(
                feature_id="",
                source_feature_index=source_feature_index,
                source_feature_family=source_feature.feature_family,
                source_feature_name=source_feature.feature_name,
                source_feature_hash=source_feature.stable_hash(),
                band_name=band_name,
                band_index=band_index,
                normalized_magnitude=norm,
                feature_projection_score=projection_score,
            )
            spectral_feature = replace(spectral_feature, feature_id=spectral_feature.stable_hash())
            band_features.append(spectral_feature)
            band_norms.append(norm)

        envelope_min = min(band_norms) if band_norms else 0.0
        envelope_max = max(band_norms) if band_norms else 0.0
        envelope_mean = _mean(tuple(band_norms), default=0.0)
        span = envelope_max - envelope_min
        band_consistency = _clamp01(1.0 - span)
        band = SpectralBand(
            band_name=band_name,
            band_index=band_index,
            feature_count=len(band_features),
            normalized_envelope_min=envelope_min,
            normalized_envelope_max=envelope_max,
            normalized_envelope_mean=envelope_mean,
            band_consistency_score=band_consistency,
            features=tuple(band_features),
            band_hash="",
        )
        band = replace(band, band_hash=band.stable_hash())
        spectral_bands.append(band)
        all_features.extend(band_features)

    expected_feature_order = tuple(range(len(schema_artifact.features)))
    realized_feature_order = tuple(feature.source_feature_index for feature in all_features)
    order_matches = sum(1 for a, b in zip(expected_feature_order, realized_feature_order) if a == b)
    spectral_coherence_score = _clamp01(float(order_matches) / len(expected_feature_order)) if expected_feature_order else 1.0

    band_consistency_score = _mean(tuple(band.band_consistency_score for band in spectral_bands), default=1.0)

    normalization_integrity_score = 1.0
    if normalized:
        if any(not 0.0 <= value <= 1.0 for value in normalized):
            raise ValueError("normalized magnitudes must be bounded in [0.0, 1.0]")
        if max(magnitudes) > 0.0:
            normalization_integrity_score = 1.0 if abs(max(normalized) - 1.0) <= 1e-12 else 0.0

    feature_projection_score = _mean(tuple(feature.feature_projection_score for feature in all_features), default=1.0)
    overall_spectral_score = _mean(
        (
            spectral_coherence_score,
            band_consistency_score,
            normalization_integrity_score,
            feature_projection_score,
        ),
        default=1.0,
    )

    for score_name, score in (
        ("spectral_coherence_score", spectral_coherence_score),
        ("band_consistency_score", band_consistency_score),
        ("normalization_integrity_score", normalization_integrity_score),
        ("feature_projection_score", feature_projection_score),
        ("overall_spectral_score", overall_spectral_score),
    ):
        _validate_unit_interval(score, score_name)

    artifact = SpectralReasoningResult(
        spectral_reasoning_version=_SPECTRAL_REASONING_VERSION,
        source_correspondence_hash=schema_artifact.source_correspondence_hash,
        source_feature_schema_hash=schema_artifact.feature_schema_hash,
        spectral_feature_count=len(all_features),
        spectral_band_count=len(spectral_bands),
        spectral_bands=tuple(spectral_bands),
        spectral_coherence_score=spectral_coherence_score,
        band_consistency_score=band_consistency_score,
        normalization_integrity_score=normalization_integrity_score,
        feature_projection_score=feature_projection_score,
        overall_spectral_score=overall_spectral_score,
        law_invariants=(
            SPECTRAL_REASONING_LAYER_LAW,
            DETERMINISTIC_BAND_ORDERING_RULE,
            REPLAY_SAFE_SPECTRAL_IDENTITY_RULE,
            BOUNDED_SPECTRAL_SCORE_RULE,
        ),
        spectral_reasoning_hash="",
    )
    return replace(artifact, spectral_reasoning_hash=artifact.stable_hash())


def export_spectral_reasoning_bytes(artifact: SpectralReasoningResult) -> bytes:
    if not isinstance(artifact, SpectralReasoningResult):
        raise ValueError("artifact must be a SpectralReasoningResult")
    return artifact.to_canonical_bytes()


def generate_spectral_reasoning_receipt(artifact: SpectralReasoningResult) -> SpectralReasoningReceipt:
    if not isinstance(artifact, SpectralReasoningResult):
        raise ValueError("artifact must be a SpectralReasoningResult")
    if artifact.stable_hash() != artifact.spectral_reasoning_hash:
        raise ValueError("artifact spectral_reasoning_hash must match stable_hash")
    receipt = SpectralReasoningReceipt(
        spectral_reasoning_version=artifact.spectral_reasoning_version,
        source_correspondence_hash=artifact.source_correspondence_hash,
        source_feature_schema_hash=artifact.source_feature_schema_hash,
        spectral_reasoning_hash=artifact.spectral_reasoning_hash,
        spectral_feature_count=artifact.spectral_feature_count,
        spectral_band_count=artifact.spectral_band_count,
        overall_spectral_score=artifact.overall_spectral_score,
        receipt_hash="",
    )
    return replace(receipt, receipt_hash=receipt.stable_hash())


__all__ = [
    "BOUNDED_SPECTRAL_SCORE_RULE",
    "DETERMINISTIC_BAND_ORDERING_RULE",
    "REPLAY_SAFE_SPECTRAL_IDENTITY_RULE",
    "SPECTRAL_REASONING_LAYER_LAW",
    "SpectralBand",
    "SpectralFeature",
    "SpectralReasoningReceipt",
    "SpectralReasoningResult",
    "build_spectral_reasoning_layer",
    "export_spectral_reasoning_bytes",
    "generate_spectral_reasoning_receipt",
]
