"""v137.9.5 — RF Equalization + Ground Station Compensation.

Deterministic Layer-4 consumer of satellite baseline artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.satellite_signal_baseline_and_orbital_noise import SatelliteBaselineResult

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_RF_EQUALIZATION_VERSION = 1
_GROUND_STATION_ORDER: tuple[str, ...] = (
    "urban_ground_station",
    "rural_ground_station",
    "desert_station",
    "maritime_station",
    "polar_station",
)
_COMPENSATION_SCENARIO_ORDER: tuple[str, ...] = (
    "nominal_station",
    "atmospheric_shift",
    "ground_reflection",
    "horizon_occlusion",
    "polar_noise",
)
_SCENARIO_SEVERITY: dict[str, float] = {
    "nominal_station": 0.00,
    "atmospheric_shift": 0.16,
    "ground_reflection": 0.24,
    "horizon_occlusion": 0.32,
    "polar_noise": 0.40,
}

RF_EQUALIZATION_LAYER_LAW = "RF_EQUALIZATION_LAYER_LAW"
DETERMINISTIC_RF_ORDERING_RULE = "DETERMINISTIC_RF_ORDERING_RULE"
REPLAY_SAFE_RF_IDENTITY_RULE = "REPLAY_SAFE_RF_IDENTITY_RULE"
BOUNDED_RF_SCORE_RULE = "BOUNDED_RF_SCORE_RULE"


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


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


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
        raise ValueError(f"{name} must be a numeric value")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be finite and in [0.0, 1.0]")


def _validate_satellite_artifact(baseline_artifact: SatelliteBaselineResult) -> None:
    if not isinstance(baseline_artifact, SatelliteBaselineResult):
        raise ValueError("baseline_artifact must be a SatelliteBaselineResult")
    if baseline_artifact.stable_hash() != baseline_artifact.satellite_baseline_hash:
        raise ValueError("baseline_artifact satellite_baseline_hash must match stable_hash")
    if baseline_artifact.segment_count != len(baseline_artifact.segments):
        raise ValueError("baseline_artifact segment_count must match len(segments)")
    if baseline_artifact.segment_count <= 0 or not baseline_artifact.segments:
        raise ValueError("baseline_artifact must contain at least one segment")
    if baseline_artifact.frame_count != sum(segment.frame_count for segment in baseline_artifact.segments):
        raise ValueError("baseline_artifact frame_count must match summed segment frame_count")

    expected_segment_order = tuple(
        sorted(
            baseline_artifact.segments,
            key=lambda s: (s.segment_index, s.orbital_scenario, s.segment_id),
        )
    )
    if baseline_artifact.segments != expected_segment_order:
        raise ValueError("baseline_artifact segments must be in canonical deterministic order")

    for segment in baseline_artifact.segments:
        if segment.segment_hash != segment.stable_hash():
            raise ValueError("baseline_artifact segment_hash must match stable_hash")
        if segment.frame_count <= 0 or not segment.frames:
            raise ValueError("baseline_artifact each segment must contain at least one frame")
        if segment.frame_count != len(segment.frames):
            raise ValueError("baseline_artifact frame_count must match len(frames) per segment")

        expected_frame_order = tuple(
            sorted(
                segment.frames,
                key=lambda frame: (frame.frame_index, frame.orbital_scenario, frame.frame_id),
            )
        )
        if segment.frames != expected_frame_order:
            raise ValueError("baseline_artifact frames must be in canonical deterministic order")
        for frame in segment.frames:
            if frame.frame_hash != frame.stable_hash():
                raise ValueError("baseline_artifact frame_hash must match stable_hash")
            if frame.frame_id != frame.frame_hash:
                raise ValueError("baseline_artifact frame_id must equal frame_hash")

    for score_name, score in (
        ("orbital_integrity_score", baseline_artifact.orbital_integrity_score),
        ("signal_latency_resilience_score", baseline_artifact.signal_latency_resilience_score),
        ("relay_handoff_score", baseline_artifact.relay_handoff_score),
        ("frame_consistency_score", baseline_artifact.frame_consistency_score),
        ("overall_satellite_score", baseline_artifact.overall_satellite_score),
    ):
        _validate_unit_interval(score, f"baseline_artifact {score_name}")


def _validate_optional_fixture_payload(
    float_fixture: tuple[float, ...] | None,
    int_fixture: tuple[int, ...] | None,
    str_fixture: tuple[str, ...] | None,
) -> None:
    if float_fixture is not None:
        if not isinstance(float_fixture, tuple):
            raise ValueError("float_fixture must be a tuple")
        for value in float_fixture:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("float_fixture values must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError("float_fixture values must be finite")

    if int_fixture is not None:
        if not isinstance(int_fixture, tuple):
            raise ValueError("int_fixture must be a tuple")
        for value in int_fixture:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("int_fixture values must be int")

    if str_fixture is not None:
        if not isinstance(str_fixture, tuple):
            raise ValueError("str_fixture must be a tuple")
        for value in str_fixture:
            if not isinstance(value, str):
                raise ValueError("str_fixture values must be str")


@dataclass(frozen=True)
class RFEqualizationFrame:
    frame_id: str
    frame_index: int
    ground_station_profile: str
    compensation_scenario: str
    segment_id: str
    source_satellite_segment_id: str
    equalization_curve_gain: float
    compensation_attenuation: float
    reflection_resilience_score: float
    frame_consistency_score: float
    frame_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "ground_station_profile": self.ground_station_profile,
            "compensation_scenario": self.compensation_scenario,
            "segment_id": self.segment_id,
            "source_satellite_segment_id": self.source_satellite_segment_id,
            "equalization_curve_gain": self.equalization_curve_gain,
            "compensation_attenuation": self.compensation_attenuation,
            "reflection_resilience_score": self.reflection_resilience_score,
            "frame_consistency_score": self.frame_consistency_score,
            "frame_hash": self.frame_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("frame_id")
        payload.pop("frame_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class GroundCompensationSegment:
    segment_id: str
    segment_index: int
    ground_station_profile: str
    compensation_scenario: str
    source_satellite_segment_id: str
    frame_count: int
    frames: tuple[RFEqualizationFrame, ...]
    equalization_integrity_score: float
    compensation_stability_score: float
    reflection_resilience_score: float
    frame_consistency_score: float
    overall_rf_score: float
    segment_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "segment_id": self.segment_id,
            "segment_index": self.segment_index,
            "ground_station_profile": self.ground_station_profile,
            "compensation_scenario": self.compensation_scenario,
            "source_satellite_segment_id": self.source_satellite_segment_id,
            "frame_count": self.frame_count,
            "frames": tuple(frame.to_dict() for frame in self.frames),
            "equalization_integrity_score": self.equalization_integrity_score,
            "compensation_stability_score": self.compensation_stability_score,
            "reflection_resilience_score": self.reflection_resilience_score,
            "frame_consistency_score": self.frame_consistency_score,
            "overall_rf_score": self.overall_rf_score,
            "segment_hash": self.segment_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("segment_id")
        payload.pop("segment_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class RFEqualizationResult:
    rf_equalization_version: int
    source_feature_schema_hash: str
    source_spectral_reasoning_hash: str
    source_copper_channel_battery_hash: str
    source_telecom_recovery_hash: str
    source_satellite_baseline_hash: str
    rf_equalization_id: str
    segment_count: int
    frame_count: int
    segments: tuple[GroundCompensationSegment, ...]
    equalization_integrity_score: float
    compensation_stability_score: float
    reflection_resilience_score: float
    frame_consistency_score: float
    overall_rf_score: float
    law_invariants: tuple[str, ...]
    rf_equalization_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "rf_equalization_version": self.rf_equalization_version,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "source_spectral_reasoning_hash": self.source_spectral_reasoning_hash,
            "source_copper_channel_battery_hash": self.source_copper_channel_battery_hash,
            "source_telecom_recovery_hash": self.source_telecom_recovery_hash,
            "source_satellite_baseline_hash": self.source_satellite_baseline_hash,
            "rf_equalization_id": self.rf_equalization_id,
            "segment_count": self.segment_count,
            "frame_count": self.frame_count,
            "segments": tuple(segment.to_dict() for segment in self.segments),
            "equalization_integrity_score": self.equalization_integrity_score,
            "compensation_stability_score": self.compensation_stability_score,
            "reflection_resilience_score": self.reflection_resilience_score,
            "frame_consistency_score": self.frame_consistency_score,
            "overall_rf_score": self.overall_rf_score,
            "law_invariants": self.law_invariants,
            "rf_equalization_hash": self.rf_equalization_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("rf_equalization_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class RFEqualizationReceipt:
    rf_equalization_version: int
    source_feature_schema_hash: str
    source_spectral_reasoning_hash: str
    source_copper_channel_battery_hash: str
    source_telecom_recovery_hash: str
    source_satellite_baseline_hash: str
    rf_equalization_id: str
    rf_equalization_hash: str
    segment_count: int
    frame_count: int
    overall_rf_score: float
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "rf_equalization_version": self.rf_equalization_version,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "source_spectral_reasoning_hash": self.source_spectral_reasoning_hash,
            "source_copper_channel_battery_hash": self.source_copper_channel_battery_hash,
            "source_telecom_recovery_hash": self.source_telecom_recovery_hash,
            "source_satellite_baseline_hash": self.source_satellite_baseline_hash,
            "rf_equalization_id": self.rf_equalization_id,
            "rf_equalization_hash": self.rf_equalization_hash,
            "segment_count": self.segment_count,
            "frame_count": self.frame_count,
            "overall_rf_score": self.overall_rf_score,
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


def run_rf_equalization(
    baseline_artifact: SatelliteBaselineResult,
    *,
    float_fixture: tuple[float, ...] | None = None,
    int_fixture: tuple[int, ...] | None = None,
    str_fixture: tuple[str, ...] | None = None,
) -> RFEqualizationResult:
    """Build replay-safe RF equalization artifacts from satellite baseline lineage.

    Optional fixture tuples are validated for deterministic-readiness only and do
    not alter the generated artifact or derived hashes.
    """

    _validate_satellite_artifact(baseline_artifact)
    _validate_optional_fixture_payload(float_fixture, int_fixture, str_fixture)

    if len(_GROUND_STATION_ORDER) != len(_COMPENSATION_SCENARIO_ORDER):
        raise ValueError("ground station/scenario configuration mismatch")

    rf_equalization_id = _sha256_hex(
        {
            "source_satellite_baseline_hash": baseline_artifact.satellite_baseline_hash,
            "rf_equalization_version": _RF_EQUALIZATION_VERSION,
        }
    )

    satellite_segments = baseline_artifact.segments
    satellite_segment_count = len(satellite_segments)
    segments: list[GroundCompensationSegment] = []

    for segment_index, (profile, scenario) in enumerate(zip(_GROUND_STATION_ORDER, _COMPENSATION_SCENARIO_ORDER)):
        source_segment = satellite_segments[segment_index % satellite_segment_count]
        severity = _SCENARIO_SEVERITY[scenario]

        segment_id = _sha256_hex(
            {
                "segment_index": segment_index,
                "ground_station_profile": profile,
                "compensation_scenario": scenario,
                "source_satellite_segment_id": source_segment.segment_id,
                "source_satellite_baseline_hash": baseline_artifact.satellite_baseline_hash,
            }
        )

        frame_total = max(1, source_segment.frame_count)
        frames: list[RFEqualizationFrame] = []
        for frame_index in range(frame_total):
            source_frame = source_segment.frames[frame_index % len(source_segment.frames)]
            source_deficit = 1.0 - source_frame.frame_consistency_score

            equalization_curve_gain = _clamp01((0.025 * (frame_index + 1)) + (severity * source_deficit))
            compensation_attenuation = _clamp01((0.02 * (segment_index + 1)) + (severity * (1.0 - source_segment.signal_latency_resilience_score)))
            reflection_resilience_score = _clamp01(
                1.0 - ((equalization_curve_gain * 0.50) + (compensation_attenuation * 0.50))
            )
            frame_consistency_score = _clamp01(
                1.0 - ((equalization_curve_gain * 0.45) + (compensation_attenuation * 0.35) + (severity * source_deficit * 0.20))
            )

            frame = RFEqualizationFrame(
                frame_id="",
                frame_index=frame_index,
                ground_station_profile=profile,
                compensation_scenario=scenario,
                segment_id=segment_id,
                source_satellite_segment_id=source_segment.segment_id,
                equalization_curve_gain=equalization_curve_gain,
                compensation_attenuation=compensation_attenuation,
                reflection_resilience_score=reflection_resilience_score,
                frame_consistency_score=frame_consistency_score,
                frame_hash="",
            )
            frame_hash = frame.stable_hash()
            frames.append(replace(frame, frame_id=frame_hash, frame_hash=frame_hash))

        baseline_deficit = 1.0 - source_segment.overall_satellite_score
        frame_consistency_score = _mean(tuple(frame.frame_consistency_score for frame in frames), default=1.0)
        reflection_resilience_score = _mean(tuple(frame.reflection_resilience_score for frame in frames), default=1.0)
        equalization_integrity_score = _clamp01(
            1.0 - ((severity * 0.65) * baseline_deficit)
        )
        compensation_stability_score = _clamp01(
            1.0 - ((severity * 0.75) * (1.0 - source_segment.orbital_integrity_score))
        )
        overall_rf_score = _mean(
            (
                equalization_integrity_score,
                compensation_stability_score,
                reflection_resilience_score,
                frame_consistency_score,
            ),
            default=1.0,
        )

        segment = GroundCompensationSegment(
            segment_id=segment_id,
            segment_index=segment_index,
            ground_station_profile=profile,
            compensation_scenario=scenario,
            source_satellite_segment_id=source_segment.segment_id,
            frame_count=len(frames),
            frames=tuple(frames),
            equalization_integrity_score=equalization_integrity_score,
            compensation_stability_score=compensation_stability_score,
            reflection_resilience_score=reflection_resilience_score,
            frame_consistency_score=frame_consistency_score,
            overall_rf_score=overall_rf_score,
            segment_hash="",
        )
        segments.append(replace(segment, segment_hash=segment.stable_hash()))

    equalization_integrity_score = _mean(tuple(segment.equalization_integrity_score for segment in segments), default=1.0)
    compensation_stability_score = _mean(tuple(segment.compensation_stability_score for segment in segments), default=1.0)
    reflection_resilience_score = _mean(tuple(segment.reflection_resilience_score for segment in segments), default=1.0)
    frame_consistency_score = _mean(tuple(segment.frame_consistency_score for segment in segments), default=1.0)
    overall_rf_score = _mean(tuple(segment.overall_rf_score for segment in segments), default=1.0)

    for score_name, score in (
        ("equalization_integrity_score", equalization_integrity_score),
        ("compensation_stability_score", compensation_stability_score),
        ("reflection_resilience_score", reflection_resilience_score),
        ("frame_consistency_score", frame_consistency_score),
        ("overall_rf_score", overall_rf_score),
    ):
        _validate_unit_interval(score, score_name)

    artifact = RFEqualizationResult(
        rf_equalization_version=_RF_EQUALIZATION_VERSION,
        source_feature_schema_hash=baseline_artifact.source_feature_schema_hash,
        source_spectral_reasoning_hash=baseline_artifact.source_spectral_reasoning_hash,
        source_copper_channel_battery_hash=baseline_artifact.source_copper_channel_battery_hash,
        source_telecom_recovery_hash=baseline_artifact.source_telecom_recovery_hash,
        source_satellite_baseline_hash=baseline_artifact.satellite_baseline_hash,
        rf_equalization_id=rf_equalization_id,
        segment_count=len(segments),
        frame_count=sum(segment.frame_count for segment in segments),
        segments=tuple(segments),
        equalization_integrity_score=equalization_integrity_score,
        compensation_stability_score=compensation_stability_score,
        reflection_resilience_score=reflection_resilience_score,
        frame_consistency_score=frame_consistency_score,
        overall_rf_score=overall_rf_score,
        law_invariants=(
            RF_EQUALIZATION_LAYER_LAW,
            DETERMINISTIC_RF_ORDERING_RULE,
            REPLAY_SAFE_RF_IDENTITY_RULE,
            BOUNDED_RF_SCORE_RULE,
        ),
        rf_equalization_hash="",
    )
    return replace(artifact, rf_equalization_hash=artifact.stable_hash())


def export_rf_equalization_bytes(artifact: RFEqualizationResult) -> bytes:
    if not isinstance(artifact, RFEqualizationResult):
        raise ValueError("artifact must be a RFEqualizationResult")
    return artifact.to_canonical_bytes()


def generate_rf_equalization_receipt(artifact: RFEqualizationResult) -> RFEqualizationReceipt:
    if not isinstance(artifact, RFEqualizationResult):
        raise ValueError("artifact must be a RFEqualizationResult")
    if artifact.stable_hash() != artifact.rf_equalization_hash:
        raise ValueError("artifact rf_equalization_hash must match stable_hash")

    receipt = RFEqualizationReceipt(
        rf_equalization_version=artifact.rf_equalization_version,
        source_feature_schema_hash=artifact.source_feature_schema_hash,
        source_spectral_reasoning_hash=artifact.source_spectral_reasoning_hash,
        source_copper_channel_battery_hash=artifact.source_copper_channel_battery_hash,
        source_telecom_recovery_hash=artifact.source_telecom_recovery_hash,
        source_satellite_baseline_hash=artifact.source_satellite_baseline_hash,
        rf_equalization_id=artifact.rf_equalization_id,
        rf_equalization_hash=artifact.rf_equalization_hash,
        segment_count=artifact.segment_count,
        frame_count=artifact.frame_count,
        overall_rf_score=artifact.overall_rf_score,
        receipt_hash="",
    )
    return replace(receipt, receipt_hash=receipt.stable_hash())


__all__ = [
    "BOUNDED_RF_SCORE_RULE",
    "DETERMINISTIC_RF_ORDERING_RULE",
    "REPLAY_SAFE_RF_IDENTITY_RULE",
    "RF_EQUALIZATION_LAYER_LAW",
    "GroundCompensationSegment",
    "RFEqualizationFrame",
    "RFEqualizationReceipt",
    "RFEqualizationResult",
    "export_rf_equalization_bytes",
    "generate_rf_equalization_receipt",
    "run_rf_equalization",
]
