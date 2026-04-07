"""v137.9.4 — Satellite Signal Baseline + Orbital Noise Envelope.

Deterministic Layer-4 consumer of telecom line recovery artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.telecom_line_recovery_and_sync import TelecomRecoveryResult

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SATELLITE_BASELINE_VERSION = 1
_ORBITAL_SCENARIO_ORDER: tuple[str, ...] = (
    "nominal_orbit",
    "solar_noise",
    "eclipse_shadow",
    "relay_handoff",
    "deep_space_latency",
)
_ORBITAL_CLASS_ORDER: tuple[str, ...] = (
    "leo",
    "meo",
    "geo",
    "relay",
    "deep_space",
)
_SCENARIO_TO_CLASS: dict[str, str] = {
    "nominal_orbit": "leo",
    "solar_noise": "meo",
    "eclipse_shadow": "geo",
    "relay_handoff": "relay",
    "deep_space_latency": "deep_space",
}
_SCENARIO_SEVERITY: dict[str, float] = {
    "nominal_orbit": 0.00,
    "solar_noise": 0.18,
    "eclipse_shadow": 0.30,
    "relay_handoff": 0.24,
    "deep_space_latency": 0.38,
}

SATELLITE_BASELINE_LAYER_LAW = "SATELLITE_BASELINE_LAYER_LAW"
DETERMINISTIC_ORBITAL_ORDERING_RULE = "DETERMINISTIC_ORBITAL_ORDERING_RULE"
REPLAY_SAFE_SATELLITE_IDENTITY_RULE = "REPLAY_SAFE_SATELLITE_IDENTITY_RULE"
BOUNDED_SATELLITE_SCORE_RULE = "BOUNDED_SATELLITE_SCORE_RULE"


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
        raise ValueError(f"{name} must be finite number in [0.0, 1.0]")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be finite number in [0.0, 1.0]")


def _validate_telecom_artifact(recovery_artifact: TelecomRecoveryResult) -> None:
    if not isinstance(recovery_artifact, TelecomRecoveryResult):
        raise ValueError("recovery_artifact must be a TelecomRecoveryResult")
    if recovery_artifact.stable_hash() != recovery_artifact.telecom_recovery_hash:
        raise ValueError("recovery_artifact telecom_recovery_hash must match stable_hash")
    if recovery_artifact.segment_count != len(recovery_artifact.segments):
        raise ValueError("recovery_artifact segment_count must match len(segments)")
    if recovery_artifact.frame_count != sum(segment.frame_count for segment in recovery_artifact.segments):
        raise ValueError("recovery_artifact frame_count must match summed segment frame_count")

    expected_segment_order = tuple(
        sorted(recovery_artifact.segments, key=lambda s: (s.segment_index, s.recovery_mode, s.segment_id))
    )
    if recovery_artifact.segments != expected_segment_order:
        raise ValueError("recovery_artifact segments must be in canonical deterministic order")

    for segment in recovery_artifact.segments:
        if segment.segment_hash != segment.stable_hash():
            raise ValueError("recovery_artifact segment_hash must match stable_hash")
        if segment.frame_count != len(segment.frames):
            raise ValueError("recovery_artifact frame_count must match len(frames) per segment")

        expected_frame_order = tuple(
            sorted(segment.frames, key=lambda frame: (frame.frame_index, frame.recovery_mode, frame.frame_id))
        )
        if segment.frames != expected_frame_order:
            raise ValueError("recovery_artifact frames must be in canonical deterministic order")
        for frame in segment.frames:
            if frame.frame_hash != frame.stable_hash():
                raise ValueError("recovery_artifact frame_hash must match stable_hash")
            if frame.frame_id != frame.frame_hash:
                raise ValueError("recovery_artifact frame_id must equal frame_hash")

    for score_name, score in (
        ("carrier_lock_integrity_score", recovery_artifact.carrier_lock_integrity_score),
        ("line_recovery_score", recovery_artifact.line_recovery_score),
        ("burst_recovery_score", recovery_artifact.burst_recovery_score),
        ("sync_frame_consistency_score", recovery_artifact.sync_frame_consistency_score),
        ("overall_recovery_score", recovery_artifact.overall_recovery_score),
    ):
        _validate_unit_interval(score, f"recovery_artifact {score_name}")


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
class OrbitalNoiseFrame:
    frame_id: str
    frame_index: int
    orbital_scenario: str
    orbital_class: str
    segment_id: str
    source_recovery_segment_id: str
    envelope_noise_floor: float
    envelope_latency_pressure: float
    frame_consistency_score: float
    frame_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "orbital_scenario": self.orbital_scenario,
            "orbital_class": self.orbital_class,
            "segment_id": self.segment_id,
            "source_recovery_segment_id": self.source_recovery_segment_id,
            "envelope_noise_floor": self.envelope_noise_floor,
            "envelope_latency_pressure": self.envelope_latency_pressure,
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
class SatelliteBaselineSegment:
    segment_id: str
    segment_index: int
    orbital_scenario: str
    orbital_class: str
    source_recovery_segment_id: str
    frame_count: int
    frames: tuple[OrbitalNoiseFrame, ...]
    orbital_integrity_score: float
    signal_latency_resilience_score: float
    relay_handoff_score: float
    frame_consistency_score: float
    overall_satellite_score: float
    segment_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "segment_id": self.segment_id,
            "segment_index": self.segment_index,
            "orbital_scenario": self.orbital_scenario,
            "orbital_class": self.orbital_class,
            "source_recovery_segment_id": self.source_recovery_segment_id,
            "frame_count": self.frame_count,
            "frames": tuple(frame.to_dict() for frame in self.frames),
            "orbital_integrity_score": self.orbital_integrity_score,
            "signal_latency_resilience_score": self.signal_latency_resilience_score,
            "relay_handoff_score": self.relay_handoff_score,
            "frame_consistency_score": self.frame_consistency_score,
            "overall_satellite_score": self.overall_satellite_score,
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
class SatelliteBaselineResult:
    satellite_baseline_version: int
    source_feature_schema_hash: str
    source_spectral_reasoning_hash: str
    source_copper_channel_battery_hash: str
    source_telecom_recovery_hash: str
    satellite_id: str
    segment_count: int
    frame_count: int
    segments: tuple[SatelliteBaselineSegment, ...]
    orbital_integrity_score: float
    signal_latency_resilience_score: float
    relay_handoff_score: float
    frame_consistency_score: float
    overall_satellite_score: float
    law_invariants: tuple[str, ...]
    satellite_baseline_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "satellite_baseline_version": self.satellite_baseline_version,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "source_spectral_reasoning_hash": self.source_spectral_reasoning_hash,
            "source_copper_channel_battery_hash": self.source_copper_channel_battery_hash,
            "source_telecom_recovery_hash": self.source_telecom_recovery_hash,
            "satellite_id": self.satellite_id,
            "segment_count": self.segment_count,
            "frame_count": self.frame_count,
            "segments": tuple(segment.to_dict() for segment in self.segments),
            "orbital_integrity_score": self.orbital_integrity_score,
            "signal_latency_resilience_score": self.signal_latency_resilience_score,
            "relay_handoff_score": self.relay_handoff_score,
            "frame_consistency_score": self.frame_consistency_score,
            "overall_satellite_score": self.overall_satellite_score,
            "law_invariants": self.law_invariants,
            "satellite_baseline_hash": self.satellite_baseline_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("satellite_baseline_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class SatelliteBaselineReceipt:
    satellite_baseline_version: int
    source_feature_schema_hash: str
    source_spectral_reasoning_hash: str
    source_copper_channel_battery_hash: str
    source_telecom_recovery_hash: str
    satellite_id: str
    satellite_baseline_hash: str
    segment_count: int
    frame_count: int
    overall_satellite_score: float
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "satellite_baseline_version": self.satellite_baseline_version,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "source_spectral_reasoning_hash": self.source_spectral_reasoning_hash,
            "source_copper_channel_battery_hash": self.source_copper_channel_battery_hash,
            "source_telecom_recovery_hash": self.source_telecom_recovery_hash,
            "satellite_id": self.satellite_id,
            "satellite_baseline_hash": self.satellite_baseline_hash,
            "segment_count": self.segment_count,
            "frame_count": self.frame_count,
            "overall_satellite_score": self.overall_satellite_score,
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


def run_satellite_signal_baseline(
    recovery_artifact: TelecomRecoveryResult,
    *,
    float_fixture: tuple[float, ...] | None = None,
    int_fixture: tuple[int, ...] | None = None,
    str_fixture: tuple[str, ...] | None = None,
) -> SatelliteBaselineResult:
    """Build replay-safe satellite baseline artifacts from telecom recovery lineage."""
    _validate_telecom_artifact(recovery_artifact)
    _validate_optional_fixture_payload(float_fixture, int_fixture, str_fixture)

    if len(_ORBITAL_SCENARIO_ORDER) != len(_ORBITAL_CLASS_ORDER):
        raise ValueError("orbital scenario/class configuration mismatch")

    satellite_id = _sha256_hex(
        {
            "source_telecom_recovery_hash": recovery_artifact.telecom_recovery_hash,
            "satellite_baseline_version": _SATELLITE_BASELINE_VERSION,
        }
    )

    segments: list[SatelliteBaselineSegment] = []
    telecom_segments = recovery_artifact.segments
    telecom_segment_count = len(telecom_segments)

    for segment_index, orbital_scenario in enumerate(_ORBITAL_SCENARIO_ORDER):
        orbital_class = _SCENARIO_TO_CLASS[orbital_scenario]
        source_segment = telecom_segments[segment_index % telecom_segment_count]
        severity = _SCENARIO_SEVERITY[orbital_scenario]

        segment_id = _sha256_hex(
            {
                "segment_index": segment_index,
                "orbital_scenario": orbital_scenario,
                "orbital_class": orbital_class,
                "source_recovery_segment_id": source_segment.segment_id,
                "source_telecom_recovery_hash": recovery_artifact.telecom_recovery_hash,
            }
        )

        frame_total = max(1, source_segment.frame_count)
        frames: list[OrbitalNoiseFrame] = []
        for frame_index in range(frame_total):
            source_frame = source_segment.frames[frame_index % len(source_segment.frames)]
            recovery_deficit = 1.0 - source_frame.frame_consistency_score

            envelope_noise_floor = _clamp01((0.03 * (frame_index + 1)) + (severity * recovery_deficit))
            envelope_latency_pressure = _clamp01((0.02 * (segment_index + 1)) + (severity * (1.0 - source_frame.sync_drift_score)))
            frame_consistency_score = _clamp01(
                1.0 - ((envelope_noise_floor * 0.55) + (envelope_latency_pressure * 0.45))
            )

            frame = OrbitalNoiseFrame(
                frame_id="",
                frame_index=frame_index,
                orbital_scenario=orbital_scenario,
                orbital_class=orbital_class,
                segment_id=segment_id,
                source_recovery_segment_id=source_segment.segment_id,
                envelope_noise_floor=envelope_noise_floor,
                envelope_latency_pressure=envelope_latency_pressure,
                frame_consistency_score=frame_consistency_score,
                frame_hash="",
            )
            frame_hash = frame.stable_hash()
            frames.append(replace(frame, frame_id=frame_hash, frame_hash=frame_hash))

        recovery_deficit = 1.0 - source_segment.overall_recovery_score
        frame_consistency_score = _mean(tuple(frame.frame_consistency_score for frame in frames), default=1.0)
        orbital_integrity_score = _clamp01(1.0 - (severity * recovery_deficit))
        signal_latency_resilience_score = _clamp01(
            1.0 - ((severity * 0.70) * (1.0 - source_segment.line_recovery_score))
        )
        relay_handoff_score = _clamp01(
            1.0 - ((severity * 0.80) * (1.0 - source_segment.burst_recovery_score))
        )
        overall_satellite_score = _mean(
            (
                orbital_integrity_score,
                signal_latency_resilience_score,
                relay_handoff_score,
                frame_consistency_score,
            ),
            default=1.0,
        )

        segment = SatelliteBaselineSegment(
            segment_id=segment_id,
            segment_index=segment_index,
            orbital_scenario=orbital_scenario,
            orbital_class=orbital_class,
            source_recovery_segment_id=source_segment.segment_id,
            frame_count=len(frames),
            frames=tuple(frames),
            orbital_integrity_score=orbital_integrity_score,
            signal_latency_resilience_score=signal_latency_resilience_score,
            relay_handoff_score=relay_handoff_score,
            frame_consistency_score=frame_consistency_score,
            overall_satellite_score=overall_satellite_score,
            segment_hash="",
        )
        segments.append(replace(segment, segment_hash=segment.stable_hash()))

    orbital_integrity_score = _mean(tuple(segment.orbital_integrity_score for segment in segments), default=1.0)
    signal_latency_resilience_score = _mean(
        tuple(segment.signal_latency_resilience_score for segment in segments), default=1.0
    )
    relay_handoff_score = _mean(tuple(segment.relay_handoff_score for segment in segments), default=1.0)
    frame_consistency_score = _mean(tuple(segment.frame_consistency_score for segment in segments), default=1.0)
    overall_satellite_score = _mean(tuple(segment.overall_satellite_score for segment in segments), default=1.0)

    for score_name, score in (
        ("orbital_integrity_score", orbital_integrity_score),
        ("signal_latency_resilience_score", signal_latency_resilience_score),
        ("relay_handoff_score", relay_handoff_score),
        ("frame_consistency_score", frame_consistency_score),
        ("overall_satellite_score", overall_satellite_score),
    ):
        _validate_unit_interval(score, score_name)

    artifact = SatelliteBaselineResult(
        satellite_baseline_version=_SATELLITE_BASELINE_VERSION,
        source_feature_schema_hash=recovery_artifact.source_feature_schema_hash,
        source_spectral_reasoning_hash=recovery_artifact.source_spectral_reasoning_hash,
        source_copper_channel_battery_hash=recovery_artifact.source_copper_channel_battery_hash,
        source_telecom_recovery_hash=recovery_artifact.telecom_recovery_hash,
        satellite_id=satellite_id,
        segment_count=len(segments),
        frame_count=sum(segment.frame_count for segment in segments),
        segments=tuple(segments),
        orbital_integrity_score=orbital_integrity_score,
        signal_latency_resilience_score=signal_latency_resilience_score,
        relay_handoff_score=relay_handoff_score,
        frame_consistency_score=frame_consistency_score,
        overall_satellite_score=overall_satellite_score,
        law_invariants=(
            SATELLITE_BASELINE_LAYER_LAW,
            DETERMINISTIC_ORBITAL_ORDERING_RULE,
            REPLAY_SAFE_SATELLITE_IDENTITY_RULE,
            BOUNDED_SATELLITE_SCORE_RULE,
        ),
        satellite_baseline_hash="",
    )
    return replace(artifact, satellite_baseline_hash=artifact.stable_hash())


def export_satellite_baseline_bytes(artifact: SatelliteBaselineResult) -> bytes:
    if not isinstance(artifact, SatelliteBaselineResult):
        raise ValueError("artifact must be a SatelliteBaselineResult")
    return artifact.to_canonical_bytes()


def generate_satellite_baseline_receipt(artifact: SatelliteBaselineResult) -> SatelliteBaselineReceipt:
    if not isinstance(artifact, SatelliteBaselineResult):
        raise ValueError("artifact must be a SatelliteBaselineResult")
    if artifact.stable_hash() != artifact.satellite_baseline_hash:
        raise ValueError("artifact satellite_baseline_hash must match stable_hash")

    receipt = SatelliteBaselineReceipt(
        satellite_baseline_version=artifact.satellite_baseline_version,
        source_feature_schema_hash=artifact.source_feature_schema_hash,
        source_spectral_reasoning_hash=artifact.source_spectral_reasoning_hash,
        source_copper_channel_battery_hash=artifact.source_copper_channel_battery_hash,
        source_telecom_recovery_hash=artifact.source_telecom_recovery_hash,
        satellite_id=artifact.satellite_id,
        satellite_baseline_hash=artifact.satellite_baseline_hash,
        segment_count=artifact.segment_count,
        frame_count=artifact.frame_count,
        overall_satellite_score=artifact.overall_satellite_score,
        receipt_hash="",
    )
    return replace(receipt, receipt_hash=receipt.stable_hash())


__all__ = [
    "BOUNDED_SATELLITE_SCORE_RULE",
    "DETERMINISTIC_ORBITAL_ORDERING_RULE",
    "REPLAY_SAFE_SATELLITE_IDENTITY_RULE",
    "SATELLITE_BASELINE_LAYER_LAW",
    "OrbitalNoiseFrame",
    "SatelliteBaselineReceipt",
    "SatelliteBaselineResult",
    "SatelliteBaselineSegment",
    "export_satellite_baseline_bytes",
    "generate_satellite_baseline_receipt",
    "run_satellite_signal_baseline",
]
