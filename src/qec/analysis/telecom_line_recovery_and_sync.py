"""v137.9.3 — Telecom Line Recovery + Carrier Synchronization.

Deterministic Layer-4 consumer of legacy copper channel battery artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.legacy_copper_noise_channel_battery import CopperChannelBatteryResult

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_TELECOM_RECOVERY_VERSION = 1
_RECOVERY_MODE_ORDER: tuple[str, ...] = (
    "line_relock",
    "burst_recovery",
    "carrier_phase_sync",
    "attenuation_compensation",
    "continuity_rebuild",
)
_RECOVERY_SCENARIOS: tuple[str, ...] = (
    "degraded_relock",
    "burst_recovery",
    "carrier_phase_lock",
    "severe_line_recovery",
    "nominal_recovery",
)

TELECOM_RECOVERY_LAYER_LAW = "TELECOM_RECOVERY_LAYER_LAW"
DETERMINISTIC_SYNC_ORDERING_RULE = "DETERMINISTIC_SYNC_ORDERING_RULE"
REPLAY_SAFE_RECOVERY_IDENTITY_RULE = "REPLAY_SAFE_RECOVERY_IDENTITY_RULE"
BOUNDED_RECOVERY_SCORE_RULE = "BOUNDED_RECOVERY_SCORE_RULE"


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


def _validate_battery_artifact(battery_artifact: CopperChannelBatteryResult) -> None:
    if not isinstance(battery_artifact, CopperChannelBatteryResult):
        raise ValueError("battery_artifact must be a CopperChannelBatteryResult")
    if battery_artifact.stable_hash() != battery_artifact.copper_channel_battery_hash:
        raise ValueError("battery_artifact copper_channel_battery_hash must match stable_hash")
    if battery_artifact.fixture_count != len(battery_artifact.fixtures):
        raise ValueError("battery_artifact fixture_count must match len(fixtures)")
    if battery_artifact.scenario_count != len(battery_artifact.scenarios):
        raise ValueError("battery_artifact scenario_count must match len(scenarios)")
    for fixture in battery_artifact.fixtures:
        if fixture.stable_hash() != fixture.fixture_hash:
            raise ValueError("battery_artifact fixture_hash must match stable_hash")
    expected_fixture_order = tuple(sorted(battery_artifact.fixtures, key=lambda f: (f.fixture_index, f.channel_family)))
    if battery_artifact.fixtures != expected_fixture_order:
        raise ValueError("battery_artifact fixtures must be in canonical deterministic order")

    expected_scenario_order = tuple(sorted(battery_artifact.scenarios, key=lambda s: (s.scenario_index, s.scenario_name)))
    if battery_artifact.scenarios != expected_scenario_order:
        raise ValueError("battery_artifact scenarios must be in canonical deterministic order")

    for score_name, score in (
        ("attenuation_integrity_score", battery_artifact.attenuation_integrity_score),
        ("channel_distortion_score", battery_artifact.channel_distortion_score),
        ("burst_noise_resilience_score", battery_artifact.burst_noise_resilience_score),
        ("fixture_consistency_score", battery_artifact.fixture_consistency_score),
        ("overall_channel_battery_score", battery_artifact.overall_channel_battery_score),
    ):
        _validate_unit_interval(score, f"battery_artifact {score_name}")


def _validate_optional_fixture_payload(
    gain_fixture: tuple[float, ...] | None,
    line_index_fixture: tuple[int, ...] | None,
    tag_fixture: tuple[str, ...] | None,
) -> None:
    if gain_fixture is not None:
        if not isinstance(gain_fixture, tuple):
            raise ValueError("gain_fixture must be a tuple")
        for value in gain_fixture:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("gain_fixture values must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError("gain_fixture values must be finite")

    if line_index_fixture is not None:
        if not isinstance(line_index_fixture, tuple):
            raise ValueError("line_index_fixture must be a tuple")
        for value in line_index_fixture:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("line_index_fixture values must be int")

    if tag_fixture is not None:
        if not isinstance(tag_fixture, tuple):
            raise ValueError("tag_fixture must be a tuple")
        for value in tag_fixture:
            if not isinstance(value, str):
                raise ValueError("tag_fixture values must be str")


@dataclass(frozen=True)
class CarrierSyncFrame:
    frame_id: str
    frame_index: int
    recovery_mode: str
    segment_id: str
    phase_reference: float
    carrier_lock_score: float
    sync_drift_score: float
    frame_consistency_score: float
    frame_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "recovery_mode": self.recovery_mode,
            "segment_id": self.segment_id,
            "phase_reference": self.phase_reference,
            "carrier_lock_score": self.carrier_lock_score,
            "sync_drift_score": self.sync_drift_score,
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
class TelecomRecoverySegment:
    segment_id: str
    segment_index: int
    recovery_mode: str
    recovery_scenario: str
    source_scenario_id: str
    fixture_ids: tuple[str, ...]
    frame_count: int
    frames: tuple[CarrierSyncFrame, ...]
    carrier_lock_integrity_score: float
    line_recovery_score: float
    burst_recovery_score: float
    sync_frame_consistency_score: float
    overall_recovery_score: float
    segment_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "segment_id": self.segment_id,
            "segment_index": self.segment_index,
            "recovery_mode": self.recovery_mode,
            "recovery_scenario": self.recovery_scenario,
            "source_scenario_id": self.source_scenario_id,
            "fixture_ids": self.fixture_ids,
            "frame_count": self.frame_count,
            "frames": tuple(frame.to_dict() for frame in self.frames),
            "carrier_lock_integrity_score": self.carrier_lock_integrity_score,
            "line_recovery_score": self.line_recovery_score,
            "burst_recovery_score": self.burst_recovery_score,
            "sync_frame_consistency_score": self.sync_frame_consistency_score,
            "overall_recovery_score": self.overall_recovery_score,
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
class TelecomRecoveryResult:
    telecom_recovery_version: int
    source_feature_schema_hash: str
    source_spectral_reasoning_hash: str
    source_copper_channel_battery_hash: str
    segment_count: int
    frame_count: int
    segments: tuple[TelecomRecoverySegment, ...]
    carrier_lock_integrity_score: float
    line_recovery_score: float
    burst_recovery_score: float
    sync_frame_consistency_score: float
    overall_recovery_score: float
    law_invariants: tuple[str, ...]
    telecom_recovery_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "telecom_recovery_version": self.telecom_recovery_version,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "source_spectral_reasoning_hash": self.source_spectral_reasoning_hash,
            "source_copper_channel_battery_hash": self.source_copper_channel_battery_hash,
            "segment_count": self.segment_count,
            "frame_count": self.frame_count,
            "segments": tuple(segment.to_dict() for segment in self.segments),
            "carrier_lock_integrity_score": self.carrier_lock_integrity_score,
            "line_recovery_score": self.line_recovery_score,
            "burst_recovery_score": self.burst_recovery_score,
            "sync_frame_consistency_score": self.sync_frame_consistency_score,
            "overall_recovery_score": self.overall_recovery_score,
            "law_invariants": self.law_invariants,
            "telecom_recovery_hash": self.telecom_recovery_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("telecom_recovery_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class TelecomRecoveryReceipt:
    telecom_recovery_version: int
    source_feature_schema_hash: str
    source_spectral_reasoning_hash: str
    source_copper_channel_battery_hash: str
    telecom_recovery_hash: str
    segment_count: int
    frame_count: int
    overall_recovery_score: float
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "telecom_recovery_version": self.telecom_recovery_version,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "source_spectral_reasoning_hash": self.source_spectral_reasoning_hash,
            "source_copper_channel_battery_hash": self.source_copper_channel_battery_hash,
            "telecom_recovery_hash": self.telecom_recovery_hash,
            "segment_count": self.segment_count,
            "frame_count": self.frame_count,
            "overall_recovery_score": self.overall_recovery_score,
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


def run_telecom_line_recovery(
    battery_artifact: CopperChannelBatteryResult,
    *,
    gain_fixture: tuple[float, ...] | None = None,
    line_index_fixture: tuple[int, ...] | None = None,
    tag_fixture: tuple[str, ...] | None = None,
) -> TelecomRecoveryResult:
    _validate_battery_artifact(battery_artifact)
    _validate_optional_fixture_payload(gain_fixture, line_index_fixture, tag_fixture)

    line_health = _mean(
        (
            battery_artifact.attenuation_integrity_score,
            battery_artifact.channel_distortion_score,
            battery_artifact.fixture_consistency_score,
        ),
        default=1.0,
    )
    burst_health = _mean((battery_artifact.burst_noise_resilience_score,), default=1.0)
    continuity_health = _mean((battery_artifact.overall_channel_battery_score, line_health), default=1.0)

    fixture_ids = tuple(fixture.fixture_id for fixture in battery_artifact.fixtures)
    source_scenario_ids = tuple(scenario.scenario_id for scenario in battery_artifact.scenarios)
    scenario_count = len(source_scenario_ids)

    segments: list[TelecomRecoverySegment] = []
    for segment_index, (recovery_mode, recovery_scenario) in enumerate(zip(_RECOVERY_MODE_ORDER, _RECOVERY_SCENARIOS)):
        mode_weight = 1.0 - (segment_index * 0.07)
        carrier_lock_integrity_score = _clamp01(continuity_health * mode_weight)
        line_recovery_score = _clamp01(line_health * (1.0 - (segment_index * 0.04)))
        burst_recovery_score = _clamp01(burst_health * (1.0 - (segment_index * 0.10)))

        source_scenario_id = source_scenario_ids[segment_index % scenario_count] if scenario_count > 0 else ""

        segment_id = _sha256_hex(
            {
                "segment_index": segment_index,
                "recovery_mode": recovery_mode,
                "recovery_scenario": recovery_scenario,
                "source_copper_channel_battery_hash": battery_artifact.copper_channel_battery_hash,
                "source_scenario_id": source_scenario_id,
            }
        )

        frame_total = 3 if fixture_ids else 1
        frames: list[CarrierSyncFrame] = []
        for frame_index in range(frame_total):
            phase_reference = _clamp01((0.125 * (frame_index + 1)) + (0.05 * segment_index))
            sync_drift_score = _clamp01(1.0 - (frame_index * 0.08) - (segment_index * 0.03))
            carrier_lock_score = _clamp01(carrier_lock_integrity_score * sync_drift_score)
            frame_consistency_score = _mean((carrier_lock_score, sync_drift_score), default=0.0)
            frame = CarrierSyncFrame(
                frame_id="",
                frame_index=frame_index,
                recovery_mode=recovery_mode,
                segment_id=segment_id,
                phase_reference=phase_reference,
                carrier_lock_score=carrier_lock_score,
                sync_drift_score=sync_drift_score,
                frame_consistency_score=frame_consistency_score,
                frame_hash="",
            )
            frame_hash = frame.stable_hash()
            frames.append(replace(frame, frame_id=frame_hash, frame_hash=frame_hash))

        sync_frame_consistency_score = _mean(tuple(frame.frame_consistency_score for frame in frames), default=1.0)
        overall_recovery_score = _mean(
            (
                carrier_lock_integrity_score,
                line_recovery_score,
                burst_recovery_score,
                sync_frame_consistency_score,
            ),
            default=0.0,
        )

        segment = TelecomRecoverySegment(
            segment_id=segment_id,
            segment_index=segment_index,
            recovery_mode=recovery_mode,
            recovery_scenario=recovery_scenario,
            source_scenario_id=source_scenario_id,
            fixture_ids=fixture_ids,
            frame_count=len(frames),
            frames=tuple(frames),
            carrier_lock_integrity_score=carrier_lock_integrity_score,
            line_recovery_score=line_recovery_score,
            burst_recovery_score=burst_recovery_score,
            sync_frame_consistency_score=sync_frame_consistency_score,
            overall_recovery_score=overall_recovery_score,
            segment_hash="",
        )
        segments.append(replace(segment, segment_hash=segment.stable_hash()))

    carrier_lock_integrity_score = _mean(tuple(s.carrier_lock_integrity_score for s in segments), default=1.0)
    line_recovery_score = _mean(tuple(s.line_recovery_score for s in segments), default=1.0)
    burst_recovery_score = _mean(tuple(s.burst_recovery_score for s in segments), default=1.0)
    sync_frame_consistency_score = _mean(tuple(s.sync_frame_consistency_score for s in segments), default=1.0)
    overall_recovery_score = _mean(tuple(s.overall_recovery_score for s in segments), default=1.0)

    for score_name, score in (
        ("carrier_lock_integrity_score", carrier_lock_integrity_score),
        ("line_recovery_score", line_recovery_score),
        ("burst_recovery_score", burst_recovery_score),
        ("sync_frame_consistency_score", sync_frame_consistency_score),
        ("overall_recovery_score", overall_recovery_score),
    ):
        _validate_unit_interval(score, score_name)

    artifact = TelecomRecoveryResult(
        telecom_recovery_version=_TELECOM_RECOVERY_VERSION,
        source_feature_schema_hash=battery_artifact.source_feature_schema_hash,
        source_spectral_reasoning_hash=battery_artifact.source_spectral_reasoning_hash,
        source_copper_channel_battery_hash=battery_artifact.copper_channel_battery_hash,
        segment_count=len(segments),
        frame_count=sum(segment.frame_count for segment in segments),
        segments=tuple(segments),
        carrier_lock_integrity_score=carrier_lock_integrity_score,
        line_recovery_score=line_recovery_score,
        burst_recovery_score=burst_recovery_score,
        sync_frame_consistency_score=sync_frame_consistency_score,
        overall_recovery_score=overall_recovery_score,
        law_invariants=(
            TELECOM_RECOVERY_LAYER_LAW,
            DETERMINISTIC_SYNC_ORDERING_RULE,
            REPLAY_SAFE_RECOVERY_IDENTITY_RULE,
            BOUNDED_RECOVERY_SCORE_RULE,
        ),
        telecom_recovery_hash="",
    )
    return replace(artifact, telecom_recovery_hash=artifact.stable_hash())


def export_telecom_recovery_bytes(artifact: TelecomRecoveryResult) -> bytes:
    if not isinstance(artifact, TelecomRecoveryResult):
        raise ValueError("artifact must be a TelecomRecoveryResult")
    return artifact.to_canonical_bytes()


def generate_telecom_recovery_receipt(artifact: TelecomRecoveryResult) -> TelecomRecoveryReceipt:
    if not isinstance(artifact, TelecomRecoveryResult):
        raise ValueError("artifact must be a TelecomRecoveryResult")
    if artifact.stable_hash() != artifact.telecom_recovery_hash:
        raise ValueError("artifact telecom_recovery_hash must match stable_hash")

    receipt = TelecomRecoveryReceipt(
        telecom_recovery_version=artifact.telecom_recovery_version,
        source_feature_schema_hash=artifact.source_feature_schema_hash,
        source_spectral_reasoning_hash=artifact.source_spectral_reasoning_hash,
        source_copper_channel_battery_hash=artifact.source_copper_channel_battery_hash,
        telecom_recovery_hash=artifact.telecom_recovery_hash,
        segment_count=artifact.segment_count,
        frame_count=artifact.frame_count,
        overall_recovery_score=artifact.overall_recovery_score,
        receipt_hash="",
    )
    return replace(receipt, receipt_hash=receipt.stable_hash())


__all__ = [
    "BOUNDED_RECOVERY_SCORE_RULE",
    "DETERMINISTIC_SYNC_ORDERING_RULE",
    "REPLAY_SAFE_RECOVERY_IDENTITY_RULE",
    "TELECOM_RECOVERY_LAYER_LAW",
    "CarrierSyncFrame",
    "TelecomRecoveryReceipt",
    "TelecomRecoveryResult",
    "TelecomRecoverySegment",
    "export_telecom_recovery_bytes",
    "generate_telecom_recovery_receipt",
    "run_telecom_line_recovery",
]
