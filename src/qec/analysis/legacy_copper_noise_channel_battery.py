"""v137.9.2 — Legacy Copper Noise Channel Battery.

Deterministic Layer-4 battery consumer of spectral reasoning artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.spectral_reasoning_layer import SpectralReasoningResult

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_COPPER_CHANNEL_BATTERY_VERSION = 1
_CHANNEL_FAMILIES: tuple[str, ...] = ("pots", "dsl", "isdn", "t1", "t3")
_SCENARIO_SPECS: tuple[tuple[str, float, float, float], ...] = (
    ("low_noise", 0.05, 0.04, 0.03),
    ("medium_noise", 0.15, 0.12, 0.10),
    ("severe_noise", 0.35, 0.30, 0.25),
    ("burst_noise", 0.45, 0.36, 0.60),
    ("line_loss", 0.75, 0.70, 0.80),
)

LEGACY_COPPER_CHANNEL_BATTERY_LAW = "LEGACY_COPPER_CHANNEL_BATTERY_LAW"
DETERMINISTIC_SCENARIO_ORDERING_RULE = "DETERMINISTIC_SCENARIO_ORDERING_RULE"
REPLAY_SAFE_CHANNEL_BATTERY_IDENTITY_RULE = "REPLAY_SAFE_CHANNEL_BATTERY_IDENTITY_RULE"
BOUNDED_CHANNEL_SCORE_RULE = "BOUNDED_CHANNEL_SCORE_RULE"


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


def _validate_spectral_artifact(spectral_artifact: SpectralReasoningResult) -> None:
    if not isinstance(spectral_artifact, SpectralReasoningResult):
        raise ValueError("spectral_artifact must be a SpectralReasoningResult")
    if spectral_artifact.stable_hash() != spectral_artifact.spectral_reasoning_hash:
        raise ValueError("spectral_artifact spectral_reasoning_hash must match stable_hash")
    if spectral_artifact.spectral_band_count != len(spectral_artifact.spectral_bands):
        raise ValueError("spectral_artifact spectral_band_count must match len(spectral_bands)")

    expected_band_keys = tuple((idx, name) for idx, name in enumerate(("low_band", "mid_band", "high_band", "residual_band")))
    observed_band_keys = tuple((band.band_index, band.band_name) for band in spectral_artifact.spectral_bands)
    if observed_band_keys != expected_band_keys:
        raise ValueError("spectral_artifact spectral_bands must be in canonical deterministic order")

    feature_total = 0
    for band in spectral_artifact.spectral_bands:
        if band.stable_hash() != band.band_hash:
            raise ValueError("spectral_artifact band_hash must match stable_hash")
        if band.feature_count != len(band.features):
            raise ValueError("spectral_artifact band feature_count must match len(features)")
        feature_total += band.feature_count
        for feature in band.features:
            if feature.stable_hash() != feature.feature_id:
                raise ValueError("spectral_artifact feature_id must match stable_hash")
    if spectral_artifact.spectral_feature_count != feature_total:
        raise ValueError("spectral_artifact spectral_feature_count must match flattened feature count")

    for score_name, score in (
        ("spectral_coherence_score", spectral_artifact.spectral_coherence_score),
        ("band_consistency_score", spectral_artifact.band_consistency_score),
        ("normalization_integrity_score", spectral_artifact.normalization_integrity_score),
        ("feature_projection_score", spectral_artifact.feature_projection_score),
        ("overall_spectral_score", spectral_artifact.overall_spectral_score),
    ):
        _validate_unit_interval(score, f"spectral_artifact {score_name}")


def _canonicalize_optional_fixture_payload(
    attenuation_fixture: tuple[float, ...] | None,
    distortion_fixture: tuple[int, ...] | None,
    label_fixture: tuple[str, ...] | None,
) -> tuple[tuple[float, ...], tuple[int, ...], tuple[str, ...]]:
    canonical_float: tuple[float, ...] = ()
    canonical_int: tuple[int, ...] = ()
    canonical_str: tuple[str, ...] = ()

    if attenuation_fixture is not None:
        if not isinstance(attenuation_fixture, tuple):
            raise ValueError("attenuation_fixture must be a tuple")
        out_float: list[float] = []
        for value in attenuation_fixture:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("attenuation_fixture values must be numeric")
            as_float = float(value)
            if not math.isfinite(as_float):
                raise ValueError("attenuation_fixture values must be finite")
            out_float.append(as_float)
        canonical_float = tuple(sorted(out_float))

    if distortion_fixture is not None:
        if not isinstance(distortion_fixture, tuple):
            raise ValueError("distortion_fixture must be a tuple")
        out_int: list[int] = []
        for value in distortion_fixture:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("distortion_fixture values must be int")
            out_int.append(int(value))
        canonical_int = tuple(sorted(out_int))

    if label_fixture is not None:
        if not isinstance(label_fixture, tuple):
            raise ValueError("label_fixture must be a tuple")
        out_str: list[str] = []
        for value in label_fixture:
            if not isinstance(value, str):
                raise ValueError("label_fixture values must be str")
            out_str.append(value)
        canonical_str = tuple(sorted(out_str))

    return canonical_float, canonical_int, canonical_str


@dataclass(frozen=True)
class CopperChannelFixture:
    fixture_id: str
    fixture_index: int
    channel_family: str
    attenuation_curve: tuple[float, ...]
    distortion_curve: tuple[float, ...]
    fixture_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "fixture_id": self.fixture_id,
            "fixture_index": self.fixture_index,
            "channel_family": self.channel_family,
            "attenuation_curve": self.attenuation_curve,
            "distortion_curve": self.distortion_curve,
            "fixture_hash": self.fixture_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("fixture_id")
        payload.pop("fixture_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class CopperNoiseScenario:
    scenario_id: str
    scenario_index: int
    scenario_name: str
    fixture_ids: tuple[str, ...]
    attenuation_integrity_score: float
    channel_distortion_score: float
    burst_noise_resilience_score: float
    fixture_consistency_score: float
    overall_channel_battery_score: float

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_index": self.scenario_index,
            "scenario_name": self.scenario_name,
            "fixture_ids": self.fixture_ids,
            "attenuation_integrity_score": self.attenuation_integrity_score,
            "channel_distortion_score": self.channel_distortion_score,
            "burst_noise_resilience_score": self.burst_noise_resilience_score,
            "fixture_consistency_score": self.fixture_consistency_score,
            "overall_channel_battery_score": self.overall_channel_battery_score,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("scenario_id")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class CopperChannelBatteryResult:
    copper_channel_battery_version: int
    source_feature_schema_hash: str
    source_spectral_reasoning_hash: str
    fixture_count: int
    scenario_count: int
    fixtures: tuple[CopperChannelFixture, ...]
    scenarios: tuple[CopperNoiseScenario, ...]
    attenuation_integrity_score: float
    channel_distortion_score: float
    burst_noise_resilience_score: float
    fixture_consistency_score: float
    overall_channel_battery_score: float
    law_invariants: tuple[str, ...]
    copper_channel_battery_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "copper_channel_battery_version": self.copper_channel_battery_version,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "source_spectral_reasoning_hash": self.source_spectral_reasoning_hash,
            "fixture_count": self.fixture_count,
            "scenario_count": self.scenario_count,
            "fixtures": tuple(fixture.to_dict() for fixture in self.fixtures),
            "scenarios": tuple(scenario.to_dict() for scenario in self.scenarios),
            "attenuation_integrity_score": self.attenuation_integrity_score,
            "channel_distortion_score": self.channel_distortion_score,
            "burst_noise_resilience_score": self.burst_noise_resilience_score,
            "fixture_consistency_score": self.fixture_consistency_score,
            "overall_channel_battery_score": self.overall_channel_battery_score,
            "law_invariants": self.law_invariants,
            "copper_channel_battery_hash": self.copper_channel_battery_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("copper_channel_battery_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class CopperChannelBatteryReceipt:
    copper_channel_battery_version: int
    source_feature_schema_hash: str
    source_spectral_reasoning_hash: str
    copper_channel_battery_hash: str
    fixture_count: int
    scenario_count: int
    overall_channel_battery_score: float
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "copper_channel_battery_version": self.copper_channel_battery_version,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "source_spectral_reasoning_hash": self.source_spectral_reasoning_hash,
            "copper_channel_battery_hash": self.copper_channel_battery_hash,
            "fixture_count": self.fixture_count,
            "scenario_count": self.scenario_count,
            "overall_channel_battery_score": self.overall_channel_battery_score,
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


def run_legacy_copper_noise_channel_battery(
    spectral_artifact: SpectralReasoningResult,
    *,
    attenuation_fixture: tuple[float, ...] | None = None,
    distortion_fixture: tuple[int, ...] | None = None,
    label_fixture: tuple[str, ...] | None = None,
) -> CopperChannelBatteryResult:
    _validate_spectral_artifact(spectral_artifact)
    # Optional payloads are canonicalized and validated for deterministic readiness,
    # but battery identity is strictly spectral-derived per
    # REPLAY_SAFE_CHANNEL_BATTERY_IDENTITY_RULE.
    _canonicalize_optional_fixture_payload(attenuation_fixture, distortion_fixture, label_fixture)

    continuity_baseline = _mean(
        (
            float(spectral_artifact.spectral_coherence_score),
            float(spectral_artifact.normalization_integrity_score),
            float(spectral_artifact.overall_spectral_score),
        ),
        default=1.0,
    )

    family_multipliers: tuple[float, ...] = (0.99, 0.97, 0.98, 0.96, 0.95)
    fixtures: list[CopperChannelFixture] = []
    for fixture_index, (channel_family, multiplier) in enumerate(zip(_CHANNEL_FAMILIES, family_multipliers)):
        attenuation_curve = tuple(
            _clamp01(continuity_baseline * multiplier * (1.0 - (step * 0.08)))
            for step in range(5)
        )
        distortion_curve = tuple(
            _clamp01((1.0 - (step * 0.15)) * spectral_artifact.band_consistency_score * multiplier)
            for step in range(4)
        )
        fixture = CopperChannelFixture(
            fixture_id="",
            fixture_index=fixture_index,
            channel_family=channel_family,
            attenuation_curve=attenuation_curve,
            distortion_curve=distortion_curve,
            fixture_hash="",
        )
        fixture_hash = fixture.stable_hash()
        fixture = replace(fixture, fixture_id=fixture_hash, fixture_hash=fixture_hash)
        fixtures.append(fixture)

    fixture_consistency_base = _mean(
        tuple(_mean(fixture.attenuation_curve, default=0.0) for fixture in fixtures),
        default=1.0,
    )

    scenarios: list[CopperNoiseScenario] = []
    fixture_ids = tuple(fixture.fixture_id for fixture in fixtures)
    for scenario_index, (scenario_name, attenuation_penalty, distortion_penalty, burst_penalty) in enumerate(_SCENARIO_SPECS):
        attenuation_integrity_score = _clamp01(continuity_baseline * (1.0 - attenuation_penalty))
        channel_distortion_score = _clamp01(
            spectral_artifact.band_consistency_score * (1.0 - distortion_penalty)
        )
        burst_noise_resilience_score = _clamp01(
            spectral_artifact.feature_projection_score * (1.0 - burst_penalty)
        )
        fixture_consistency_score = _clamp01(fixture_consistency_base * (1.0 - (attenuation_penalty / 2.0)))
        overall_channel_battery_score = _mean(
            (
                attenuation_integrity_score,
                channel_distortion_score,
                burst_noise_resilience_score,
                fixture_consistency_score,
            ),
            default=0.0,
        )
        scenario = CopperNoiseScenario(
            scenario_id="",
            scenario_index=scenario_index,
            scenario_name=scenario_name,
            fixture_ids=fixture_ids,
            attenuation_integrity_score=attenuation_integrity_score,
            channel_distortion_score=channel_distortion_score,
            burst_noise_resilience_score=burst_noise_resilience_score,
            fixture_consistency_score=fixture_consistency_score,
            overall_channel_battery_score=overall_channel_battery_score,
        )
        scenarios.append(replace(scenario, scenario_id=scenario.stable_hash()))

    attenuation_integrity_score = _mean(tuple(s.attenuation_integrity_score for s in scenarios), default=1.0)
    channel_distortion_score = _mean(tuple(s.channel_distortion_score for s in scenarios), default=1.0)
    burst_noise_resilience_score = _mean(tuple(s.burst_noise_resilience_score for s in scenarios), default=1.0)
    fixture_consistency_score = _mean(tuple(s.fixture_consistency_score for s in scenarios), default=1.0)
    overall_channel_battery_score = _mean(tuple(s.overall_channel_battery_score for s in scenarios), default=1.0)

    for score_name, score in (
        ("attenuation_integrity_score", attenuation_integrity_score),
        ("channel_distortion_score", channel_distortion_score),
        ("burst_noise_resilience_score", burst_noise_resilience_score),
        ("fixture_consistency_score", fixture_consistency_score),
        ("overall_channel_battery_score", overall_channel_battery_score),
    ):
        _validate_unit_interval(score, score_name)

    artifact = CopperChannelBatteryResult(
        copper_channel_battery_version=_COPPER_CHANNEL_BATTERY_VERSION,
        source_feature_schema_hash=spectral_artifact.source_feature_schema_hash,
        source_spectral_reasoning_hash=spectral_artifact.spectral_reasoning_hash,
        fixture_count=len(fixtures),
        scenario_count=len(scenarios),
        fixtures=tuple(fixtures),
        scenarios=tuple(scenarios),
        attenuation_integrity_score=attenuation_integrity_score,
        channel_distortion_score=channel_distortion_score,
        burst_noise_resilience_score=burst_noise_resilience_score,
        fixture_consistency_score=fixture_consistency_score,
        overall_channel_battery_score=overall_channel_battery_score,
        law_invariants=(
            LEGACY_COPPER_CHANNEL_BATTERY_LAW,
            DETERMINISTIC_SCENARIO_ORDERING_RULE,
            REPLAY_SAFE_CHANNEL_BATTERY_IDENTITY_RULE,
            BOUNDED_CHANNEL_SCORE_RULE,
        ),
        copper_channel_battery_hash="",
    )
    return replace(artifact, copper_channel_battery_hash=artifact.stable_hash())


def export_copper_channel_battery_bytes(artifact: CopperChannelBatteryResult) -> bytes:
    if not isinstance(artifact, CopperChannelBatteryResult):
        raise ValueError("artifact must be a CopperChannelBatteryResult")
    return artifact.to_canonical_bytes()


def generate_copper_channel_battery_receipt(artifact: CopperChannelBatteryResult) -> CopperChannelBatteryReceipt:
    if not isinstance(artifact, CopperChannelBatteryResult):
        raise ValueError("artifact must be a CopperChannelBatteryResult")
    if artifact.stable_hash() != artifact.copper_channel_battery_hash:
        raise ValueError("artifact copper_channel_battery_hash must match stable_hash")

    receipt = CopperChannelBatteryReceipt(
        copper_channel_battery_version=artifact.copper_channel_battery_version,
        source_feature_schema_hash=artifact.source_feature_schema_hash,
        source_spectral_reasoning_hash=artifact.source_spectral_reasoning_hash,
        copper_channel_battery_hash=artifact.copper_channel_battery_hash,
        fixture_count=artifact.fixture_count,
        scenario_count=artifact.scenario_count,
        overall_channel_battery_score=artifact.overall_channel_battery_score,
        receipt_hash="",
    )
    return replace(receipt, receipt_hash=receipt.stable_hash())


__all__ = [
    "BOUNDED_CHANNEL_SCORE_RULE",
    "DETERMINISTIC_SCENARIO_ORDERING_RULE",
    "LEGACY_COPPER_CHANNEL_BATTERY_LAW",
    "REPLAY_SAFE_CHANNEL_BATTERY_IDENTITY_RULE",
    "CopperChannelBatteryReceipt",
    "CopperChannelBatteryResult",
    "CopperChannelFixture",
    "CopperNoiseScenario",
    "export_copper_channel_battery_bytes",
    "generate_copper_channel_battery_receipt",
    "run_legacy_copper_noise_channel_battery",
]
