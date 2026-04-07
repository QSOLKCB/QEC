"""v137.9.6 — Cross-Modal Replay Certification.

Deterministic Layer-4 certification consumer of RF equalization artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.multimodal_feature_schema import MultimodalFeatureSchemaResult
from qec.analysis.rf_equalization_and_ground_station_compensation import RFEqualizationResult
from qec.analysis.satellite_signal_baseline_and_orbital_noise import SatelliteBaselineResult
from qec.analysis.spectral_reasoning_layer import SpectralReasoningResult
from qec.analysis.telecom_line_recovery_and_sync import TelecomRecoveryResult

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_CROSS_MODAL_REPLAY_CERTIFICATION_VERSION = 1
_CERTIFICATION_PROFILE_ORDER: tuple[str, ...] = (
    "layer_to_layer",
    "full_stack",
    "degraded_signal_replay",
    "recovery_replay",
    "end_to_end_replay",
)

CROSS_MODAL_REPLAY_CERTIFICATION_LAW = "CROSS_MODAL_REPLAY_CERTIFICATION_LAW"
DETERMINISTIC_CERTIFICATION_ORDERING_RULE = "DETERMINISTIC_CERTIFICATION_ORDERING_RULE"
REPLAY_SAFE_CERTIFICATION_IDENTITY_RULE = "REPLAY_SAFE_CERTIFICATION_IDENTITY_RULE"
BOUNDED_CERTIFICATION_SCORE_RULE = "BOUNDED_CERTIFICATION_SCORE_RULE"


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


def _validate_profile(certification_profile: str) -> None:
    if certification_profile not in _CERTIFICATION_PROFILE_ORDER:
        raise ValueError(
            f"certification_profile must be one of {_CERTIFICATION_PROFILE_ORDER}; "
            f"received {certification_profile!r}"
        )


def _validate_rf_artifact(rf_artifact: RFEqualizationResult) -> None:
    if not isinstance(rf_artifact, RFEqualizationResult):
        raise ValueError("rf_artifact must be a RFEqualizationResult")
    if rf_artifact.stable_hash() != rf_artifact.rf_equalization_hash:
        raise ValueError("rf_artifact rf_equalization_hash must match stable_hash")
    if rf_artifact.segment_count != len(rf_artifact.segments):
        raise ValueError("rf_artifact segment_count must match len(segments)")
    if rf_artifact.segment_count <= 0 or not rf_artifact.segments:
        raise ValueError("rf_artifact must contain at least one segment")

    expected_segment_order = tuple(
        sorted(
            rf_artifact.segments,
            key=lambda s: (s.segment_index, s.ground_station_profile, s.compensation_scenario, s.segment_id),
        )
    )
    if rf_artifact.segments != expected_segment_order:
        raise ValueError("rf_artifact segments must be in canonical deterministic order")

    if rf_artifact.frame_count != sum(segment.frame_count for segment in rf_artifact.segments):
        raise ValueError("rf_artifact frame_count must match summed segment frame_count")

    for score_name, score_value in (
        ("equalization_integrity_score", rf_artifact.equalization_integrity_score),
        ("compensation_stability_score", rf_artifact.compensation_stability_score),
        ("reflection_resilience_score", rf_artifact.reflection_resilience_score),
        ("frame_consistency_score", rf_artifact.frame_consistency_score),
    ):
        _validate_unit_interval(score_value, f"rf_artifact {score_name}")

    for segment in rf_artifact.segments:
        if segment.segment_hash != segment.stable_hash():
            raise ValueError("rf_artifact segment_hash must match stable_hash")
        if segment.frame_count <= 0 or not segment.frames:
            raise ValueError("rf_artifact each segment must contain at least one frame")
        if segment.frame_count != len(segment.frames):
            raise ValueError("rf_artifact frame_count must match len(frames) per segment")
        for score_name, score_value in (
            ("equalization_integrity_score", segment.equalization_integrity_score),
            ("compensation_stability_score", segment.compensation_stability_score),
            ("reflection_resilience_score", segment.reflection_resilience_score),
            ("frame_consistency_score", segment.frame_consistency_score),
        ):
            _validate_unit_interval(score_value, f"rf_artifact segment {score_name}")
        expected_frame_order = tuple(
            sorted(segment.frames, key=lambda f: (f.frame_index, f.compensation_scenario, f.frame_id))
        )
        if segment.frames != expected_frame_order:
            raise ValueError("rf_artifact frames must be in canonical deterministic order")
        for frame in segment.frames:
            if frame.frame_hash != frame.stable_hash():
                raise ValueError("rf_artifact frame_hash must match stable_hash")
            if frame.frame_id != frame.frame_hash:
                raise ValueError("rf_artifact frame_id must equal frame_hash")
            for score_name, score_value in (
                ("reflection_resilience_score", frame.reflection_resilience_score),
                ("frame_consistency_score", frame.frame_consistency_score),
            ):
                _validate_unit_interval(score_value, f"rf_artifact frame {score_name}")


def _validate_direct_lineage(
    rf_artifact: RFEqualizationResult,
    *,
    satellite_artifact: SatelliteBaselineResult | None,
    telecom_artifact: TelecomRecoveryResult | None,
    spectral_artifact: SpectralReasoningResult | None,
    schema_artifact: MultimodalFeatureSchemaResult | None,
) -> None:
    if satellite_artifact is not None:
        if not isinstance(satellite_artifact, SatelliteBaselineResult):
            raise ValueError("satellite_artifact must be a SatelliteBaselineResult")
        if satellite_artifact.stable_hash() != satellite_artifact.satellite_baseline_hash:
            raise ValueError("satellite_artifact satellite_baseline_hash must match stable_hash")
        if satellite_artifact.satellite_baseline_hash != rf_artifact.source_satellite_baseline_hash:
            raise ValueError("direct lineage mismatch: satellite_baseline_hash")
        if satellite_artifact.source_telecom_recovery_hash != rf_artifact.source_telecom_recovery_hash:
            raise ValueError("direct lineage mismatch: satellite->telecom")

    if telecom_artifact is not None:
        if not isinstance(telecom_artifact, TelecomRecoveryResult):
            raise ValueError("telecom_artifact must be a TelecomRecoveryResult")
        if telecom_artifact.stable_hash() != telecom_artifact.telecom_recovery_hash:
            raise ValueError("telecom_artifact telecom_recovery_hash must match stable_hash")
        if telecom_artifact.telecom_recovery_hash != rf_artifact.source_telecom_recovery_hash:
            raise ValueError("direct lineage mismatch: telecom_recovery_hash")
        if telecom_artifact.source_copper_channel_battery_hash != rf_artifact.source_copper_channel_battery_hash:
            raise ValueError("direct lineage mismatch: telecom->copper")
        if telecom_artifact.source_spectral_reasoning_hash != rf_artifact.source_spectral_reasoning_hash:
            raise ValueError("direct lineage mismatch: telecom->spectral")

    if spectral_artifact is not None:
        if not isinstance(spectral_artifact, SpectralReasoningResult):
            raise ValueError("spectral_artifact must be a SpectralReasoningResult")
        if spectral_artifact.stable_hash() != spectral_artifact.spectral_reasoning_hash:
            raise ValueError("spectral_artifact spectral_reasoning_hash must match stable_hash")
        if spectral_artifact.spectral_reasoning_hash != rf_artifact.source_spectral_reasoning_hash:
            raise ValueError("direct lineage mismatch: spectral_reasoning_hash")
        if spectral_artifact.source_feature_schema_hash != rf_artifact.source_feature_schema_hash:
            raise ValueError("direct lineage mismatch: spectral->schema")

    if schema_artifact is not None:
        if not isinstance(schema_artifact, MultimodalFeatureSchemaResult):
            raise ValueError("schema_artifact must be a MultimodalFeatureSchemaResult")
        if schema_artifact.stable_hash() != schema_artifact.feature_schema_hash:
            raise ValueError("schema_artifact feature_schema_hash must match stable_hash")
        if schema_artifact.feature_schema_hash != rf_artifact.source_feature_schema_hash:
            raise ValueError("direct lineage mismatch: feature_schema_hash")

    if satellite_artifact is not None and telecom_artifact is not None:
        if satellite_artifact.source_telecom_recovery_hash != telecom_artifact.telecom_recovery_hash:
            raise ValueError("direct lineage mismatch: satellite->telecom")
    if telecom_artifact is not None and spectral_artifact is not None:
        if telecom_artifact.source_spectral_reasoning_hash != spectral_artifact.spectral_reasoning_hash:
            raise ValueError("direct lineage mismatch: telecom->spectral")
    if spectral_artifact is not None and schema_artifact is not None:
        if spectral_artifact.source_feature_schema_hash != schema_artifact.feature_schema_hash:
            raise ValueError("direct lineage mismatch: spectral->schema")


def _validate_optional_fixture_payload(score_fixture: tuple[float, ...] | None) -> None:
    if score_fixture is None:
        return
    if not isinstance(score_fixture, tuple):
        raise ValueError("score_fixture must be a tuple")
    for value in score_fixture:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("score_fixture values must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError("score_fixture values must be finite")


def _finalize_identity(obj: Any, hash_field: str) -> Any:
    return replace(obj, **{hash_field: obj.stable_hash()})


@dataclass(frozen=True)
class CrossModalReplayWitness:
    witness_id: str
    witness_index: int
    certification_profile: str
    source_rf_segment_id: str
    source_rf_frame_count: int
    lineage_chain: tuple[str, ...]
    modal_alignment_score: float
    replay_identity_score: float
    witness_integrity_score: float
    witness_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "witness_id": self.witness_id,
            "witness_index": self.witness_index,
            "certification_profile": self.certification_profile,
            "source_rf_segment_id": self.source_rf_segment_id,
            "source_rf_frame_count": self.source_rf_frame_count,
            "lineage_chain": self.lineage_chain,
            "modal_alignment_score": self.modal_alignment_score,
            "replay_identity_score": self.replay_identity_score,
            "witness_integrity_score": self.witness_integrity_score,
            "witness_hash": self.witness_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("witness_id")
        payload.pop("witness_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class CrossModalReplayLedger:
    ledger_id: str
    certification_profile: str
    lineage_chain: tuple[str, ...]
    witness_count: int
    witnesses: tuple[CrossModalReplayWitness, ...]
    ledger_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "ledger_id": self.ledger_id,
            "certification_profile": self.certification_profile,
            "lineage_chain": self.lineage_chain,
            "witness_count": self.witness_count,
            "witnesses": tuple(witness.to_dict() for witness in self.witnesses),
            "ledger_hash": self.ledger_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("ledger_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class CrossModalReplayCertificationResult:
    cross_modal_replay_certification_version: int
    source_feature_schema_hash: str
    source_spectral_reasoning_hash: str
    source_copper_channel_battery_hash: str
    source_telecom_recovery_hash: str
    source_satellite_baseline_hash: str
    source_rf_equalization_hash: str
    replay_certification_id: str
    certification_profile: str
    ledger: CrossModalReplayLedger
    lineage_consistency_score: float
    modal_alignment_score: float
    replay_identity_score: float
    witness_integrity_score: float
    overall_certification_score: float
    law_invariants: tuple[str, ...]
    replay_certification_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "cross_modal_replay_certification_version": self.cross_modal_replay_certification_version,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "source_spectral_reasoning_hash": self.source_spectral_reasoning_hash,
            "source_copper_channel_battery_hash": self.source_copper_channel_battery_hash,
            "source_telecom_recovery_hash": self.source_telecom_recovery_hash,
            "source_satellite_baseline_hash": self.source_satellite_baseline_hash,
            "source_rf_equalization_hash": self.source_rf_equalization_hash,
            "replay_certification_id": self.replay_certification_id,
            "certification_profile": self.certification_profile,
            "ledger": self.ledger.to_dict(),
            "lineage_consistency_score": self.lineage_consistency_score,
            "modal_alignment_score": self.modal_alignment_score,
            "replay_identity_score": self.replay_identity_score,
            "witness_integrity_score": self.witness_integrity_score,
            "overall_certification_score": self.overall_certification_score,
            "law_invariants": self.law_invariants,
            "replay_certification_hash": self.replay_certification_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("replay_certification_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class CrossModalReplayCertificationReceipt:
    cross_modal_replay_certification_version: int
    source_feature_schema_hash: str
    source_spectral_reasoning_hash: str
    source_copper_channel_battery_hash: str
    source_telecom_recovery_hash: str
    source_satellite_baseline_hash: str
    source_rf_equalization_hash: str
    replay_certification_id: str
    certification_profile: str
    witness_count: int
    overall_certification_score: float
    replay_certification_hash: str
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "cross_modal_replay_certification_version": self.cross_modal_replay_certification_version,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "source_spectral_reasoning_hash": self.source_spectral_reasoning_hash,
            "source_copper_channel_battery_hash": self.source_copper_channel_battery_hash,
            "source_telecom_recovery_hash": self.source_telecom_recovery_hash,
            "source_satellite_baseline_hash": self.source_satellite_baseline_hash,
            "source_rf_equalization_hash": self.source_rf_equalization_hash,
            "replay_certification_id": self.replay_certification_id,
            "certification_profile": self.certification_profile,
            "witness_count": self.witness_count,
            "overall_certification_score": self.overall_certification_score,
            "replay_certification_hash": self.replay_certification_hash,
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


def _build_lineage_chain(rf_artifact: RFEqualizationResult) -> tuple[str, ...]:
    return (
        rf_artifact.source_feature_schema_hash,
        rf_artifact.source_spectral_reasoning_hash,
        rf_artifact.source_copper_channel_battery_hash,
        rf_artifact.source_telecom_recovery_hash,
        rf_artifact.source_satellite_baseline_hash,
        rf_artifact.rf_equalization_hash,
    )


def run_cross_modal_replay_certification(
    rf_artifact: RFEqualizationResult,
    *,
    certification_profile: str = "full_stack",
    score_fixture: tuple[float, ...] | None = None,
    satellite_artifact: SatelliteBaselineResult | None = None,
    telecom_artifact: TelecomRecoveryResult | None = None,
    spectral_artifact: SpectralReasoningResult | None = None,
    schema_artifact: MultimodalFeatureSchemaResult | None = None,
) -> CrossModalReplayCertificationResult:
    """Construct deterministic replay certification artifacts for v137.9.x lineage.

    Optional direct lineage artifacts are used exclusively for validation and never
    mutate upstream identity or produced hashes.

    ``score_fixture`` is a deterministic-readiness validation parameter only (analogous
    to ``float_fixture`` in ``run_rf_equalization``).  It is validated for finiteness
    but is not incorporated into any certification score computation.
    """

    _validate_profile(certification_profile)
    _validate_rf_artifact(rf_artifact)
    _validate_optional_fixture_payload(score_fixture)
    _validate_direct_lineage(
        rf_artifact,
        satellite_artifact=satellite_artifact,
        telecom_artifact=telecom_artifact,
        spectral_artifact=spectral_artifact,
        schema_artifact=schema_artifact,
    )

    replay_certification_id = _sha256_hex(
        {
            "source_rf_equalization_hash": rf_artifact.rf_equalization_hash,
            "cross_modal_replay_certification_version": _CROSS_MODAL_REPLAY_CERTIFICATION_VERSION,
            "certification_profile": certification_profile,
        }
    )

    lineage_chain = _build_lineage_chain(rf_artifact)
    witnesses: list[CrossModalReplayWitness] = []

    for witness_index, segment in enumerate(rf_artifact.segments):
        modal_alignment_score = _mean(
            (
                segment.equalization_integrity_score,
                segment.compensation_stability_score,
                segment.reflection_resilience_score,
                segment.frame_consistency_score,
            ),
            default=1.0,
        )

        frame_identity_score = _mean(
            tuple(
                1.0
                if frame.frame_hash == frame.stable_hash() and frame.frame_id == frame.frame_hash
                else 0.0
                for frame in segment.frames
            ),
            default=1.0,
        )
        replay_identity_score = _mean(
            (
                1.0 if segment.segment_hash == segment.stable_hash() else 0.0,
                frame_identity_score,
                1.0 if rf_artifact.rf_equalization_hash == rf_artifact.stable_hash() else 0.0,
            ),
            default=1.0,
        )

        witness = CrossModalReplayWitness(
            witness_id="",
            witness_index=witness_index,
            certification_profile=certification_profile,
            source_rf_segment_id=segment.segment_id,
            source_rf_frame_count=segment.frame_count,
            lineage_chain=lineage_chain,
            modal_alignment_score=modal_alignment_score,
            replay_identity_score=replay_identity_score,
            witness_integrity_score=_mean((modal_alignment_score, replay_identity_score), default=1.0),
            witness_hash="",
        )
        witness_hash = _finalize_identity(witness, "witness_hash").witness_hash
        witnesses.append(replace(witness, witness_id=witness_hash, witness_hash=witness_hash))

    ordered_witnesses = tuple(
        sorted(
            witnesses,
            key=lambda w: (w.witness_index, w.source_rf_segment_id, w.witness_id),
        )
    )

    ledger = CrossModalReplayLedger(
        ledger_id=_sha256_hex(
            {
                "replay_certification_id": replay_certification_id,
                "lineage_chain": lineage_chain,
                "witness_ids": tuple(w.witness_id for w in ordered_witnesses),
            }
        ),
        certification_profile=certification_profile,
        lineage_chain=lineage_chain,
        witness_count=len(ordered_witnesses),
        witnesses=ordered_witnesses,
        ledger_hash="",
    )
    ledger = _finalize_identity(ledger, "ledger_hash")

    lineage_consistency_score = 1.0
    modal_alignment_score = _mean(tuple(w.modal_alignment_score for w in ledger.witnesses), default=1.0)
    replay_identity_score = _mean(tuple(w.replay_identity_score for w in ledger.witnesses), default=1.0)
    witness_integrity_score = _mean(
        tuple(
            1.0 if witness.witness_hash == witness.stable_hash() and witness.witness_id == witness.witness_hash else 0.0
            for witness in ledger.witnesses
        ),
        default=1.0,
    )
    overall_certification_score = _mean(
        (
            lineage_consistency_score,
            modal_alignment_score,
            replay_identity_score,
            witness_integrity_score,
        ),
        default=1.0,
    )

    for score_name, score in (
        ("lineage_consistency_score", lineage_consistency_score),
        ("modal_alignment_score", modal_alignment_score),
        ("replay_identity_score", replay_identity_score),
        ("witness_integrity_score", witness_integrity_score),
        ("overall_certification_score", overall_certification_score),
    ):
        _validate_unit_interval(score, score_name)

    artifact = CrossModalReplayCertificationResult(
        cross_modal_replay_certification_version=_CROSS_MODAL_REPLAY_CERTIFICATION_VERSION,
        source_feature_schema_hash=rf_artifact.source_feature_schema_hash,
        source_spectral_reasoning_hash=rf_artifact.source_spectral_reasoning_hash,
        source_copper_channel_battery_hash=rf_artifact.source_copper_channel_battery_hash,
        source_telecom_recovery_hash=rf_artifact.source_telecom_recovery_hash,
        source_satellite_baseline_hash=rf_artifact.source_satellite_baseline_hash,
        source_rf_equalization_hash=rf_artifact.rf_equalization_hash,
        replay_certification_id=replay_certification_id,
        certification_profile=certification_profile,
        ledger=ledger,
        lineage_consistency_score=lineage_consistency_score,
        modal_alignment_score=modal_alignment_score,
        replay_identity_score=replay_identity_score,
        witness_integrity_score=witness_integrity_score,
        overall_certification_score=overall_certification_score,
        law_invariants=(
            CROSS_MODAL_REPLAY_CERTIFICATION_LAW,
            DETERMINISTIC_CERTIFICATION_ORDERING_RULE,
            REPLAY_SAFE_CERTIFICATION_IDENTITY_RULE,
            BOUNDED_CERTIFICATION_SCORE_RULE,
        ),
        replay_certification_hash="",
    )
    return _finalize_identity(artifact, "replay_certification_hash")


def export_cross_modal_replay_certification_bytes(artifact: CrossModalReplayCertificationResult) -> bytes:
    if not isinstance(artifact, CrossModalReplayCertificationResult):
        raise ValueError("artifact must be a CrossModalReplayCertificationResult")
    return artifact.to_canonical_bytes()


def generate_cross_modal_replay_certification_receipt(
    artifact: CrossModalReplayCertificationResult,
) -> CrossModalReplayCertificationReceipt:
    if not isinstance(artifact, CrossModalReplayCertificationResult):
        raise ValueError("artifact must be a CrossModalReplayCertificationResult")
    if artifact.stable_hash() != artifact.replay_certification_hash:
        raise ValueError("artifact replay_certification_hash must match stable_hash")

    receipt = CrossModalReplayCertificationReceipt(
        cross_modal_replay_certification_version=artifact.cross_modal_replay_certification_version,
        source_feature_schema_hash=artifact.source_feature_schema_hash,
        source_spectral_reasoning_hash=artifact.source_spectral_reasoning_hash,
        source_copper_channel_battery_hash=artifact.source_copper_channel_battery_hash,
        source_telecom_recovery_hash=artifact.source_telecom_recovery_hash,
        source_satellite_baseline_hash=artifact.source_satellite_baseline_hash,
        source_rf_equalization_hash=artifact.source_rf_equalization_hash,
        replay_certification_id=artifact.replay_certification_id,
        certification_profile=artifact.certification_profile,
        witness_count=artifact.ledger.witness_count,
        overall_certification_score=artifact.overall_certification_score,
        replay_certification_hash=artifact.replay_certification_hash,
        receipt_hash="",
    )
    return _finalize_identity(receipt, "receipt_hash")


__all__ = [
    "BOUNDED_CERTIFICATION_SCORE_RULE",
    "CROSS_MODAL_REPLAY_CERTIFICATION_LAW",
    "DETERMINISTIC_CERTIFICATION_ORDERING_RULE",
    "REPLAY_SAFE_CERTIFICATION_IDENTITY_RULE",
    "CrossModalReplayCertificationReceipt",
    "CrossModalReplayCertificationResult",
    "CrossModalReplayLedger",
    "CrossModalReplayWitness",
    "export_cross_modal_replay_certification_bytes",
    "generate_cross_modal_replay_certification_receipt",
    "run_cross_modal_replay_certification",
]
