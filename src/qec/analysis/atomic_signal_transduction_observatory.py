"""v137.9.7 — Atomic Signal Transduction Observatory.

Deterministic Layer-4 observability consumer of replay certification artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.cross_modal_replay_certification import CrossModalReplayCertificationResult
from qec.analysis.multimodal_feature_schema import MultimodalFeatureSchemaResult
from qec.analysis.rf_equalization_and_ground_station_compensation import RFEqualizationResult
from qec.analysis.satellite_signal_baseline_and_orbital_noise import SatelliteBaselineResult
from qec.analysis.spectral_reasoning_layer import SpectralReasoningResult
from qec.analysis.telecom_line_recovery_and_sync import TelecomRecoveryResult

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_ATOMIC_SIGNAL_TRANSDUCTION_OBSERVATORY_VERSION = 1
_OBSERVATORY_PROFILE_ORDER: tuple[str, ...] = (
    "frame_level_observatory",
    "segment_level_observatory",
    "lineage_transition_observatory",
    "replay_integrity_observatory",
    "end_to_end_signal_window",
)

ATOMIC_SIGNAL_TRANSDUCTION_OBSERVATORY_LAW = "ATOMIC_SIGNAL_TRANSDUCTION_OBSERVATORY_LAW"
DETERMINISTIC_OBSERVATORY_ORDERING_RULE = "DETERMINISTIC_OBSERVATORY_ORDERING_RULE"
REPLAY_SAFE_OBSERVATORY_IDENTITY_RULE = "REPLAY_SAFE_OBSERVATORY_IDENTITY_RULE"
BOUNDED_OBSERVATORY_SCORE_RULE = "BOUNDED_OBSERVATORY_SCORE_RULE"


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


def _validate_profile(observatory_profile: str) -> None:
    if observatory_profile not in _OBSERVATORY_PROFILE_ORDER:
        raise ValueError(
            f"observatory_profile must be one of {_OBSERVATORY_PROFILE_ORDER}; "
            f"received {observatory_profile!r}"
        )


def _validate_replay_certification_artifact(
    certification_artifact: CrossModalReplayCertificationResult,
) -> None:
    if not isinstance(certification_artifact, CrossModalReplayCertificationResult):
        raise ValueError("certification_artifact must be a CrossModalReplayCertificationResult")
    if certification_artifact.stable_hash() != certification_artifact.replay_certification_hash:
        raise ValueError("certification_artifact replay_certification_hash must match stable_hash")

    ledger = certification_artifact.ledger
    if ledger.ledger_hash != ledger.stable_hash():
        raise ValueError("certification_artifact ledger_hash must match stable_hash")
    if ledger.witness_count != len(ledger.witnesses):
        raise ValueError("certification_artifact witness_count must match len(witnesses)")
    if ledger.witness_count <= 0 or not ledger.witnesses:
        raise ValueError("certification_artifact must contain at least one witness")

    expected_witness_order = tuple(
        sorted(
            ledger.witnesses,
            key=lambda witness: (witness.witness_index, witness.source_rf_segment_id, witness.witness_id),
        )
    )
    if ledger.witnesses != expected_witness_order:
        raise ValueError("certification_artifact witnesses must be in canonical deterministic order")

    for score_name, score in (
        ("lineage_consistency_score", certification_artifact.lineage_consistency_score),
        ("modal_alignment_score", certification_artifact.modal_alignment_score),
        ("replay_identity_score", certification_artifact.replay_identity_score),
        ("witness_integrity_score", certification_artifact.witness_integrity_score),
        ("overall_certification_score", certification_artifact.overall_certification_score),
    ):
        _validate_unit_interval(score, f"certification_artifact {score_name}")

    # Verify top-level source_* fields match ledger.lineage_chain to prevent forged provenance:
    # an attacker could mutate source_* fields while leaving the ledger unchanged and recompute
    # replay_certification_hash, creating an artifact that passes hash checks but carries incorrect
    # upstream lineage.
    expected_lineage_chain = (
        certification_artifact.source_feature_schema_hash,
        certification_artifact.source_spectral_reasoning_hash,
        certification_artifact.source_copper_channel_battery_hash,
        certification_artifact.source_telecom_recovery_hash,
        certification_artifact.source_satellite_baseline_hash,
        certification_artifact.source_rf_equalization_hash,
    )
    if ledger.lineage_chain != expected_lineage_chain:
        raise ValueError(
            "certification_artifact source fields are inconsistent with ledger lineage_chain"
        )

    for witness in ledger.witnesses:
        if witness.witness_hash != witness.stable_hash():
            raise ValueError("certification_artifact witness_hash must match stable_hash")
        if witness.witness_id != witness.witness_hash:
            raise ValueError("certification_artifact witness_id must equal witness_hash")
        if witness.lineage_chain != ledger.lineage_chain:
            raise ValueError(
                "certification_artifact witness lineage_chain must match ledger lineage_chain"
            )
        for score_name, score in (
            ("modal_alignment_score", witness.modal_alignment_score),
            ("replay_identity_score", witness.replay_identity_score),
            ("witness_integrity_score", witness.witness_integrity_score),
        ):
            _validate_unit_interval(score, f"certification_artifact witness {score_name}")


def _validate_direct_lineage(
    certification_artifact: CrossModalReplayCertificationResult,
    *,
    rf_artifact: RFEqualizationResult | None,
    satellite_artifact: SatelliteBaselineResult | None,
    telecom_artifact: TelecomRecoveryResult | None,
    spectral_artifact: SpectralReasoningResult | None,
    schema_artifact: MultimodalFeatureSchemaResult | None,
) -> None:
    if rf_artifact is not None:
        if not isinstance(rf_artifact, RFEqualizationResult):
            raise ValueError("rf_artifact must be a RFEqualizationResult")
        if rf_artifact.stable_hash() != rf_artifact.rf_equalization_hash:
            raise ValueError("rf_artifact rf_equalization_hash must match stable_hash")
        if rf_artifact.rf_equalization_hash != certification_artifact.source_rf_equalization_hash:
            raise ValueError("direct lineage mismatch: rf_equalization_hash")

    if satellite_artifact is not None:
        if not isinstance(satellite_artifact, SatelliteBaselineResult):
            raise ValueError("satellite_artifact must be a SatelliteBaselineResult")
        if satellite_artifact.stable_hash() != satellite_artifact.satellite_baseline_hash:
            raise ValueError("satellite_artifact satellite_baseline_hash must match stable_hash")
        if satellite_artifact.satellite_baseline_hash != certification_artifact.source_satellite_baseline_hash:
            raise ValueError("direct lineage mismatch: satellite_baseline_hash")

    if telecom_artifact is not None:
        if not isinstance(telecom_artifact, TelecomRecoveryResult):
            raise ValueError("telecom_artifact must be a TelecomRecoveryResult")
        if telecom_artifact.stable_hash() != telecom_artifact.telecom_recovery_hash:
            raise ValueError("telecom_artifact telecom_recovery_hash must match stable_hash")
        if telecom_artifact.telecom_recovery_hash != certification_artifact.source_telecom_recovery_hash:
            raise ValueError("direct lineage mismatch: telecom_recovery_hash")

    if spectral_artifact is not None:
        if not isinstance(spectral_artifact, SpectralReasoningResult):
            raise ValueError("spectral_artifact must be a SpectralReasoningResult")
        if spectral_artifact.stable_hash() != spectral_artifact.spectral_reasoning_hash:
            raise ValueError("spectral_artifact spectral_reasoning_hash must match stable_hash")
        if spectral_artifact.spectral_reasoning_hash != certification_artifact.source_spectral_reasoning_hash:
            raise ValueError("direct lineage mismatch: spectral_reasoning_hash")

    if schema_artifact is not None:
        if not isinstance(schema_artifact, MultimodalFeatureSchemaResult):
            raise ValueError("schema_artifact must be a MultimodalFeatureSchemaResult")
        if schema_artifact.stable_hash() != schema_artifact.feature_schema_hash:
            raise ValueError("schema_artifact feature_schema_hash must match stable_hash")
        if schema_artifact.feature_schema_hash != certification_artifact.source_feature_schema_hash:
            raise ValueError("direct lineage mismatch: feature_schema_hash")


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
class AtomicSignalObservation:
    observation_id: str
    observation_index: int
    observatory_profile: str
    source_witness_id: str
    source_segment_id: str
    lineage_chain: tuple[str, ...]
    transduction_integrity_score: float
    observation_alignment_score: float
    replay_visibility_score: float
    observation_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "observation_id": self.observation_id,
            "observation_index": self.observation_index,
            "observatory_profile": self.observatory_profile,
            "source_witness_id": self.source_witness_id,
            "source_segment_id": self.source_segment_id,
            "lineage_chain": self.lineage_chain,
            "transduction_integrity_score": self.transduction_integrity_score,
            "observation_alignment_score": self.observation_alignment_score,
            "replay_visibility_score": self.replay_visibility_score,
            "observation_hash": self.observation_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("observation_id")
        payload.pop("observation_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class AtomicSignalWindow:
    window_id: str
    window_index: int
    observatory_profile: str
    observation_ids: tuple[str, ...]
    lineage_chain: tuple[str, ...]
    transduction_integrity_score: float
    window_consistency_score: float
    replay_visibility_score: float
    window_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "window_id": self.window_id,
            "window_index": self.window_index,
            "observatory_profile": self.observatory_profile,
            "observation_ids": self.observation_ids,
            "lineage_chain": self.lineage_chain,
            "transduction_integrity_score": self.transduction_integrity_score,
            "window_consistency_score": self.window_consistency_score,
            "replay_visibility_score": self.replay_visibility_score,
            "window_hash": self.window_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("window_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class AtomicSignalTransductionObservatoryResult:
    atomic_signal_transduction_observatory_version: int
    source_feature_schema_hash: str
    source_spectral_reasoning_hash: str
    source_copper_channel_battery_hash: str
    source_telecom_recovery_hash: str
    source_satellite_baseline_hash: str
    source_rf_equalization_hash: str
    source_replay_certification_hash: str
    atomic_signal_observatory_id: str
    observatory_profile: str
    observations: tuple[AtomicSignalObservation, ...]
    observation_count: int
    windows: tuple[AtomicSignalWindow, ...]
    window_count: int
    transduction_integrity_score: float
    window_consistency_score: float
    observation_alignment_score: float
    replay_visibility_score: float
    overall_observatory_score: float
    law_invariants: tuple[str, ...]
    atomic_signal_observatory_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "atomic_signal_transduction_observatory_version": self.atomic_signal_transduction_observatory_version,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "source_spectral_reasoning_hash": self.source_spectral_reasoning_hash,
            "source_copper_channel_battery_hash": self.source_copper_channel_battery_hash,
            "source_telecom_recovery_hash": self.source_telecom_recovery_hash,
            "source_satellite_baseline_hash": self.source_satellite_baseline_hash,
            "source_rf_equalization_hash": self.source_rf_equalization_hash,
            "source_replay_certification_hash": self.source_replay_certification_hash,
            "atomic_signal_observatory_id": self.atomic_signal_observatory_id,
            "observatory_profile": self.observatory_profile,
            "observations": tuple(observation.to_dict() for observation in self.observations),
            "observation_count": self.observation_count,
            "windows": tuple(window.to_dict() for window in self.windows),
            "window_count": self.window_count,
            "transduction_integrity_score": self.transduction_integrity_score,
            "window_consistency_score": self.window_consistency_score,
            "observation_alignment_score": self.observation_alignment_score,
            "replay_visibility_score": self.replay_visibility_score,
            "overall_observatory_score": self.overall_observatory_score,
            "law_invariants": self.law_invariants,
            "atomic_signal_observatory_hash": self.atomic_signal_observatory_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("atomic_signal_observatory_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class AtomicSignalObservatoryReceipt:
    atomic_signal_transduction_observatory_version: int
    source_feature_schema_hash: str
    source_spectral_reasoning_hash: str
    source_copper_channel_battery_hash: str
    source_telecom_recovery_hash: str
    source_satellite_baseline_hash: str
    source_rf_equalization_hash: str
    source_replay_certification_hash: str
    atomic_signal_observatory_id: str
    observatory_profile: str
    observation_count: int
    window_count: int
    overall_observatory_score: float
    atomic_signal_observatory_hash: str
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "atomic_signal_transduction_observatory_version": self.atomic_signal_transduction_observatory_version,
            "source_feature_schema_hash": self.source_feature_schema_hash,
            "source_spectral_reasoning_hash": self.source_spectral_reasoning_hash,
            "source_copper_channel_battery_hash": self.source_copper_channel_battery_hash,
            "source_telecom_recovery_hash": self.source_telecom_recovery_hash,
            "source_satellite_baseline_hash": self.source_satellite_baseline_hash,
            "source_rf_equalization_hash": self.source_rf_equalization_hash,
            "source_replay_certification_hash": self.source_replay_certification_hash,
            "atomic_signal_observatory_id": self.atomic_signal_observatory_id,
            "observatory_profile": self.observatory_profile,
            "observation_count": self.observation_count,
            "window_count": self.window_count,
            "overall_observatory_score": self.overall_observatory_score,
            "atomic_signal_observatory_hash": self.atomic_signal_observatory_hash,
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


def _lineage_chain(certification_artifact: CrossModalReplayCertificationResult) -> tuple[str, ...]:
    return (
        certification_artifact.source_feature_schema_hash,
        certification_artifact.source_spectral_reasoning_hash,
        certification_artifact.source_copper_channel_battery_hash,
        certification_artifact.source_telecom_recovery_hash,
        certification_artifact.source_satellite_baseline_hash,
        certification_artifact.source_rf_equalization_hash,
        certification_artifact.replay_certification_hash,
    )


def _build_observations(
    certification_artifact: CrossModalReplayCertificationResult,
    *,
    observatory_profile: str,
    lineage_chain: tuple[str, ...],
) -> tuple[AtomicSignalObservation, ...]:
    observations: list[AtomicSignalObservation] = []
    witnesses = certification_artifact.ledger.witnesses
    for observation_index, profile_name in enumerate(_OBSERVATORY_PROFILE_ORDER):
        witness = witnesses[observation_index % len(witnesses)]
        transduction_integrity_score = _mean(
            (
                witness.witness_integrity_score,
                certification_artifact.witness_integrity_score,
                certification_artifact.overall_certification_score,
            ),
            default=1.0,
        )
        observation_alignment_score = _mean(
            (
                witness.modal_alignment_score,
                certification_artifact.modal_alignment_score,
                certification_artifact.lineage_consistency_score,
            ),
            default=1.0,
        )
        replay_visibility_score = _mean(
            (
                witness.replay_identity_score,
                certification_artifact.replay_identity_score,
                1.0 if certification_artifact.replay_certification_hash == certification_artifact.stable_hash() else 0.0,
            ),
            default=1.0,
        )

        observation = AtomicSignalObservation(
            observation_id="",
            observation_index=observation_index,
            observatory_profile=observatory_profile,
            source_witness_id=witness.witness_id,
            source_segment_id=witness.source_rf_segment_id,
            lineage_chain=lineage_chain,
            transduction_integrity_score=transduction_integrity_score,
            observation_alignment_score=observation_alignment_score,
            replay_visibility_score=replay_visibility_score,
            observation_hash="",
        )
        observation_hash = _finalize_identity(observation, "observation_hash").observation_hash
        observations.append(replace(observation, observation_id=observation_hash, observation_hash=observation_hash))

    return tuple(
        sorted(
            observations,
            key=lambda observation: (observation.observation_index, observation.observatory_profile, observation.observation_id),
        )
    )


def _build_windows(
    observations: tuple[AtomicSignalObservation, ...],
    *,
    observatory_profile: str,
    lineage_chain: tuple[str, ...],
) -> tuple[AtomicSignalWindow, ...]:
    windows: list[AtomicSignalWindow] = []
    for window_index in range(len(observations)):
        left = observations[window_index]
        right = observations[(window_index + 1) % len(observations)]
        observation_ids = (left.observation_id, right.observation_id)

        transduction_integrity_score = _mean(
            (left.transduction_integrity_score, right.transduction_integrity_score),
            default=1.0,
        )
        replay_visibility_score = _mean(
            (left.replay_visibility_score, right.replay_visibility_score),
            default=1.0,
        )
        window_consistency_score = _mean(
            (
                transduction_integrity_score,
                replay_visibility_score,
                1.0 if left.lineage_chain == right.lineage_chain == lineage_chain else 0.0,
            ),
            default=1.0,
        )

        window = AtomicSignalWindow(
            window_id=_sha256_hex(
                {
                    "window_index": window_index,
                    "observatory_profile": observatory_profile,
                    "observation_ids": observation_ids,
                    "lineage_chain": lineage_chain,
                }
            ),
            window_index=window_index,
            observatory_profile=observatory_profile,
            observation_ids=observation_ids,
            lineage_chain=lineage_chain,
            transduction_integrity_score=transduction_integrity_score,
            window_consistency_score=window_consistency_score,
            replay_visibility_score=replay_visibility_score,
            window_hash="",
        )
        windows.append(_finalize_identity(window, "window_hash"))

    return tuple(
        sorted(
            windows,
            key=lambda window: (window.window_index, window.window_id, window.window_hash),
        )
    )


def run_atomic_signal_transduction_observatory(
    certification_artifact: CrossModalReplayCertificationResult,
    *,
    observatory_profile: str = "end_to_end_signal_window",
    score_fixture: tuple[float, ...] | None = None,
    rf_artifact: RFEqualizationResult | None = None,
    satellite_artifact: SatelliteBaselineResult | None = None,
    telecom_artifact: TelecomRecoveryResult | None = None,
    spectral_artifact: SpectralReasoningResult | None = None,
    schema_artifact: MultimodalFeatureSchemaResult | None = None,
) -> AtomicSignalTransductionObservatoryResult:
    """Build deterministic replay-safe atomic signal observatory artifacts.

    Optional direct lineage artifacts are used exclusively for validation and never
    mutate upstream identity or produced hashes.

    ``score_fixture`` is a deterministic-readiness validation parameter only (analogous
    to ``score_fixture`` in ``run_cross_modal_replay_certification``).  It is validated
    for finiteness but is not incorporated into any observatory score computation.
    """

    _validate_profile(observatory_profile)
    _validate_replay_certification_artifact(certification_artifact)
    _validate_optional_fixture_payload(score_fixture)
    _validate_direct_lineage(
        certification_artifact,
        rf_artifact=rf_artifact,
        satellite_artifact=satellite_artifact,
        telecom_artifact=telecom_artifact,
        spectral_artifact=spectral_artifact,
        schema_artifact=schema_artifact,
    )

    lineage_chain = _lineage_chain(certification_artifact)
    observations = _build_observations(
        certification_artifact,
        observatory_profile=observatory_profile,
        lineage_chain=lineage_chain,
    )
    windows = _build_windows(observations, observatory_profile=observatory_profile, lineage_chain=lineage_chain)

    transduction_integrity_score = _mean(tuple(o.transduction_integrity_score for o in observations), default=1.0)
    window_consistency_score = _mean(tuple(window.window_consistency_score for window in windows), default=1.0)
    observation_alignment_score = _mean(tuple(o.observation_alignment_score for o in observations), default=1.0)
    replay_visibility_score = _mean(
        (
            certification_artifact.replay_identity_score,
            _mean(tuple(o.replay_visibility_score for o in observations), default=1.0),
            _mean(tuple(window.replay_visibility_score for window in windows), default=1.0),
        ),
        default=1.0,
    )
    overall_observatory_score = _mean(
        (
            transduction_integrity_score,
            window_consistency_score,
            observation_alignment_score,
            replay_visibility_score,
        ),
        default=1.0,
    )

    for score_name, score in (
        ("transduction_integrity_score", transduction_integrity_score),
        ("window_consistency_score", window_consistency_score),
        ("observation_alignment_score", observation_alignment_score),
        ("replay_visibility_score", replay_visibility_score),
        ("overall_observatory_score", overall_observatory_score),
    ):
        _validate_unit_interval(score, score_name)

    atomic_signal_observatory_id = _sha256_hex(
        {
            "source_replay_certification_hash": certification_artifact.replay_certification_hash,
            "atomic_signal_transduction_observatory_version": _ATOMIC_SIGNAL_TRANSDUCTION_OBSERVATORY_VERSION,
            "observatory_profile": observatory_profile,
        }
    )

    artifact = AtomicSignalTransductionObservatoryResult(
        atomic_signal_transduction_observatory_version=_ATOMIC_SIGNAL_TRANSDUCTION_OBSERVATORY_VERSION,
        source_feature_schema_hash=certification_artifact.source_feature_schema_hash,
        source_spectral_reasoning_hash=certification_artifact.source_spectral_reasoning_hash,
        source_copper_channel_battery_hash=certification_artifact.source_copper_channel_battery_hash,
        source_telecom_recovery_hash=certification_artifact.source_telecom_recovery_hash,
        source_satellite_baseline_hash=certification_artifact.source_satellite_baseline_hash,
        source_rf_equalization_hash=certification_artifact.source_rf_equalization_hash,
        source_replay_certification_hash=certification_artifact.replay_certification_hash,
        atomic_signal_observatory_id=atomic_signal_observatory_id,
        observatory_profile=observatory_profile,
        observations=observations,
        observation_count=len(observations),
        windows=windows,
        window_count=len(windows),
        transduction_integrity_score=transduction_integrity_score,
        window_consistency_score=window_consistency_score,
        observation_alignment_score=observation_alignment_score,
        replay_visibility_score=replay_visibility_score,
        overall_observatory_score=overall_observatory_score,
        law_invariants=(
            ATOMIC_SIGNAL_TRANSDUCTION_OBSERVATORY_LAW,
            DETERMINISTIC_OBSERVATORY_ORDERING_RULE,
            REPLAY_SAFE_OBSERVATORY_IDENTITY_RULE,
            BOUNDED_OBSERVATORY_SCORE_RULE,
        ),
        atomic_signal_observatory_hash="",
    )
    return _finalize_identity(artifact, "atomic_signal_observatory_hash")


def export_atomic_signal_observatory_bytes(artifact: AtomicSignalTransductionObservatoryResult) -> bytes:
    if not isinstance(artifact, AtomicSignalTransductionObservatoryResult):
        raise ValueError("artifact must be an AtomicSignalTransductionObservatoryResult")
    return artifact.to_canonical_bytes()


def generate_atomic_signal_observatory_receipt(
    artifact: AtomicSignalTransductionObservatoryResult,
) -> AtomicSignalObservatoryReceipt:
    if not isinstance(artifact, AtomicSignalTransductionObservatoryResult):
        raise ValueError("artifact must be an AtomicSignalTransductionObservatoryResult")
    if artifact.stable_hash() != artifact.atomic_signal_observatory_hash:
        raise ValueError("artifact atomic_signal_observatory_hash must match stable_hash")

    receipt = AtomicSignalObservatoryReceipt(
        atomic_signal_transduction_observatory_version=artifact.atomic_signal_transduction_observatory_version,
        source_feature_schema_hash=artifact.source_feature_schema_hash,
        source_spectral_reasoning_hash=artifact.source_spectral_reasoning_hash,
        source_copper_channel_battery_hash=artifact.source_copper_channel_battery_hash,
        source_telecom_recovery_hash=artifact.source_telecom_recovery_hash,
        source_satellite_baseline_hash=artifact.source_satellite_baseline_hash,
        source_rf_equalization_hash=artifact.source_rf_equalization_hash,
        source_replay_certification_hash=artifact.source_replay_certification_hash,
        atomic_signal_observatory_id=artifact.atomic_signal_observatory_id,
        observatory_profile=artifact.observatory_profile,
        observation_count=artifact.observation_count,
        window_count=artifact.window_count,
        overall_observatory_score=artifact.overall_observatory_score,
        atomic_signal_observatory_hash=artifact.atomic_signal_observatory_hash,
        receipt_hash="",
    )
    return _finalize_identity(receipt, "receipt_hash")


__all__ = [
    "ATOMIC_SIGNAL_TRANSDUCTION_OBSERVATORY_LAW",
    "BOUNDED_OBSERVATORY_SCORE_RULE",
    "DETERMINISTIC_OBSERVATORY_ORDERING_RULE",
    "REPLAY_SAFE_OBSERVATORY_IDENTITY_RULE",
    "AtomicSignalObservation",
    "AtomicSignalObservatoryReceipt",
    "AtomicSignalTransductionObservatoryResult",
    "AtomicSignalWindow",
    "export_atomic_signal_observatory_bytes",
    "generate_atomic_signal_observatory_receipt",
    "run_atomic_signal_transduction_observatory",
]
