"""v137.7.3 — Fragmentation Recovery Engine.

Deterministic Layer-4 recovery for fragmented compressed-memory chains.

Mandatory invariants encoded by this module:
- FRAGMENTATION_RECOVERY_LAW
- DETERMINISTIC_CHAIN_REPAIR_RULE
- REPLAY_SAFE_RECOVERY_CHAIN
- BOUNDED_FRAGMENT_RECOVERY_SCORE
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.hash_preserving_memory_compression import CompressedMemoryArtifact, CompressionRecord
from qec.analysis.semantic_memory_sonification import SonificationProjectionSpec

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1
_MIN_HASH_HEX_LEN = 8

FRAGMENTATION_RECOVERY_LAW = "FRAGMENTATION_RECOVERY_LAW"
DETERMINISTIC_CHAIN_REPAIR_RULE = "DETERMINISTIC_CHAIN_REPAIR_RULE"
REPLAY_SAFE_RECOVERY_CHAIN = "REPLAY_SAFE_RECOVERY_CHAIN"
BOUNDED_FRAGMENT_RECOVERY_SCORE = "BOUNDED_FRAGMENT_RECOVERY_SCORE"


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


def _validate_non_empty_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = value.strip()
    if stripped == "":
        raise ValueError(f"{field_name} must be non-empty")
    return stripped


def _validate_hex_hash(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain leading or trailing whitespace")
    if len(value) < _MIN_HASH_HEX_LEN:
        raise ValueError(
            f"{field_name} must be at least {_MIN_HASH_HEX_LEN} hex characters, got {len(value)}"
        )
    try:
        int(value, 16)
    except ValueError:
        raise ValueError(f"{field_name} must be a valid hex string, got {value!r}") from None
    return value


def _clamp_score(value: float) -> float:
    if not isinstance(value, float):
        raise ValueError("score must be float")
    if not math.isfinite(value):
        raise ValueError("score must be finite")
    return min(1.0, max(0.0, value))


def _validate_compression_record(record: CompressionRecord, *, field_prefix: str = "record") -> None:
    if not isinstance(record, CompressionRecord):
        raise ValueError(f"{field_prefix} must be a CompressionRecord")
    _validate_non_empty_str(record.theme_id, field_name=f"{field_prefix}.theme_id")
    if not isinstance(record.theme_index, int):
        raise ValueError(f"{field_prefix}.theme_index must be int")
    _validate_hex_hash(record.source_theme_hash, field_name=f"{field_prefix}.source_theme_hash")
    _validate_hex_hash(
        record.source_replay_identity_hash,
        field_name=f"{field_prefix}.source_replay_identity_hash",
    )
    _validate_hex_hash(
        record.source_parent_theme_hash,
        field_name=f"{field_prefix}.source_parent_theme_hash",
    )
    _validate_hex_hash(record.compression_record_hash, field_name=f"{field_prefix}.compression_record_hash")


def _validate_source_artifact(artifact: CompressedMemoryArtifact) -> tuple[CompressionRecord, ...]:
    if not isinstance(artifact, CompressedMemoryArtifact):
        raise ValueError("source_artifact must be a CompressedMemoryArtifact")
    _validate_hex_hash(artifact.compression_hash, field_name="source_artifact.compression_hash")
    if artifact.compressed_record_count != len(artifact.records):
        raise ValueError("source_artifact compressed_record_count must match records length")
    if artifact.source_theme_count != len(artifact.records):
        raise ValueError("source_artifact source_theme_count must match records length")

    records = artifact.records
    if not records:
        raise ValueError("source_artifact must contain at least one CompressionRecord")
    for idx, record in enumerate(records):
        _validate_compression_record(record, field_prefix=f"source_artifact.records[{idx}]")
        if record.theme_index != idx:
            raise ValueError("source_artifact records must be contiguous in theme_index")
        if idx > 0 and record.source_parent_theme_hash != records[idx - 1].source_replay_identity_hash:
            raise ValueError("source_artifact contains unrecoverable lineage corruption")

    if artifact.compression_chain_head != records[-1].source_replay_identity_hash:
        raise ValueError("source_artifact compression_chain_head must match last replay identity")
    return records


@dataclass(frozen=True)
class RepairedLineageReport:
    source_record_count: int
    observed_record_count: int
    repaired_record_count: int
    missing_theme_indices: tuple[int, ...]
    disordered_theme_indices: tuple[int, ...]
    corrupted_theme_indices: tuple[int, ...]
    recoverable: bool
    classification: str
    preserved_hash_count: int
    report_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "source_record_count": self.source_record_count,
            "observed_record_count": self.observed_record_count,
            "repaired_record_count": self.repaired_record_count,
            "missing_theme_indices": self.missing_theme_indices,
            "disordered_theme_indices": self.disordered_theme_indices,
            "corrupted_theme_indices": self.corrupted_theme_indices,
            "recoverable": self.recoverable,
            "classification": self.classification,
            "preserved_hash_count": self.preserved_hash_count,
            "report_hash": self.report_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "source_record_count": self.source_record_count,
            "observed_record_count": self.observed_record_count,
            "repaired_record_count": self.repaired_record_count,
            "missing_theme_indices": self.missing_theme_indices,
            "disordered_theme_indices": self.disordered_theme_indices,
            "corrupted_theme_indices": self.corrupted_theme_indices,
            "recoverable": self.recoverable,
            "classification": self.classification,
            "preserved_hash_count": self.preserved_hash_count,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class SonificationContinuityRepairMetadata:
    source_compression_hash: str
    source_sonification_spec_hash: str
    preserved_audio_projection_hash: str
    continuity_repaired: bool
    sonification_lineage_preserved: bool
    metadata_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "source_compression_hash": self.source_compression_hash,
            "source_sonification_spec_hash": self.source_sonification_spec_hash,
            "preserved_audio_projection_hash": self.preserved_audio_projection_hash,
            "continuity_repaired": self.continuity_repaired,
            "sonification_lineage_preserved": self.sonification_lineage_preserved,
            "metadata_hash": self.metadata_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "source_compression_hash": self.source_compression_hash,
            "source_sonification_spec_hash": self.source_sonification_spec_hash,
            "preserved_audio_projection_hash": self.preserved_audio_projection_hash,
            "continuity_repaired": self.continuity_repaired,
            "sonification_lineage_preserved": self.sonification_lineage_preserved,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class FragmentationRecoveryArtifact:
    schema_version: int
    source_compression_hash: str
    source_replay_identity_hash: str
    repaired_chain_head: str
    repaired_records: tuple[CompressionRecord, ...]
    preserved_theme_hashes: tuple[str, ...]
    continuity_score: float
    recovery_confidence: float
    fracture_severity: float
    lineage_report: RepairedLineageReport
    sonification_repair_metadata: SonificationContinuityRepairMetadata | None
    replay_safe: bool
    law_invariants: tuple[str, ...]
    recovery_artifact_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_compression_hash": self.source_compression_hash,
            "source_replay_identity_hash": self.source_replay_identity_hash,
            "repaired_chain_head": self.repaired_chain_head,
            "repaired_records": tuple(record.to_dict() for record in self.repaired_records),
            "preserved_theme_hashes": self.preserved_theme_hashes,
            "continuity_score": self.continuity_score,
            "recovery_confidence": self.recovery_confidence,
            "fracture_severity": self.fracture_severity,
            "lineage_report": self.lineage_report.to_dict(),
            "sonification_repair_metadata": (
                None if self.sonification_repair_metadata is None else self.sonification_repair_metadata.to_dict()
            ),
            "replay_safe": self.replay_safe,
            "law_invariants": self.law_invariants,
            "recovery_artifact_hash": self.recovery_artifact_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_compression_hash": self.source_compression_hash,
            "source_replay_identity_hash": self.source_replay_identity_hash,
            "repaired_chain_head": self.repaired_chain_head,
            "repaired_records": tuple(record.to_dict() for record in self.repaired_records),
            "preserved_theme_hashes": self.preserved_theme_hashes,
            "continuity_score": self.continuity_score,
            "recovery_confidence": self.recovery_confidence,
            "fracture_severity": self.fracture_severity,
            "lineage_report": self.lineage_report.to_dict(),
            "sonification_repair_metadata": (
                None if self.sonification_repair_metadata is None else self.sonification_repair_metadata.to_dict()
            ),
            "replay_safe": self.replay_safe,
            "law_invariants": self.law_invariants,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class FragmentationRecoveryReceipt:
    schema_version: int
    source_compression_hash: str
    recovery_artifact_hash: str
    repaired_chain_head: str
    replay_safe: bool
    classification: str
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_compression_hash": self.source_compression_hash,
            "recovery_artifact_hash": self.recovery_artifact_hash,
            "repaired_chain_head": self.repaired_chain_head,
            "replay_safe": self.replay_safe,
            "classification": self.classification,
            "receipt_hash": self.receipt_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_compression_hash": self.source_compression_hash,
            "recovery_artifact_hash": self.recovery_artifact_hash,
            "repaired_chain_head": self.repaired_chain_head,
            "replay_safe": self.replay_safe,
            "classification": self.classification,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


def recover_fragmented_compression_chain(
    source_artifact: CompressedMemoryArtifact,
    *,
    observed_records: tuple[CompressionRecord, ...] | None = None,
    sonification_spec: SonificationProjectionSpec | None = None,
    enable_fragmentation_recovery: bool = False,
) -> FragmentationRecoveryArtifact:
    """Perform deterministic, replay-safe continuity repair for fragmented chains.

    FRAGMENTATION_RECOVERY_LAW: only restore structure from deterministic source evidence.
    DETERMINISTIC_CHAIN_REPAIR_RULE: same source+observations yields byte-identical output.
    REPLAY_SAFE_RECOVERY_CHAIN: repaired lineage remains hash-addressable and canonical.
    BOUNDED_FRAGMENT_RECOVERY_SCORE: continuity_score/recovery_confidence/fracture_severity
    are clamped to [0.0, 1.0].
    """

    if enable_fragmentation_recovery is not True:
        raise ValueError("enable_fragmentation_recovery must be explicitly True")

    source_records = _validate_source_artifact(source_artifact)
    observations = source_records if observed_records is None else tuple(observed_records)

    indexed_observations: list[tuple[int, CompressionRecord]] = []
    observed_theme_indices_in_order: list[int] = []
    for original_pos, record in enumerate(observations):
        _validate_compression_record(record, field_prefix=f"observed_records[{original_pos}]")
        if record.theme_index < 0 or record.theme_index >= len(source_records):
            raise ValueError("observed_records theme_index out of source chain bounds")
        indexed_observations.append((original_pos, record))
        observed_theme_indices_in_order.append(record.theme_index)

    disordered_set: set[int] = set()
    previous_seen_index = -1
    for current_index in observed_theme_indices_in_order:
        if current_index < previous_seen_index:
            disordered_set.add(current_index)
        if current_index > previous_seen_index:
            previous_seen_index = current_index
    disordered_indices = tuple(sorted(disordered_set))

    by_index: dict[int, list[CompressionRecord]] = {}
    for _, obs in indexed_observations:
        by_index.setdefault(obs.theme_index, []).append(obs)

    repaired_records: list[CompressionRecord] = []
    missing_indices: list[int] = []
    corrupted_indices: list[int] = []
    preserved_hash_count = 0

    for expected in source_records:
        candidates = by_index.get(expected.theme_index, [])
        exact_matches = [candidate for candidate in candidates if candidate == expected]
        if exact_matches:
            chosen = min(
                exact_matches,
                key=lambda c: (
                    c.compression_record_hash,
                    c.source_replay_identity_hash,
                    c.theme_id,
                ),
            )
            repaired_records.append(chosen)
            preserved_hash_count += 1
            continue

        if candidates:
            corrupted_indices.append(expected.theme_index)

        missing_indices.append(expected.theme_index)
        repaired_records.append(expected)

    repaired_chain = tuple(repaired_records)
    fractured_theme_indices = sorted(
        set(missing_indices) | set(disordered_indices) | set(corrupted_indices)
    )
    fracture_count = len(fractured_theme_indices)
    denominator = max(1, len(source_records))
    fracture_severity = _clamp_score(float(fracture_count / denominator))
    continuity_score = _clamp_score(float(1.0 - fracture_severity))

    classification = "no_op"
    if len(corrupted_indices) > 0 and len(missing_indices) == len(source_records):
        classification = "unrecoverable_corruption"
    elif len(missing_indices) > 0 or len(disordered_indices) > 0:
        classification = "partial_fragment_repair"

    recoverable = classification != "unrecoverable_corruption"
    recovery_confidence = _clamp_score(1.0 if recoverable else 0.0)
    replay_safe = repaired_chain == source_records

    lineage_report = RepairedLineageReport(
        source_record_count=len(source_records),
        observed_record_count=len(observations),
        repaired_record_count=len(repaired_chain),
        missing_theme_indices=tuple(missing_indices),
        disordered_theme_indices=disordered_indices,
        corrupted_theme_indices=tuple(sorted(set(corrupted_indices))),
        recoverable=recoverable,
        classification=classification,
        preserved_hash_count=preserved_hash_count,
        report_hash="",
    )
    lineage_report = RepairedLineageReport(
        source_record_count=lineage_report.source_record_count,
        observed_record_count=lineage_report.observed_record_count,
        repaired_record_count=lineage_report.repaired_record_count,
        missing_theme_indices=lineage_report.missing_theme_indices,
        disordered_theme_indices=lineage_report.disordered_theme_indices,
        corrupted_theme_indices=lineage_report.corrupted_theme_indices,
        recoverable=lineage_report.recoverable,
        classification=lineage_report.classification,
        preserved_hash_count=lineage_report.preserved_hash_count,
        report_hash=_sha256_hex(lineage_report.to_hash_payload_dict()),
    )

    sonification_metadata: SonificationContinuityRepairMetadata | None = None
    if sonification_spec is not None:
        if sonification_spec.source_compression_hash != source_artifact.compression_hash:
            raise ValueError("sonification_spec source_compression_hash must match source_artifact")
        sonification_metadata = SonificationContinuityRepairMetadata(
            source_compression_hash=source_artifact.compression_hash,
            source_sonification_spec_hash=sonification_spec.sonification_spec_hash,
            preserved_audio_projection_hash=sonification_spec.audio_projection_hash,
            continuity_repaired=classification != "no_op",
            sonification_lineage_preserved=replay_safe,
            metadata_hash="",
        )
        sonification_metadata = SonificationContinuityRepairMetadata(
            source_compression_hash=sonification_metadata.source_compression_hash,
            source_sonification_spec_hash=sonification_metadata.source_sonification_spec_hash,
            preserved_audio_projection_hash=sonification_metadata.preserved_audio_projection_hash,
            continuity_repaired=sonification_metadata.continuity_repaired,
            sonification_lineage_preserved=sonification_metadata.sonification_lineage_preserved,
            metadata_hash=_sha256_hex(sonification_metadata.to_hash_payload_dict()),
        )

    law_invariants = (
        FRAGMENTATION_RECOVERY_LAW,
        DETERMINISTIC_CHAIN_REPAIR_RULE,
        REPLAY_SAFE_RECOVERY_CHAIN,
        BOUNDED_FRAGMENT_RECOVERY_SCORE,
    )

    artifact = FragmentationRecoveryArtifact(
        schema_version=_SCHEMA_VERSION,
        source_compression_hash=source_artifact.compression_hash,
        source_replay_identity_hash=source_artifact.replay_identity_hash,
        repaired_chain_head=repaired_chain[-1].source_replay_identity_hash,
        repaired_records=repaired_chain,
        preserved_theme_hashes=source_artifact.preserved_theme_hashes,
        continuity_score=continuity_score,
        recovery_confidence=recovery_confidence,
        fracture_severity=fracture_severity,
        lineage_report=lineage_report,
        sonification_repair_metadata=sonification_metadata,
        replay_safe=replay_safe,
        law_invariants=law_invariants,
        recovery_artifact_hash="",
    )
    return FragmentationRecoveryArtifact(
        schema_version=artifact.schema_version,
        source_compression_hash=artifact.source_compression_hash,
        source_replay_identity_hash=artifact.source_replay_identity_hash,
        repaired_chain_head=artifact.repaired_chain_head,
        repaired_records=artifact.repaired_records,
        preserved_theme_hashes=artifact.preserved_theme_hashes,
        continuity_score=artifact.continuity_score,
        recovery_confidence=artifact.recovery_confidence,
        fracture_severity=artifact.fracture_severity,
        lineage_report=artifact.lineage_report,
        sonification_repair_metadata=artifact.sonification_repair_metadata,
        replay_safe=artifact.replay_safe,
        law_invariants=artifact.law_invariants,
        recovery_artifact_hash=_sha256_hex(artifact.to_hash_payload_dict()),
    )


def generate_fragmentation_recovery_receipt(
    artifact: FragmentationRecoveryArtifact,
) -> FragmentationRecoveryReceipt:
    if not isinstance(artifact, FragmentationRecoveryArtifact):
        raise ValueError("artifact must be a FragmentationRecoveryArtifact")
    receipt = FragmentationRecoveryReceipt(
        schema_version=artifact.schema_version,
        source_compression_hash=artifact.source_compression_hash,
        recovery_artifact_hash=artifact.recovery_artifact_hash,
        repaired_chain_head=artifact.repaired_chain_head,
        replay_safe=artifact.replay_safe,
        classification=artifact.lineage_report.classification,
        receipt_hash="",
    )
    return FragmentationRecoveryReceipt(
        schema_version=receipt.schema_version,
        source_compression_hash=receipt.source_compression_hash,
        recovery_artifact_hash=receipt.recovery_artifact_hash,
        repaired_chain_head=receipt.repaired_chain_head,
        replay_safe=receipt.replay_safe,
        classification=receipt.classification,
        receipt_hash=_sha256_hex(receipt.to_hash_payload_dict()),
    )


def export_fragmentation_recovery_bytes(artifact: FragmentationRecoveryArtifact) -> bytes:
    if not isinstance(artifact, FragmentationRecoveryArtifact):
        raise ValueError("artifact must be a FragmentationRecoveryArtifact")
    return artifact.to_canonical_bytes()


__all__ = [
    "BOUNDED_FRAGMENT_RECOVERY_SCORE",
    "DETERMINISTIC_CHAIN_REPAIR_RULE",
    "FRAGMENTATION_RECOVERY_LAW",
    "REPLAY_SAFE_RECOVERY_CHAIN",
    "FragmentationRecoveryArtifact",
    "FragmentationRecoveryReceipt",
    "RepairedLineageReport",
    "SonificationContinuityRepairMetadata",
    "export_fragmentation_recovery_bytes",
    "generate_fragmentation_recovery_receipt",
    "recover_fragmented_compression_chain",
]
