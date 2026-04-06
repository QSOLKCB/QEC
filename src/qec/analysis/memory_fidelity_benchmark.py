"""v137.7.4 — Memory Fidelity Benchmark.

Deterministic Layer-4 benchmark for fidelity and replay identity across the
v137.7.x memory chain.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from qec.analysis.fragmentation_recovery_engine import FragmentationRecoveryArtifact
from qec.analysis.hash_preserving_memory_compression import CompressedMemoryArtifact, CompressionRecord

_JSONScalar = str | int | float | bool | None
_JSONValue = _JSONScalar | tuple["_JSONValue", ...] | dict[str, "_JSONValue"]

_SCHEMA_VERSION = 1

MEMORY_FIDELITY_BENCHMARK_LAW = "MEMORY_FIDELITY_BENCHMARK_LAW"
REPLAY_IDENTITY_FIDELITY_RULE = "REPLAY_IDENTITY_FIDELITY_RULE"
HASH_PRESERVATION_FIDELITY_RULE = "HASH_PRESERVATION_FIDELITY_RULE"
BOUNDED_FIDELITY_SCORE_RULE = "BOUNDED_FIDELITY_SCORE_RULE"


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


def _clamp_score(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, float):
        raise ValueError("score must be float")
    if not math.isfinite(value):
        raise ValueError("score must be finite")
    return min(1.0, max(0.0, value))


def _safe_fraction(num: int, den: int) -> float:
    if den <= 0:
        return 1.0
    return _clamp_score(float(num / den))


def _validate_source_artifact(artifact: CompressedMemoryArtifact) -> tuple[CompressionRecord, ...]:
    if not isinstance(artifact, CompressedMemoryArtifact):
        raise ValueError("source_artifact must be a CompressedMemoryArtifact")

    records = artifact.records
    if artifact.compressed_record_count != len(records):
        raise ValueError("source_artifact compressed_record_count must match records length")
    if artifact.source_theme_count != len(records):
        raise ValueError("source_artifact source_theme_count must match records length")
    if len(records) == 0:
        raise ValueError("source_artifact must contain at least one CompressionRecord")
    if artifact.compression_chain_head != records[-1].source_replay_identity_hash:
        raise ValueError(
            "source_artifact compression_chain_head must match the last record source_replay_identity_hash"
        )

    for expected_theme_index, record in enumerate(records):
        if record.theme_index != expected_theme_index:
            raise ValueError("source_artifact records must have contiguous theme_index values")

        expected_parent_theme_index = None if expected_theme_index == 0 else expected_theme_index - 1
        if record.parent_theme_index != expected_parent_theme_index:
            raise ValueError(
                "source_artifact records must maintain sequential parent_theme_index lineage"
            )

    return records
def _validate_recovery_artifact(
    artifact: FragmentationRecoveryArtifact,
    *,
    source_artifact: CompressedMemoryArtifact,
    source_records: tuple[CompressionRecord, ...],
) -> tuple[CompressionRecord, ...]:
    if not isinstance(artifact, FragmentationRecoveryArtifact):
        raise ValueError("recovery_artifact must be a FragmentationRecoveryArtifact")
    if artifact.source_compression_hash != source_artifact.compression_hash:
        raise ValueError("recovery_artifact source_compression_hash must match source_artifact")
    if artifact.source_replay_identity_hash != source_artifact.replay_identity_hash:
        raise ValueError("recovery_artifact source_replay_identity_hash must match source_artifact")
    if len(artifact.repaired_records) != len(source_records):
        raise ValueError("recovery_artifact repaired_records must match source record count")
    return artifact.repaired_records


def _score_chain_continuity(
    source_records: tuple[CompressionRecord, ...],
    repaired_records: tuple[CompressionRecord, ...],
    *,
    observed_continuity_score: float,
) -> float:
    continuity_matches = 0
    for idx, source_record in enumerate(source_records):
        repaired = repaired_records[idx]
        if repaired.source_replay_identity_hash == source_record.source_replay_identity_hash:
            continuity_matches += 1
    direct_score = _safe_fraction(continuity_matches, len(source_records))
    return _clamp_score(float(min(direct_score, observed_continuity_score)))


def _score_theme_ordering(
    source_records: tuple[CompressionRecord, ...],
    repaired_records: tuple[CompressionRecord, ...],
) -> float:
    ordering_matches = 0
    for idx, source_record in enumerate(source_records):
        repaired = repaired_records[idx]
        if repaired.theme_index == idx and repaired.theme_id == source_record.theme_id:
            ordering_matches += 1
    return _safe_fraction(ordering_matches, len(source_records))


def _score_hash_preservation(
    source_records: tuple[CompressionRecord, ...],
    repaired_records: tuple[CompressionRecord, ...],
) -> float:
    total_checks = len(source_records) * 2
    preserved = 0
    for idx, source_record in enumerate(source_records):
        repaired = repaired_records[idx]
        if repaired.source_theme_hash == source_record.source_theme_hash:
            preserved += 1
        if repaired.compression_record_hash == source_record.compression_record_hash:
            preserved += 1
    return _safe_fraction(preserved, total_checks)


def _score_replay_identity(
    source_artifact: CompressedMemoryArtifact,
    repaired_records: tuple[CompressionRecord, ...],
) -> float:
    source_payload = {
        "records": tuple(record.to_dict() for record in source_artifact.records),
        "preserved_theme_hashes": source_artifact.preserved_theme_hashes,
        "compression_chain_head": source_artifact.compression_chain_head,
        "replay_identity_hash": source_artifact.replay_identity_hash,
    }
    repaired_payload = {
        "records": tuple(record.to_dict() for record in repaired_records),
        "preserved_theme_hashes": source_artifact.preserved_theme_hashes,
        "compression_chain_head": repaired_records[-1].source_replay_identity_hash,
        "replay_identity_hash": source_artifact.replay_identity_hash,
    }
    return 1.0 if _canonical_bytes(source_payload) == _canonical_bytes(repaired_payload) else 0.0


def _score_sonification_lineage(recovery_artifact: FragmentationRecoveryArtifact) -> float:
    metadata = recovery_artifact.sonification_repair_metadata
    if metadata is None:
        return 1.0
    if metadata.sonification_lineage_preserved and metadata.source_compression_hash == recovery_artifact.source_compression_hash:
        return 1.0
    return 0.0


def _expected_classification(recovery_artifact: FragmentationRecoveryArtifact) -> str:
    report = recovery_artifact.lineage_report
    if len(report.corrupted_theme_indices) > 0 and len(report.missing_theme_indices) == report.source_record_count:
        return "unrecoverable_corruption"
    if len(report.missing_theme_indices) > 0 or len(report.disordered_theme_indices) > 0:
        return "partial_fragment_repair"
    return "no_op"


def _score_recovery_correctness(recovery_artifact: FragmentationRecoveryArtifact) -> float:
    expected = _expected_classification(recovery_artifact)
    classified_correctly = 1.0 if recovery_artifact.lineage_report.classification == expected else 0.0
    confidence_alignment = 1.0 if expected != "unrecoverable_corruption" else 0.0
    confidence_error = abs(recovery_artifact.recovery_confidence - confidence_alignment)
    return _clamp_score(float((classified_correctly + (1.0 - confidence_error)) * 0.5))


def _score_path_fidelity(recovery_artifact: FragmentationRecoveryArtifact, *, target: str) -> float:
    expected = _expected_classification(recovery_artifact)
    if expected != target:
        return 1.0
    return 1.0 if recovery_artifact.lineage_report.classification == target else 0.0


def _weighted_overall_score(scores: dict[str, float]) -> float:
    weighted = (
        scores["continuity_fidelity_score"] * 0.20
        + scores["ordering_fidelity_score"] * 0.15
        + scores["hash_fidelity_score"] * 0.15
        + scores["replay_fidelity_score"] * 0.20
        + scores["sonification_fidelity_score"] * 0.10
        + scores["recovery_correctness_fidelity_score"] * 0.10
        + scores["noop_path_fidelity_score"] * 0.03
        + scores["partial_repair_fidelity_score"] * 0.03
        + scores["unrecoverable_corruption_classification_fidelity_score"] * 0.04
    )
    return _clamp_score(float(weighted))


@dataclass(frozen=True)
class MemoryFidelitySnapshot:
    schema_version: int
    source_compression_hash: str
    source_replay_identity_hash: str
    source_record_count: int
    repaired_record_count: int
    source_chain_head: str
    repaired_chain_head: str
    classification: str
    sonification_metadata_present: bool
    continuity_fidelity_score: float
    ordering_fidelity_score: float
    hash_fidelity_score: float
    replay_fidelity_score: float
    sonification_fidelity_score: float
    recovery_correctness_fidelity_score: float
    noop_path_fidelity_score: float
    partial_repair_fidelity_score: float
    unrecoverable_corruption_classification_fidelity_score: float
    overall_fidelity_score: float
    snapshot_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_compression_hash": self.source_compression_hash,
            "source_replay_identity_hash": self.source_replay_identity_hash,
            "source_record_count": self.source_record_count,
            "repaired_record_count": self.repaired_record_count,
            "source_chain_head": self.source_chain_head,
            "repaired_chain_head": self.repaired_chain_head,
            "classification": self.classification,
            "sonification_metadata_present": self.sonification_metadata_present,
            "continuity_fidelity_score": self.continuity_fidelity_score,
            "ordering_fidelity_score": self.ordering_fidelity_score,
            "hash_fidelity_score": self.hash_fidelity_score,
            "replay_fidelity_score": self.replay_fidelity_score,
            "sonification_fidelity_score": self.sonification_fidelity_score,
            "recovery_correctness_fidelity_score": self.recovery_correctness_fidelity_score,
            "noop_path_fidelity_score": self.noop_path_fidelity_score,
            "partial_repair_fidelity_score": self.partial_repair_fidelity_score,
            "unrecoverable_corruption_classification_fidelity_score": self.unrecoverable_corruption_classification_fidelity_score,
            "overall_fidelity_score": self.overall_fidelity_score,
            "snapshot_hash": self.snapshot_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        payload = self.to_dict()
        payload.pop("snapshot_hash")
        return payload

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class MemoryFidelityBenchmarkResult:
    schema_version: int
    snapshot: MemoryFidelitySnapshot
    law_invariants: tuple[str, ...]
    benchmark_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "snapshot": self.snapshot.to_dict(),
            "law_invariants": self.law_invariants,
            "benchmark_hash": self.benchmark_hash,
        }

    def to_hash_payload_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "snapshot": self.snapshot.to_dict(),
            "law_invariants": self.law_invariants,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")

    def stable_hash(self) -> str:
        return _sha256_hex(self.to_hash_payload_dict())


@dataclass(frozen=True)
class MemoryFidelityReceipt:
    schema_version: int
    source_compression_hash: str
    benchmark_hash: str
    classification: str
    overall_fidelity_score: float
    receipt_hash: str

    def to_dict(self) -> dict[str, _JSONValue]:
        return {
            "schema_version": self.schema_version,
            "source_compression_hash": self.source_compression_hash,
            "benchmark_hash": self.benchmark_hash,
            "classification": self.classification,
            "overall_fidelity_score": self.overall_fidelity_score,
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


def run_memory_fidelity_benchmark(
    source_artifact: CompressedMemoryArtifact,
    recovery_artifact: FragmentationRecoveryArtifact,
) -> MemoryFidelityBenchmarkResult:
    source_records = _validate_source_artifact(source_artifact)
    repaired_records = _validate_recovery_artifact(
        recovery_artifact,
        source_artifact=source_artifact,
        source_records=source_records,
    )

    scores = {
        "continuity_fidelity_score": _score_chain_continuity(
            source_records,
            repaired_records,
            observed_continuity_score=float(recovery_artifact.continuity_score),
        ),
        "ordering_fidelity_score": _score_theme_ordering(source_records, repaired_records),
        "hash_fidelity_score": _score_hash_preservation(source_records, repaired_records),
        "replay_fidelity_score": _score_replay_identity(source_artifact, repaired_records),
        "sonification_fidelity_score": _score_sonification_lineage(recovery_artifact),
        "recovery_correctness_fidelity_score": _score_recovery_correctness(recovery_artifact),
        "noop_path_fidelity_score": _score_path_fidelity(recovery_artifact, target="no_op"),
        "partial_repair_fidelity_score": _score_path_fidelity(recovery_artifact, target="partial_fragment_repair"),
        "unrecoverable_corruption_classification_fidelity_score": _score_path_fidelity(
            recovery_artifact,
            target="unrecoverable_corruption",
        ),
    }

    for key, value in tuple(scores.items()):
        scores[key] = _clamp_score(float(value))
    overall_score = _weighted_overall_score(scores)

    snapshot = MemoryFidelitySnapshot(
        schema_version=_SCHEMA_VERSION,
        source_compression_hash=source_artifact.compression_hash,
        source_replay_identity_hash=source_artifact.replay_identity_hash,
        source_record_count=len(source_records),
        repaired_record_count=len(repaired_records),
        source_chain_head=source_artifact.compression_chain_head,
        repaired_chain_head=recovery_artifact.repaired_chain_head,
        classification=recovery_artifact.lineage_report.classification,
        sonification_metadata_present=recovery_artifact.sonification_repair_metadata is not None,
        continuity_fidelity_score=scores["continuity_fidelity_score"],
        ordering_fidelity_score=scores["ordering_fidelity_score"],
        hash_fidelity_score=scores["hash_fidelity_score"],
        replay_fidelity_score=scores["replay_fidelity_score"],
        sonification_fidelity_score=scores["sonification_fidelity_score"],
        recovery_correctness_fidelity_score=scores["recovery_correctness_fidelity_score"],
        noop_path_fidelity_score=scores["noop_path_fidelity_score"],
        partial_repair_fidelity_score=scores["partial_repair_fidelity_score"],
        unrecoverable_corruption_classification_fidelity_score=scores[
            "unrecoverable_corruption_classification_fidelity_score"
        ],
        overall_fidelity_score=overall_score,
        snapshot_hash="",
    )
    snapshot = MemoryFidelitySnapshot(
        schema_version=snapshot.schema_version,
        source_compression_hash=snapshot.source_compression_hash,
        source_replay_identity_hash=snapshot.source_replay_identity_hash,
        source_record_count=snapshot.source_record_count,
        repaired_record_count=snapshot.repaired_record_count,
        source_chain_head=snapshot.source_chain_head,
        repaired_chain_head=snapshot.repaired_chain_head,
        classification=snapshot.classification,
        sonification_metadata_present=snapshot.sonification_metadata_present,
        continuity_fidelity_score=snapshot.continuity_fidelity_score,
        ordering_fidelity_score=snapshot.ordering_fidelity_score,
        hash_fidelity_score=snapshot.hash_fidelity_score,
        replay_fidelity_score=snapshot.replay_fidelity_score,
        sonification_fidelity_score=snapshot.sonification_fidelity_score,
        recovery_correctness_fidelity_score=snapshot.recovery_correctness_fidelity_score,
        noop_path_fidelity_score=snapshot.noop_path_fidelity_score,
        partial_repair_fidelity_score=snapshot.partial_repair_fidelity_score,
        unrecoverable_corruption_classification_fidelity_score=snapshot.unrecoverable_corruption_classification_fidelity_score,
        overall_fidelity_score=snapshot.overall_fidelity_score,
        snapshot_hash=snapshot.stable_hash(),
    )

    result = MemoryFidelityBenchmarkResult(
        schema_version=_SCHEMA_VERSION,
        snapshot=snapshot,
        law_invariants=(
            MEMORY_FIDELITY_BENCHMARK_LAW,
            REPLAY_IDENTITY_FIDELITY_RULE,
            HASH_PRESERVATION_FIDELITY_RULE,
            BOUNDED_FIDELITY_SCORE_RULE,
        ),
        benchmark_hash="",
    )
    return MemoryFidelityBenchmarkResult(
        schema_version=result.schema_version,
        snapshot=result.snapshot,
        law_invariants=result.law_invariants,
        benchmark_hash=result.stable_hash(),
    )


def export_memory_fidelity_benchmark_bytes(artifact: MemoryFidelityBenchmarkResult) -> bytes:
    if not isinstance(artifact, MemoryFidelityBenchmarkResult):
        raise ValueError("artifact must be a MemoryFidelityBenchmarkResult")
    return artifact.to_canonical_bytes()


def generate_memory_fidelity_receipt(
    artifact: MemoryFidelityBenchmarkResult,
) -> MemoryFidelityReceipt:
    if not isinstance(artifact, MemoryFidelityBenchmarkResult):
        raise ValueError("artifact must be a MemoryFidelityBenchmarkResult")
    receipt = MemoryFidelityReceipt(
        schema_version=artifact.schema_version,
        source_compression_hash=artifact.snapshot.source_compression_hash,
        benchmark_hash=artifact.benchmark_hash,
        classification=artifact.snapshot.classification,
        overall_fidelity_score=artifact.snapshot.overall_fidelity_score,
        receipt_hash="",
    )
    return MemoryFidelityReceipt(
        schema_version=receipt.schema_version,
        source_compression_hash=receipt.source_compression_hash,
        benchmark_hash=receipt.benchmark_hash,
        classification=receipt.classification,
        overall_fidelity_score=receipt.overall_fidelity_score,
        receipt_hash=receipt.stable_hash(),
    )


__all__ = [
    "BOUNDED_FIDELITY_SCORE_RULE",
    "HASH_PRESERVATION_FIDELITY_RULE",
    "MEMORY_FIDELITY_BENCHMARK_LAW",
    "REPLAY_IDENTITY_FIDELITY_RULE",
    "MemoryFidelityBenchmarkResult",
    "MemoryFidelityReceipt",
    "MemoryFidelitySnapshot",
    "export_memory_fidelity_benchmark_bytes",
    "generate_memory_fidelity_receipt",
    "run_memory_fidelity_benchmark",
]
