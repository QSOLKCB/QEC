from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from qec.analysis.canonical_hashing import sha256_hex
from qec.analysis.deterministic_compression_storage import (
    DETERMINISTIC_COMPRESSION_STORAGE_MODULE_VERSION,
    CanonicalStorageArtifact,
    CompressionSegment,
    CompressionStoragePlan,
    CompressionStorageReceipt,
    plan_deterministic_compression_storage,
)


def _hash(label: str) -> str:
    return sha256_hex({"seed": label})


def _artifact(
    *,
    artifact_id: str,
    artifact_type: str = "memory",
    canonical_size_bytes: int = 100,
    priority: int = 1,
    retention_class: str = "SESSION",
) -> CanonicalStorageArtifact:
    return CanonicalStorageArtifact(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        artifact_hash=_hash(f"artifact::{artifact_id}"),
        canonical_size_bytes=canonical_size_bytes,
        lineage_hashes=(_hash(f"lineage::{artifact_id}"),),
        priority=priority,
        retention_class=retention_class,
    )


def test_empty_artifacts_returns_empty_receipt() -> None:
    receipt = plan_deterministic_compression_storage(())
    assert receipt.module_version == DETERMINISTIC_COMPRESSION_STORAGE_MODULE_VERSION
    assert receipt.artifact_count == 0
    assert receipt.plan.plan_status == "EMPTY"
    assert receipt.plan.segment_count == 0
    assert receipt.plan.overall_compression_ratio == 0.0
    assert receipt.invariant_preserved is True
    assert receipt.receipt_reason == "empty"


def test_compression_disabled_creates_one_segment_per_artifact() -> None:
    artifacts = (
        _artifact(artifact_id="a-2", retention_class="RELEASE", artifact_type="governance", canonical_size_bytes=200),
        _artifact(artifact_id="a-1", retention_class="SESSION", artifact_type="memory", canonical_size_bytes=50),
    )
    receipt = plan_deterministic_compression_storage(artifacts, compression_enabled=False)
    assert receipt.compression_enabled is False
    assert receipt.receipt_reason == "compression_disabled"
    assert receipt.plan.plan_status == "NO_GAIN"
    assert receipt.plan.segment_count == 2
    assert all(len(segment.artifact_ids) == 1 for segment in receipt.plan.segments)
    assert all(segment.compression_ratio == 1.0 for segment in receipt.plan.segments)
    assert receipt.plan.total_compressed_size_bytes == receipt.plan.total_canonical_size_bytes
    assert receipt.plan.overall_compression_ratio == 1.0


def test_compression_enabled_groups_by_retention_and_type() -> None:
    artifacts = (
        _artifact(artifact_id="a-1", retention_class="SESSION", artifact_type="memory"),
        _artifact(artifact_id="a-2", retention_class="SESSION", artifact_type="memory"),
        _artifact(artifact_id="a-3", retention_class="SESSION", artifact_type="governance"),
        _artifact(artifact_id="a-4", retention_class="RELEASE", artifact_type="memory"),
    )
    receipt = plan_deterministic_compression_storage(artifacts)
    grouped = {(segment.storage_class, len(segment.artifact_ids)) for segment in receipt.plan.segments}
    assert ("WARM", 2) in grouped
    assert ("WARM", 1) in grouped
    assert ("COLD", 1) in grouped


def test_archival_compression_factor_applied() -> None:
    receipt = plan_deterministic_compression_storage((_artifact(artifact_id="a-1", canonical_size_bytes=100, retention_class="ARCHIVAL"),))
    segment = receipt.plan.segments[0]
    assert segment.compressed_size_bytes == 55
    assert segment.compression_ratio == 0.55
    assert segment.storage_class == "ARCHIVAL"


def test_release_compression_factor_applied() -> None:
    receipt = plan_deterministic_compression_storage((_artifact(artifact_id="a-1", canonical_size_bytes=100, retention_class="RELEASE"),))
    segment = receipt.plan.segments[0]
    assert segment.compressed_size_bytes == 65
    assert segment.compression_ratio == 0.65
    assert segment.storage_class == "COLD"


def test_session_compression_factor_applied() -> None:
    receipt = plan_deterministic_compression_storage((_artifact(artifact_id="a-1", canonical_size_bytes=100, retention_class="SESSION"),))
    segment = receipt.plan.segments[0]
    assert segment.compressed_size_bytes == 80
    assert segment.compression_ratio == 0.8
    assert segment.storage_class == "WARM"


def test_ephemeral_produces_no_gain() -> None:
    receipt = plan_deterministic_compression_storage((_artifact(artifact_id="a-1", canonical_size_bytes=100, retention_class="EPHEMERAL"),))
    segment = receipt.plan.segments[0]
    assert segment.compressed_size_bytes == 100
    assert segment.compression_ratio == 1.0
    assert segment.storage_class == "HOT"
    assert receipt.plan.plan_status == "NO_GAIN"


def test_zero_size_artifact_remains_zero_size() -> None:
    receipt = plan_deterministic_compression_storage((_artifact(artifact_id="a-1", canonical_size_bytes=0, retention_class="ARCHIVAL"),))
    segment = receipt.plan.segments[0]
    assert segment.compressed_size_bytes == 0
    assert segment.compression_ratio == 0.0
    assert receipt.plan.total_canonical_size_bytes == 0
    assert receipt.plan.total_compressed_size_bytes == 0
    assert receipt.plan.overall_compression_ratio == 0.0


def test_compression_disabled_zero_size_artifacts_keep_zero_ratio() -> None:
    artifacts = (
        _artifact(artifact_id="a-1", canonical_size_bytes=0, retention_class="SESSION"),
        _artifact(artifact_id="a-2", canonical_size_bytes=0, retention_class="ARCHIVAL"),
    )
    receipt = plan_deterministic_compression_storage(artifacts, compression_enabled=False)
    assert receipt.plan.plan_status == "NO_GAIN"
    assert receipt.plan.total_canonical_size_bytes == 0
    assert receipt.plan.total_compressed_size_bytes == 0
    assert receipt.plan.overall_compression_ratio == 0.0
    assert all(segment.compression_ratio == 0.0 for segment in receipt.plan.segments)


def test_deterministic_ordering_independent_of_input_order() -> None:
    artifacts = (
        _artifact(artifact_id="a-1", retention_class="RELEASE", artifact_type="execution", priority=1),
        _artifact(artifact_id="a-2", retention_class="ARCHIVAL", artifact_type="memory", priority=5),
        _artifact(artifact_id="a-3", retention_class="SESSION", artifact_type="governance", priority=3),
    )
    left = plan_deterministic_compression_storage(artifacts)
    right = plan_deterministic_compression_storage(tuple(reversed(artifacts)))
    assert left.to_dict() == right.to_dict()
    assert left.to_canonical_json() == right.to_canonical_json()
    assert left.stable_hash() == right.stable_hash()


def test_duplicate_artifact_id_rejected() -> None:
    a1 = _artifact(artifact_id="dup")
    a2 = CanonicalStorageArtifact(
        artifact_id="dup",
        artifact_type="memory",
        artifact_hash=_hash("artifact::other"),
        canonical_size_bytes=100,
        lineage_hashes=(_hash("lineage::other"),),
        priority=1,
        retention_class="SESSION",
    )
    with pytest.raises(ValueError, match="duplicate artifact_id"):
        plan_deterministic_compression_storage((a1, a2))


def test_duplicate_artifact_hash_rejected() -> None:
    shared_hash = _hash("same")
    a1 = CanonicalStorageArtifact(
        artifact_id="a-1",
        artifact_type="memory",
        artifact_hash=shared_hash,
        canonical_size_bytes=100,
        lineage_hashes=(_hash("lineage::a-1"),),
        priority=1,
        retention_class="SESSION",
    )
    a2 = CanonicalStorageArtifact(
        artifact_id="a-2",
        artifact_type="memory",
        artifact_hash=shared_hash,
        canonical_size_bytes=100,
        lineage_hashes=(_hash("lineage::a-2"),),
        priority=1,
        retention_class="SESSION",
    )
    with pytest.raises(ValueError, match="duplicate artifact_hash"):
        plan_deterministic_compression_storage((a1, a2))


def test_invalid_sha256_rejected() -> None:
    with pytest.raises(ValueError, match="artifact_hash must be a valid SHA-256 hex"):
        CanonicalStorageArtifact(
            artifact_id="a-1",
            artifact_type="memory",
            artifact_hash="xyz",
            canonical_size_bytes=1,
            lineage_hashes=tuple(),
            priority=0,
            retention_class="SESSION",
        )


def test_invalid_retention_class_rejected() -> None:
    with pytest.raises(ValueError, match="retention_class must be one of"):
        _artifact(artifact_id="a-1", retention_class="INVALID")


def test_invalid_storage_class_rejected() -> None:
    with pytest.raises(ValueError, match="storage_class must be one of"):
        CompressionSegment(
            segment_id="seg-session-memory-000000000000",
            artifact_ids=("a-1",),
            source_hashes=(_hash("src"),),
            segment_hash=_hash("segment"),
            compressed_size_bytes=1,
            compression_ratio=1.0,
            storage_class="INVALID",
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_compression_ratio_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="compression_ratio must be bounded"):
        CompressionSegment(
            segment_id="seg-session-memory-000000000000",
            artifact_ids=("a-1",),
            source_hashes=(_hash("src"),),
            segment_hash=_hash("segment"),
            compressed_size_bytes=1,
            compression_ratio=value,
            storage_class="WARM",
        )


def test_plan_status_ratio_invariant_enforced() -> None:
    segment = CompressionSegment(
        segment_id="seg-session-memory-000000000000",
        artifact_ids=("a-1",),
        source_hashes=(_hash("src"),),
        segment_hash=_hash("segment"),
        compressed_size_bytes=100,
        compression_ratio=1.0,
        storage_class="WARM",
    )
    with pytest.raises(ValueError, match="COMPRESSED plan_status"):
        CompressionStoragePlan(
            plan_status="COMPRESSED",
            artifact_count=1,
            segment_count=1,
            segments=(segment,),
            total_canonical_size_bytes=100,
            total_compressed_size_bytes=100,
            overall_compression_ratio=1.0,
            storage_integrity_hash=_hash("integrity"),
        )


def test_receipt_artifact_count_mismatch_rejected() -> None:
    segment = CompressionSegment(
        segment_id="seg-session-memory-000000000000",
        artifact_ids=("a-1",),
        source_hashes=(_hash("src"),),
        segment_hash=_hash("segment"),
        compressed_size_bytes=100,
        compression_ratio=1.0,
        storage_class="WARM",
    )
    plan = CompressionStoragePlan(
        plan_status="NO_GAIN",
        artifact_count=1,
        segment_count=1,
        segments=(segment,),
        total_canonical_size_bytes=100,
        total_compressed_size_bytes=100,
        overall_compression_ratio=1.0,
        storage_integrity_hash=_hash("integrity"),
    )
    with pytest.raises(ValueError, match="artifact_count must match plan.artifact_count"):
        CompressionStorageReceipt(
            module_version=DETERMINISTIC_COMPRESSION_STORAGE_MODULE_VERSION,
            artifact_count=2,
            plan=plan,
            compression_enabled=True,
            invariant_preserved=True,
            receipt_reason="invariant_preserved",
        )


def test_frozen_dataclass_immutability() -> None:
    receipt = plan_deterministic_compression_storage((_artifact(artifact_id="a-1"),))
    with pytest.raises(FrozenInstanceError):
        receipt.artifact_count = 9


def test_canonical_json_stability() -> None:
    receipt = plan_deterministic_compression_storage((_artifact(artifact_id="a-1", retention_class="ARCHIVAL"),))
    replayed = CompressionStorageReceipt(
        module_version=receipt.module_version,
        artifact_count=receipt.artifact_count,
        plan=receipt.plan,
        compression_enabled=receipt.compression_enabled,
        invariant_preserved=receipt.invariant_preserved,
        receipt_reason=receipt.receipt_reason,
        stable_hash_input=receipt.stable_hash(),
    )
    assert receipt.to_canonical_json() == replayed.to_canonical_json()
    assert receipt.to_canonical_bytes() == replayed.to_canonical_bytes()


def test_stable_hash_replay_stability() -> None:
    receipt_a = plan_deterministic_compression_storage(
        (
            _artifact(artifact_id="a-1", retention_class="RELEASE", artifact_type="memory"),
            _artifact(artifact_id="a-2", retention_class="SESSION", artifact_type="governance"),
        )
    )
    receipt_b = plan_deterministic_compression_storage(
        (
            _artifact(artifact_id="a-2", retention_class="SESSION", artifact_type="governance"),
            _artifact(artifact_id="a-1", retention_class="RELEASE", artifact_type="memory"),
        )
    )
    assert receipt_a.stable_hash() == receipt_b.stable_hash()


def test_invalid_plan_status_constructible_for_tests() -> None:
    plan = CompressionStoragePlan(
        plan_status="INVALID",
        artifact_count=0,
        segment_count=0,
        segments=tuple(),
        total_canonical_size_bytes=0,
        total_compressed_size_bytes=0,
        overall_compression_ratio=0.0,
        storage_integrity_hash=_hash("integrity"),
    )
    assert plan.plan_status == "INVALID"
