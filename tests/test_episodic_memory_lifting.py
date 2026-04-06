from __future__ import annotations

import pytest

from qec.analysis.episodic_memory_lifting import (
    EpisodeBoundaryConfig,
    detect_episode_boundaries,
    export_episodic_memory_bytes,
    generate_episodic_memory_receipt,
    lift_raw_records_to_episodic_memory,
)


def _records() -> tuple[dict[str, object], ...]:
    return (
        {
            "record_id": "r0",
            "sequence_index": 0,
            "source_id": "sensor-a",
            "provenance_id": "p0",
            "state_token": "s0",
            "timestamp": 0.0,
        },
        {
            "record_id": "r1",
            "sequence_index": 1,
            "source_id": "sensor-a",
            "provenance_id": "p0",
            "state_token": "s0",
            "timestamp": 1.0,
        },
        {
            "record_id": "r2",
            "sequence_index": 2,
            "source_id": "sensor-a",
            "provenance_id": "p0",
            "state_token": "s0",
            "timestamp": 2.0,
            "task_completed": True,
        },
        {
            "record_id": "r3",
            "sequence_index": 3,
            "source_id": "sensor-a",
            "provenance_id": "p0",
            "state_token": "s0",
            "timestamp": 3.0,
        },
        {
            "record_id": "r4",
            "sequence_index": 4,
            "source_id": "sensor-a",
            "provenance_id": "p0",
            "state_token": "s0",
            "timestamp": 7.0,
            "is_reset": True,
        },
    )


def test_deterministic_episode_segmentation() -> None:
    config = EpisodeBoundaryConfig(max_episode_length=10, timestamp_gap_seconds=10.0)
    a = lift_raw_records_to_episodic_memory(_records(), config=config)
    b = lift_raw_records_to_episodic_memory(_records(), config=config)

    assert a == b
    assert a.to_canonical_bytes() == b.to_canonical_bytes()


def test_stable_boundary_detection() -> None:
    config = EpisodeBoundaryConfig(max_episode_length=10, timestamp_gap_seconds=2.0)
    boundaries = detect_episode_boundaries(_records(), config=config)

    assert boundaries == (
        (3, ("task_completion",)),
        (4, ("reset_marker", "timestamp_gap")),
    )


def test_bounded_max_length_segmentation() -> None:
    records = tuple({"record_id": f"r{i}", "sequence_index": i} for i in range(5))
    artifact = lift_raw_records_to_episodic_memory(records, config=EpisodeBoundaryConfig(max_episode_length=2))

    assert artifact.episode_count == 3
    assert tuple(ep.record_ids for ep in artifact.episodes) == (("r0", "r1"), ("r2", "r3"), ("r4",))


def test_reset_marker_segmentation() -> None:
    records = (
        {"record_id": "a", "sequence_index": 0},
        {"record_id": "b", "sequence_index": 1, "is_reset": True},
        {"record_id": "c", "sequence_index": 2},
    )
    artifact = lift_raw_records_to_episodic_memory(records)
    assert artifact.episode_count == 2
    assert artifact.episodes[1].boundary_reasons == ("reset_marker",)


def test_task_completion_segmentation() -> None:
    records = (
        {"record_id": "a", "sequence_index": 0, "task_completed": True},
        {"record_id": "b", "sequence_index": 1},
        {"record_id": "c", "sequence_index": 2},
    )
    artifact = lift_raw_records_to_episodic_memory(records)
    assert artifact.episode_count == 2
    assert artifact.episodes[1].boundary_reasons == ("task_completion",)


def test_canonical_export_stability() -> None:
    artifact = lift_raw_records_to_episodic_memory(_records(), config=EpisodeBoundaryConfig(max_episode_length=3))
    a = artifact.to_canonical_json()
    b = artifact.to_canonical_json()
    assert a == b


def test_stable_receipt_hash_stability() -> None:
    artifact_a = lift_raw_records_to_episodic_memory(_records())
    artifact_b = lift_raw_records_to_episodic_memory(_records())

    receipt_a = generate_episodic_memory_receipt(artifact_a)
    receipt_b = generate_episodic_memory_receipt(artifact_b)

    assert receipt_a.receipt_hash == receipt_b.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_repeated_run_byte_identity_n_times() -> None:
    config = EpisodeBoundaryConfig(max_episode_length=3, timestamp_gap_seconds=2.5)
    blobs = [
        export_episodic_memory_bytes(lift_raw_records_to_episodic_memory(_records(), config=config))
        for _ in range(7)
    ]
    assert len(set(blobs)) == 1


def test_fail_fast_malformed_input() -> None:
    with pytest.raises(ValueError, match="records must be non-empty"):
        lift_raw_records_to_episodic_memory(())

    with pytest.raises(ValueError, match="record_id must be non-empty"):
        lift_raw_records_to_episodic_memory(({"record_id": "   ", "sequence_index": 0},))

    with pytest.raises(ValueError, match="record_id values must be unique"):
        lift_raw_records_to_episodic_memory(
            (
                {"record_id": "dup", "sequence_index": 0},
                {"record_id": "dup", "sequence_index": 1},
            )
        )

    with pytest.raises(ValueError, match="sequence_index must be non-decreasing"):
        lift_raw_records_to_episodic_memory(
            (
                {"record_id": "a", "sequence_index": 2},
                {"record_id": "b", "sequence_index": 1},
            )
        )

    with pytest.raises(ValueError, match="max_episode_length must be a positive integer"):
        EpisodeBoundaryConfig(max_episode_length=0)

    with pytest.raises(ValueError, match="timestamp_gap_seconds must be finite and > 0"):
        EpisodeBoundaryConfig(timestamp_gap_seconds=0.0)


def test_adversarial_ordering_normalization() -> None:
    records = (
        {"record_id": "r2", "sequence_index": 2},
        {"record_id": "r0", "sequence_index": 0},
        {"record_id": "r1", "sequence_index": 1},
    )

    with pytest.raises(ValueError, match="sequence_index must be non-decreasing"):
        lift_raw_records_to_episodic_memory(records)

    normalized = lift_raw_records_to_episodic_memory(
        records,
        config=EpisodeBoundaryConfig(normalize_by_sequence_index=True, max_episode_length=10),
    )
    assert normalized.episodes[0].record_ids == ("r0", "r1", "r2")
