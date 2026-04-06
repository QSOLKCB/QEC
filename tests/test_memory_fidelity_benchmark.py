from __future__ import annotations

from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.fragmentation_recovery_engine import recover_fragmented_compression_chain
from qec.analysis.hash_preserving_memory_compression import compress_semantic_theme_memory
from qec.analysis.memory_fidelity_benchmark import (
    BOUNDED_FIDELITY_SCORE_RULE,
    HASH_PRESERVATION_FIDELITY_RULE,
    MEMORY_FIDELITY_BENCHMARK_LAW,
    REPLAY_IDENTITY_FIDELITY_RULE,
    export_memory_fidelity_benchmark_bytes,
    generate_memory_fidelity_receipt,
    run_memory_fidelity_benchmark,
)
from qec.analysis.semantic_memory_sonification import project_compressed_memory_to_sonification
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:alpha-{i:03d}" if i % 6 == 0 else f"ops-{i:03d}",
            "sequence_index": i,
            "source_id": "src-a" if i < 30 else "src-b",
            "provenance_id": "prov-a" if i % 3 == 0 else "prov-b",
            "state_token": "S" if i % 2 == 0 else "T",
            "task_completed": bool(i % 11 == 0),
            "is_reset": bool(i > 0 and i % 17 == 0),
        }
        for i in range(54)
    )


def _compressed_artifact():
    episodic = lift_raw_records_to_episodic_memory(_records())
    semantic = compact_episodic_memory_to_semantic_themes(episodic)
    return compress_semantic_theme_memory(semantic)


def test_identical_input_produces_byte_identical_benchmark_artifact() -> None:
    compressed = _compressed_artifact()
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=tuple(compressed.records),
        enable_fragmentation_recovery=True,
    )

    result_a = run_memory_fidelity_benchmark(compressed, recovery)
    result_b = run_memory_fidelity_benchmark(compressed, recovery)

    assert result_a == result_b
    assert result_a.to_canonical_bytes() == result_b.to_canonical_bytes()
    assert export_memory_fidelity_benchmark_bytes(result_a) == export_memory_fidelity_benchmark_bytes(result_b)


def test_deterministic_repeated_runs_and_canonical_bytes() -> None:
    compressed = _compressed_artifact()
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=(compressed.records[3], compressed.records[1], compressed.records[0]),
        enable_fragmentation_recovery=True,
    )

    blobs = [export_memory_fidelity_benchmark_bytes(run_memory_fidelity_benchmark(compressed, recovery)) for _ in range(25)]
    assert len(set(blobs)) == 1


def test_bounded_score_invariants() -> None:
    compressed = _compressed_artifact()
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=(compressed.records[0],),
        enable_fragmentation_recovery=True,
    )
    result = run_memory_fidelity_benchmark(compressed, recovery)
    snapshot = result.snapshot

    assert result.law_invariants == (
        MEMORY_FIDELITY_BENCHMARK_LAW,
        REPLAY_IDENTITY_FIDELITY_RULE,
        HASH_PRESERVATION_FIDELITY_RULE,
        BOUNDED_FIDELITY_SCORE_RULE,
    )

    scores = (
        snapshot.continuity_fidelity_score,
        snapshot.ordering_fidelity_score,
        snapshot.hash_fidelity_score,
        snapshot.replay_fidelity_score,
        snapshot.sonification_fidelity_score,
        snapshot.recovery_correctness_fidelity_score,
        snapshot.noop_path_fidelity_score,
        snapshot.partial_repair_fidelity_score,
        snapshot.unrecoverable_corruption_classification_fidelity_score,
        snapshot.overall_fidelity_score,
    )
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_perfect_chain_fidelity_is_one() -> None:
    compressed = _compressed_artifact()
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=tuple(compressed.records),
        enable_fragmentation_recovery=True,
    )

    result = run_memory_fidelity_benchmark(compressed, recovery)
    assert result.snapshot.overall_fidelity_score == 1.0
    assert result.snapshot.replay_fidelity_score == 1.0


def test_partial_fracture_fidelity_is_less_than_one() -> None:
    compressed = _compressed_artifact()
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=(compressed.records[2], compressed.records[0]),
        enable_fragmentation_recovery=True,
    )

    result = run_memory_fidelity_benchmark(compressed, recovery)
    assert result.snapshot.classification == "partial_fragment_repair"
    assert result.snapshot.overall_fidelity_score < 1.0


def test_unrecoverable_corruption_is_lowest_fidelity_path() -> None:
    compressed = _compressed_artifact()
    no_op_recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=tuple(compressed.records),
        enable_fragmentation_recovery=True,
    )
    partial_recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=(compressed.records[4], compressed.records[0]),
        enable_fragmentation_recovery=True,
    )
    wrong = type(compressed.records[0])(
        theme_id=compressed.records[0].theme_id,
        theme_index=0,
        source_theme_hash=compressed.records[1].source_theme_hash,
        source_replay_identity_hash=compressed.records[1].source_replay_identity_hash,
        source_parent_theme_hash=compressed.records[1].source_parent_theme_hash,
        signature_ref=compressed.records[1].signature_ref,
        reason_ref=compressed.records[1].reason_ref,
        episode_hashes_ref=compressed.records[1].episode_hashes_ref,
        compression_record_hash=compressed.records[1].compression_record_hash,
    )
    unrecoverable_recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=(wrong,),
        enable_fragmentation_recovery=True,
    )

    no_op_score = run_memory_fidelity_benchmark(compressed, no_op_recovery).snapshot.overall_fidelity_score
    partial_score = run_memory_fidelity_benchmark(compressed, partial_recovery).snapshot.overall_fidelity_score
    unrecoverable_score = run_memory_fidelity_benchmark(compressed, unrecoverable_recovery).snapshot.overall_fidelity_score

    assert no_op_score >= partial_score >= unrecoverable_score


def test_stable_hash_and_receipt_determinism() -> None:
    compressed = _compressed_artifact()
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=(compressed.records[1], compressed.records[0]),
        enable_fragmentation_recovery=True,
    )
    benchmark = run_memory_fidelity_benchmark(compressed, recovery)
    receipt_a = generate_memory_fidelity_receipt(benchmark)
    receipt_b = generate_memory_fidelity_receipt(benchmark)

    assert benchmark.snapshot.stable_hash() == benchmark.snapshot.snapshot_hash
    assert benchmark.stable_hash() == benchmark.benchmark_hash
    assert receipt_a == receipt_b
    assert receipt_a.stable_hash() == receipt_a.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_sonification_lineage_fidelity_when_metadata_present() -> None:
    compressed = _compressed_artifact()
    sonification = project_compressed_memory_to_sonification(compressed)
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=(compressed.records[3], compressed.records[0]),
        sonification_spec=sonification,
        enable_fragmentation_recovery=True,
    )
    result = run_memory_fidelity_benchmark(compressed, recovery)

    assert result.snapshot.sonification_metadata_present is True
    assert result.snapshot.sonification_fidelity_score == 1.0
