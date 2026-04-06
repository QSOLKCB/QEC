from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.arithmetic_topology_correspondence_engine import build_arithmetic_topology_correspondence
from qec.analysis.e8_symmetry_projection_layer import build_e8_symmetry_projection
from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.fragmentation_recovery_engine import recover_fragmented_compression_chain
from qec.analysis.hash_preserving_memory_compression import CompressedMemoryArtifact, compress_semantic_theme_memory
from qec.analysis.legacy_copper_noise_channel_battery import run_legacy_copper_noise_channel_battery
from qec.analysis.manifold_traversal_planner import build_manifold_traversal_plan
from qec.analysis.multimodal_feature_schema import MultimodalFeatureSchemaResult, build_multimodal_feature_schema
from qec.analysis.polytope_reasoning_engine import build_polytope_reasoning_engine
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes
from qec.analysis.spectral_reasoning_layer import build_spectral_reasoning_layer
from qec.analysis.telecom_line_recovery_and_sync import (
    BOUNDED_RECOVERY_SCORE_RULE,
    DETERMINISTIC_SYNC_ORDERING_RULE,
    REPLAY_SAFE_RECOVERY_IDENTITY_RULE,
    TELECOM_RECOVERY_LAYER_LAW,
    export_telecom_recovery_bytes,
    generate_telecom_recovery_receipt,
    run_telecom_line_recovery,
)
from qec.analysis.topological_graph_kernel import build_topological_graph_kernel
from qec.analysis.topology_divergence_battery import run_topology_divergence_battery


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:telecom-{i:03d}" if i % 2 == 0 else f"telecom-{i:03d}",
            "sequence_index": i,
            "source_id": "src-a" if i < 12 else "src-b",
            "provenance_id": "prov-a" if i % 3 == 0 else "prov-b",
            "state_token": "A" if i % 2 == 0 else "B",
            "task_completed": bool(i % 4 == 0),
            "is_reset": bool(i > 0 and i % 11 == 0),
        }
        for i in range(24)
    )


def _compressed_artifact() -> CompressedMemoryArtifact:
    episodic = lift_raw_records_to_episodic_memory(_records())
    semantic = compact_episodic_memory_to_semantic_themes(episodic)
    return compress_semantic_theme_memory(semantic)


def _schema_artifact() -> MultimodalFeatureSchemaResult:
    compressed = _compressed_artifact()
    seed = tuple(range(min(6, len(compressed.records))))
    observed = tuple(compressed.records[idx] for idx in seed)
    recovery = recover_fragmented_compression_chain(
        compressed,
        observed_records=observed,
        enable_fragmentation_recovery=True,
    )
    graph = build_topological_graph_kernel(compressed, recovery)
    polytope = build_polytope_reasoning_engine(graph)
    symmetry = build_e8_symmetry_projection(polytope, graph_artifact=graph)
    traversal = build_manifold_traversal_plan(symmetry, polytope_artifact=polytope, graph_artifact=graph)
    divergence = run_topology_divergence_battery(
        traversal,
        symmetry_artifact=symmetry,
        polytope_artifact=polytope,
        graph_artifact=graph,
    )
    correspondence = build_arithmetic_topology_correspondence(
        divergence,
        traversal_artifact=traversal,
        symmetry_artifact=symmetry,
        polytope_artifact=polytope,
        graph_artifact=graph,
    )
    return build_multimodal_feature_schema(correspondence, float_payload={"zeta": 0.25, "alpha": 0.5})


def _battery_artifact():
    spectral = build_spectral_reasoning_layer(_schema_artifact())
    return run_legacy_copper_noise_channel_battery(
        spectral,
        attenuation_fixture=(1.0,),
        distortion_fixture=(1,),
        label_fixture=("fixture",),
    )


def test_identical_input_produces_byte_identical_recovery_artifacts() -> None:
    battery = _battery_artifact()

    artifact_a = run_telecom_line_recovery(battery)
    artifact_b = run_telecom_line_recovery(battery)

    assert artifact_a == artifact_b
    assert artifact_a.to_canonical_bytes() == artifact_b.to_canonical_bytes()
    assert export_telecom_recovery_bytes(artifact_a) == export_telecom_recovery_bytes(artifact_b)
    assert artifact_a.stable_hash() == artifact_a.telecom_recovery_hash


def test_export_rejects_non_recovery_artifact() -> None:
    with pytest.raises(ValueError, match="must be a TelecomRecoveryResult"):
        export_telecom_recovery_bytes(object())


def test_deterministic_repeated_runs() -> None:
    battery = _battery_artifact()

    artifacts = [
        export_telecom_recovery_bytes(
            run_telecom_line_recovery(
                battery,
                gain_fixture=(3.0, 1.0, 2.0),
                line_index_fixture=(3, 1, 2),
                tag_fixture=("z", "a"),
            )
        )
        for _ in range(25)
    ]
    assert len(set(artifacts)) == 1


def test_stable_sync_frame_ordering() -> None:
    battery = _battery_artifact()
    artifact = run_telecom_line_recovery(battery)

    for segment in artifact.segments:
        frame_keys = tuple((frame.frame_index, frame.recovery_mode) for frame in segment.frames)
        assert frame_keys == (
            (0, segment.recovery_mode),
            (1, segment.recovery_mode),
            (2, segment.recovery_mode),
        )


def test_stable_recovery_ordering() -> None:
    battery = _battery_artifact()
    artifact = run_telecom_line_recovery(battery)

    segment_keys = tuple((segment.segment_index, segment.recovery_mode) for segment in artifact.segments)
    assert segment_keys == (
        (0, "line_relock"),
        (1, "burst_recovery"),
        (2, "carrier_phase_sync"),
        (3, "attenuation_compensation"),
        (4, "continuity_rebuild"),
    )


def test_stable_hash_invariants() -> None:
    battery = _battery_artifact()
    artifact = run_telecom_line_recovery(battery)

    assert artifact.law_invariants == (
        TELECOM_RECOVERY_LAYER_LAW,
        DETERMINISTIC_SYNC_ORDERING_RULE,
        REPLAY_SAFE_RECOVERY_IDENTITY_RULE,
        BOUNDED_RECOVERY_SCORE_RULE,
    )
    assert artifact.stable_hash() == artifact.telecom_recovery_hash
    for segment in artifact.segments:
        assert segment.segment_hash == segment.stable_hash()
        assert len(segment.segment_id) == 64
        for frame in segment.frames:
            assert frame.frame_hash == frame.stable_hash()
            assert frame.frame_id == frame.frame_hash


def test_fail_fast_malformed_battery_input() -> None:
    battery = _battery_artifact()
    invalid = replace(battery, copper_channel_battery_hash="bad-hash")

    with pytest.raises(ValueError, match="copper_channel_battery_hash must match stable_hash"):
        run_telecom_line_recovery(invalid)


def test_fail_fast_zero_scenario_battery() -> None:
    battery = _battery_artifact()
    stripped = replace(battery, scenarios=(), scenario_count=0)
    zero_scenario = replace(stripped, copper_channel_battery_hash=stripped.stable_hash())

    with pytest.raises(ValueError, match="zero-scenario batteries cannot produce valid recovery lineage"):
        run_telecom_line_recovery(zero_scenario)


def test_deterministic_relock_behavior() -> None:
    battery = _battery_artifact()
    artifact = run_telecom_line_recovery(battery)

    relock_segment = artifact.segments[0]
    continuity_segment = artifact.segments[-1]

    assert relock_segment.recovery_mode == "line_relock"
    assert relock_segment.line_recovery_score > continuity_segment.line_recovery_score
    assert relock_segment.overall_recovery_score > continuity_segment.overall_recovery_score


def test_receipt_determinism() -> None:
    battery = _battery_artifact()
    artifact = run_telecom_line_recovery(battery)

    receipt_a = generate_telecom_recovery_receipt(artifact)
    receipt_b = generate_telecom_recovery_receipt(artifact)

    assert receipt_a == receipt_b
    assert receipt_a.stable_hash() == receipt_a.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_generate_receipt_rejects_non_recovery_artifact() -> None:
    with pytest.raises(ValueError, match="must be a TelecomRecoveryResult"):
        generate_telecom_recovery_receipt(object())


def test_generate_receipt_rejects_mismatched_recovery_hash() -> None:
    battery = _battery_artifact()
    artifact = run_telecom_line_recovery(battery)
    tampered = replace(artifact, telecom_recovery_hash="deadbeef")

    with pytest.raises(ValueError, match="must match stable_hash"):
        generate_telecom_recovery_receipt(tampered)


def test_lineage_hash_preservation() -> None:
    battery = _battery_artifact()
    artifact = run_telecom_line_recovery(battery)

    assert artifact.source_feature_schema_hash == battery.source_feature_schema_hash
    assert artifact.source_spectral_reasoning_hash == battery.source_spectral_reasoning_hash
    assert artifact.source_copper_channel_battery_hash == battery.copper_channel_battery_hash


def test_bounded_score_checks() -> None:
    battery = _battery_artifact()
    artifact = run_telecom_line_recovery(battery)

    scores = (
        artifact.carrier_lock_integrity_score,
        artifact.line_recovery_score,
        artifact.burst_recovery_score,
        artifact.sync_frame_consistency_score,
        artifact.overall_recovery_score,
    )
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_nan_infinity_rejection() -> None:
    battery = _battery_artifact()

    with pytest.raises(ValueError, match="finite"):
        run_telecom_line_recovery(battery, gain_fixture=(1.0, float("nan")))

    with pytest.raises(ValueError, match="finite"):
        run_telecom_line_recovery(battery, gain_fixture=(1.0, float("inf")))


def test_battery_identity_controls_recovery_identity() -> None:
    battery = _battery_artifact()

    baseline = run_telecom_line_recovery(battery)
    with_fixtures = run_telecom_line_recovery(
        battery,
        gain_fixture=(4.0, 1.0, 2.0),
        line_index_fixture=(4, 1, 2),
        tag_fixture=("gamma", "alpha"),
    )

    assert baseline.to_canonical_bytes() == with_fixtures.to_canonical_bytes()
    assert baseline.telecom_recovery_hash == with_fixtures.telecom_recovery_hash
