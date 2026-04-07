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
from qec.analysis.satellite_signal_baseline_and_orbital_noise import (
    BOUNDED_SATELLITE_SCORE_RULE,
    DETERMINISTIC_ORBITAL_ORDERING_RULE,
    REPLAY_SAFE_SATELLITE_IDENTITY_RULE,
    SATELLITE_BASELINE_LAYER_LAW,
    export_satellite_baseline_bytes,
    generate_satellite_baseline_receipt,
    run_satellite_signal_baseline,
)
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes
from qec.analysis.spectral_reasoning_layer import build_spectral_reasoning_layer
from qec.analysis.telecom_line_recovery_and_sync import run_telecom_line_recovery
from qec.analysis.topological_graph_kernel import build_topological_graph_kernel
from qec.analysis.topology_divergence_battery import run_topology_divergence_battery


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:satellite-{i:03d}" if i % 2 == 0 else f"satellite-{i:03d}",
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


def _telecom_artifact():
    spectral = build_spectral_reasoning_layer(_schema_artifact())
    battery = run_legacy_copper_noise_channel_battery(
        spectral,
        attenuation_fixture=(1.0,),
        distortion_fixture=(1,),
        label_fixture=("fixture",),
    )
    return run_telecom_line_recovery(battery)


def _with_uniform_telecom_scores(telecom, value: float):
    frame_sets = []
    for segment in telecom.segments:
        frames = []
        for frame in segment.frames:
            updated_frame = replace(
                frame,
                carrier_lock_score=value,
                sync_drift_score=value,
                frame_consistency_score=value,
            )
            frames.append(replace(updated_frame, frame_hash=updated_frame.stable_hash(), frame_id=updated_frame.stable_hash()))

        updated_segment = replace(
            segment,
            frames=tuple(frames),
            frame_count=len(frames),
            carrier_lock_integrity_score=value,
            line_recovery_score=value,
            burst_recovery_score=value,
            sync_frame_consistency_score=value,
            overall_recovery_score=value,
        )
        frame_sets.append(replace(updated_segment, segment_hash=updated_segment.stable_hash()))

    updated = replace(
        telecom,
        segments=tuple(frame_sets),
        segment_count=len(frame_sets),
        frame_count=sum(segment.frame_count for segment in frame_sets),
        carrier_lock_integrity_score=value,
        line_recovery_score=value,
        burst_recovery_score=value,
        sync_frame_consistency_score=value,
        overall_recovery_score=value,
    )
    return replace(updated, telecom_recovery_hash=updated.stable_hash())


def test_identical_input_produces_byte_identical_satellite_artifacts() -> None:
    telecom = _telecom_artifact()

    artifact_a = run_satellite_signal_baseline(telecom)
    artifact_b = run_satellite_signal_baseline(telecom)

    assert artifact_a == artifact_b
    assert artifact_a.to_canonical_bytes() == artifact_b.to_canonical_bytes()
    assert export_satellite_baseline_bytes(artifact_a) == export_satellite_baseline_bytes(artifact_b)
    assert artifact_a.stable_hash() == artifact_a.satellite_baseline_hash


def test_deterministic_repeated_runs() -> None:
    telecom = _telecom_artifact()

    artifacts = [
        export_satellite_baseline_bytes(
            run_satellite_signal_baseline(
                telecom,
                float_fixture=(3.0, 1.0, 2.0),
                int_fixture=(3, 1, 2),
                str_fixture=("z", "a"),
            )
        )
        for _ in range(25)
    ]
    assert len(set(artifacts)) == 1


def test_stable_frame_ordering() -> None:
    telecom = _telecom_artifact()
    artifact = run_satellite_signal_baseline(telecom)

    for segment in artifact.segments:
        frame_keys = tuple((frame.frame_index, frame.orbital_scenario) for frame in segment.frames)
        assert frame_keys == (
            (0, segment.orbital_scenario),
            (1, segment.orbital_scenario),
            (2, segment.orbital_scenario),
        )


def test_stable_segment_ordering_and_orbital_scenarios() -> None:
    telecom = _telecom_artifact()
    artifact = run_satellite_signal_baseline(telecom)

    segment_keys = tuple((segment.segment_index, segment.orbital_scenario, segment.orbital_class) for segment in artifact.segments)
    assert segment_keys == (
        (0, "nominal_orbit", "leo"),
        (1, "solar_noise", "meo"),
        (2, "eclipse_shadow", "geo"),
        (3, "relay_handoff", "relay"),
        (4, "deep_space_latency", "deep_space"),
    )


def test_stable_hash_invariants() -> None:
    telecom = _telecom_artifact()
    artifact = run_satellite_signal_baseline(telecom)

    assert artifact.law_invariants == (
        SATELLITE_BASELINE_LAYER_LAW,
        DETERMINISTIC_ORBITAL_ORDERING_RULE,
        REPLAY_SAFE_SATELLITE_IDENTITY_RULE,
        BOUNDED_SATELLITE_SCORE_RULE,
    )
    assert artifact.stable_hash() == artifact.satellite_baseline_hash
    for segment in artifact.segments:
        assert segment.segment_hash == segment.stable_hash()
        assert len(segment.segment_id) == 64
        for frame in segment.frames:
            assert frame.frame_hash == frame.stable_hash()
            assert frame.frame_id == frame.frame_hash


def test_fail_fast_malformed_telecom_input() -> None:
    telecom = _telecom_artifact()
    invalid = replace(telecom, telecom_recovery_hash="bad-hash")

    with pytest.raises(ValueError, match="telecom_recovery_hash must match stable_hash"):
        run_satellite_signal_baseline(invalid)


def test_fail_fast_zero_segments() -> None:
    telecom = _telecom_artifact()
    zero_seg = replace(telecom, segment_count=0, segments=(), frame_count=0)
    zero_seg = replace(zero_seg, telecom_recovery_hash=zero_seg.stable_hash())

    with pytest.raises(ValueError, match="at least one segment"):
        run_satellite_signal_baseline(zero_seg)


def test_fail_fast_zero_frames_in_segment() -> None:
    telecom = _telecom_artifact()
    first_seg = telecom.segments[0]
    zero_frame_seg = replace(first_seg, frame_count=0, frames=())
    zero_frame_seg = replace(zero_frame_seg, segment_hash=zero_frame_seg.stable_hash())
    new_segments = (zero_frame_seg,) + telecom.segments[1:]
    updated = replace(
        telecom,
        segments=new_segments,
        frame_count=sum(s.frame_count for s in new_segments),
    )
    updated = replace(updated, telecom_recovery_hash=updated.stable_hash())

    with pytest.raises(ValueError, match="at least one frame"):
        run_satellite_signal_baseline(updated)


def test_receipt_determinism() -> None:
    telecom = _telecom_artifact()
    artifact = run_satellite_signal_baseline(telecom)

    receipt_a = generate_satellite_baseline_receipt(artifact)
    receipt_b = generate_satellite_baseline_receipt(artifact)

    assert receipt_a == receipt_b
    assert receipt_a.stable_hash() == receipt_a.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_lineage_hash_preservation() -> None:
    telecom = _telecom_artifact()
    artifact = run_satellite_signal_baseline(telecom)

    assert artifact.source_feature_schema_hash == telecom.source_feature_schema_hash
    assert artifact.source_spectral_reasoning_hash == telecom.source_spectral_reasoning_hash
    assert artifact.source_copper_channel_battery_hash == telecom.source_copper_channel_battery_hash
    assert artifact.source_telecom_recovery_hash == telecom.telecom_recovery_hash


def test_bounded_score_checks() -> None:
    telecom = _telecom_artifact()
    artifact = run_satellite_signal_baseline(telecom)

    scores = (
        artifact.orbital_integrity_score,
        artifact.signal_latency_resilience_score,
        artifact.relay_handoff_score,
        artifact.frame_consistency_score,
        artifact.overall_satellite_score,
    )
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_nan_infinity_rejection() -> None:
    telecom = _telecom_artifact()

    with pytest.raises(ValueError, match="finite"):
        run_satellite_signal_baseline(telecom, float_fixture=(1.0, float("nan")))

    with pytest.raises(ValueError, match="finite"):
        run_satellite_signal_baseline(telecom, float_fixture=(1.0, float("inf")))


def test_export_and_receipt_reject_non_baseline_artifact() -> None:
    with pytest.raises(ValueError, match="must be a SatelliteBaselineResult"):
        export_satellite_baseline_bytes(object())

    with pytest.raises(ValueError, match="must be a SatelliteBaselineResult"):
        generate_satellite_baseline_receipt(object())


def test_generate_receipt_rejects_mismatched_baseline_hash() -> None:
    telecom = _telecom_artifact()
    artifact = run_satellite_signal_baseline(telecom)
    tampered = replace(artifact, satellite_baseline_hash="deadbeef")

    with pytest.raises(ValueError, match="must match stable_hash"):
        generate_satellite_baseline_receipt(tampered)


def test_telecom_identity_controls_satellite_identity() -> None:
    telecom = _telecom_artifact()

    baseline = run_satellite_signal_baseline(telecom)
    with_fixtures = run_satellite_signal_baseline(
        telecom,
        float_fixture=(4.0, 1.0, 2.0),
        int_fixture=(4, 1, 2),
        str_fixture=("gamma", "alpha"),
    )

    assert baseline.to_canonical_bytes() == with_fixtures.to_canonical_bytes()
    assert baseline.satellite_baseline_hash == with_fixtures.satellite_baseline_hash


def test_perfect_continuity_produces_max_scores() -> None:
    telecom = _telecom_artifact()
    normalized = _with_uniform_telecom_scores(telecom, 1.0)

    artifact = run_satellite_signal_baseline(normalized)

    assert artifact.orbital_integrity_score == 1.0
    assert artifact.signal_latency_resilience_score == 1.0
    assert artifact.relay_handoff_score == 1.0


def test_severe_orbital_degradation_lowers_scores_deterministically() -> None:
    telecom = _telecom_artifact()
    degraded = _with_uniform_telecom_scores(telecom, 0.0)

    artifact_a = run_satellite_signal_baseline(degraded)
    artifact_b = run_satellite_signal_baseline(degraded)

    assert artifact_a == artifact_b
    assert artifact_a.orbital_integrity_score < 1.0
    assert artifact_a.signal_latency_resilience_score < 1.0
    assert artifact_a.relay_handoff_score < 1.0
    assert artifact_a.overall_satellite_score < 1.0
