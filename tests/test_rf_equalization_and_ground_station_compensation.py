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
from qec.analysis.rf_equalization_and_ground_station_compensation import (
    BOUNDED_RF_SCORE_RULE,
    DETERMINISTIC_RF_ORDERING_RULE,
    REPLAY_SAFE_RF_IDENTITY_RULE,
    RF_EQUALIZATION_LAYER_LAW,
    export_rf_equalization_bytes,
    generate_rf_equalization_receipt,
    run_rf_equalization,
)
from qec.analysis.satellite_signal_baseline_and_orbital_noise import run_satellite_signal_baseline
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes
from qec.analysis.spectral_reasoning_layer import build_spectral_reasoning_layer
from qec.analysis.telecom_line_recovery_and_sync import run_telecom_line_recovery
from qec.analysis.topological_graph_kernel import build_topological_graph_kernel
from qec.analysis.topology_divergence_battery import run_topology_divergence_battery


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:rf-{i:03d}" if i % 2 == 0 else f"rf-{i:03d}",
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


def _satellite_artifact():
    spectral = build_spectral_reasoning_layer(_schema_artifact())
    battery = run_legacy_copper_noise_channel_battery(
        spectral,
        attenuation_fixture=(1.0,),
        distortion_fixture=(1,),
        label_fixture=("fixture",),
    )
    telecom = run_telecom_line_recovery(battery)
    return run_satellite_signal_baseline(telecom)


def _with_uniform_satellite_scores(satellite, value: float):
    segment_sets = []
    for segment in satellite.segments:
        frames = []
        for frame in segment.frames:
            updated_frame = replace(
                frame,
                envelope_noise_floor=1.0 - value,
                envelope_latency_pressure=1.0 - value,
                frame_consistency_score=value,
            )
            frames.append(replace(updated_frame, frame_hash=updated_frame.stable_hash(), frame_id=updated_frame.stable_hash()))

        updated_segment = replace(
            segment,
            frames=tuple(frames),
            frame_count=len(frames),
            orbital_integrity_score=value,
            signal_latency_resilience_score=value,
            relay_handoff_score=value,
            frame_consistency_score=value,
            overall_satellite_score=value,
        )
        segment_sets.append(replace(updated_segment, segment_hash=updated_segment.stable_hash(), segment_id=updated_segment.stable_hash()))

    updated = replace(
        satellite,
        segments=tuple(segment_sets),
        segment_count=len(segment_sets),
        frame_count=sum(segment.frame_count for segment in segment_sets),
        orbital_integrity_score=value,
        signal_latency_resilience_score=value,
        relay_handoff_score=value,
        frame_consistency_score=value,
        overall_satellite_score=value,
    )
    return replace(updated, satellite_baseline_hash=updated.stable_hash())


def test_identical_input_produces_byte_identical_rf_artifacts() -> None:
    satellite = _satellite_artifact()

    artifact_a = run_rf_equalization(satellite)
    artifact_b = run_rf_equalization(satellite)

    assert artifact_a == artifact_b
    assert artifact_a.to_canonical_bytes() == artifact_b.to_canonical_bytes()
    assert export_rf_equalization_bytes(artifact_a) == export_rf_equalization_bytes(artifact_b)
    assert artifact_a.stable_hash() == artifact_a.rf_equalization_hash


def test_deterministic_repeated_runs() -> None:
    satellite = _satellite_artifact()

    artifacts = [
        export_rf_equalization_bytes(
            run_rf_equalization(
                satellite,
                float_fixture=(3.0, 1.0, 2.0),
                int_fixture=(3, 1, 2),
                str_fixture=("z", "a"),
            )
        )
        for _ in range(25)
    ]
    assert len(set(artifacts)) == 1


def test_stable_frame_ordering() -> None:
    satellite = _satellite_artifact()
    artifact = run_rf_equalization(satellite)

    for segment in artifact.segments:
        frame_keys = tuple((frame.frame_index, frame.compensation_scenario) for frame in segment.frames)
        assert frame_keys == (
            (0, segment.compensation_scenario),
            (1, segment.compensation_scenario),
            (2, segment.compensation_scenario),
        )


def test_stable_segment_ordering_and_compensation_scenarios() -> None:
    satellite = _satellite_artifact()
    artifact = run_rf_equalization(satellite)

    segment_keys = tuple((segment.segment_index, segment.ground_station_profile, segment.compensation_scenario) for segment in artifact.segments)
    assert segment_keys == (
        (0, "urban_ground_station", "nominal_station"),
        (1, "rural_ground_station", "atmospheric_shift"),
        (2, "desert_station", "ground_reflection"),
        (3, "maritime_station", "horizon_occlusion"),
        (4, "polar_station", "polar_noise"),
    )


def test_stable_hash_invariants() -> None:
    satellite = _satellite_artifact()
    artifact = run_rf_equalization(satellite)

    assert artifact.law_invariants == (
        RF_EQUALIZATION_LAYER_LAW,
        DETERMINISTIC_RF_ORDERING_RULE,
        REPLAY_SAFE_RF_IDENTITY_RULE,
        BOUNDED_RF_SCORE_RULE,
    )
    assert artifact.stable_hash() == artifact.rf_equalization_hash
    for segment in artifact.segments:
        assert segment.segment_hash == segment.stable_hash()
        assert len(segment.segment_id) == 64
        for frame in segment.frames:
            assert frame.frame_hash == frame.stable_hash()
            assert frame.frame_id == frame.frame_hash


def test_fail_fast_malformed_satellite_input() -> None:
    satellite = _satellite_artifact()
    invalid = replace(satellite, satellite_baseline_hash="bad-hash")

    with pytest.raises(ValueError, match="satellite_baseline_hash must match stable_hash"):
        run_rf_equalization(invalid)


def test_fail_fast_zero_segments() -> None:
    satellite = _satellite_artifact()
    zero_seg = replace(satellite, segment_count=0, segments=(), frame_count=0)
    zero_seg = replace(zero_seg, satellite_baseline_hash=zero_seg.stable_hash())

    with pytest.raises(ValueError, match="at least one segment"):
        run_rf_equalization(zero_seg)


def test_fail_fast_zero_frames_in_segment() -> None:
    satellite = _satellite_artifact()
    first_seg = satellite.segments[0]
    zero_frame_seg = replace(first_seg, frame_count=0, frames=())
    zero_frame_seg = replace(zero_frame_seg, segment_hash=zero_frame_seg.stable_hash(), segment_id=zero_frame_seg.stable_hash())
    new_segments = (zero_frame_seg,) + satellite.segments[1:]
    updated = replace(
        satellite,
        segments=new_segments,
        frame_count=sum(s.frame_count for s in new_segments),
    )
    updated = replace(updated, satellite_baseline_hash=updated.stable_hash())

    with pytest.raises(ValueError, match="at least one frame"):
        run_rf_equalization(updated)


def test_receipt_determinism() -> None:
    satellite = _satellite_artifact()
    artifact = run_rf_equalization(satellite)

    receipt_a = generate_rf_equalization_receipt(artifact)
    receipt_b = generate_rf_equalization_receipt(artifact)

    assert receipt_a == receipt_b
    assert receipt_a.stable_hash() == receipt_a.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_lineage_hash_preservation() -> None:
    satellite = _satellite_artifact()
    artifact = run_rf_equalization(satellite)

    assert artifact.source_feature_schema_hash == satellite.source_feature_schema_hash
    assert artifact.source_spectral_reasoning_hash == satellite.source_spectral_reasoning_hash
    assert artifact.source_copper_channel_battery_hash == satellite.source_copper_channel_battery_hash
    assert artifact.source_telecom_recovery_hash == satellite.source_telecom_recovery_hash
    assert artifact.source_satellite_baseline_hash == satellite.satellite_baseline_hash


def test_bounded_score_checks() -> None:
    satellite = _satellite_artifact()
    artifact = run_rf_equalization(satellite)

    scores = (
        artifact.equalization_integrity_score,
        artifact.compensation_stability_score,
        artifact.reflection_resilience_score,
        artifact.frame_consistency_score,
        artifact.overall_rf_score,
    )
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_nan_infinity_rejection() -> None:
    satellite = _satellite_artifact()

    with pytest.raises(ValueError, match="finite"):
        run_rf_equalization(satellite, float_fixture=(1.0, float("nan")))

    with pytest.raises(ValueError, match="finite"):
        run_rf_equalization(satellite, float_fixture=(1.0, float("inf")))


def test_export_and_receipt_reject_non_rf_artifact() -> None:
    with pytest.raises(ValueError, match="must be a RFEqualizationResult"):
        export_rf_equalization_bytes(object())

    with pytest.raises(ValueError, match="must be a RFEqualizationResult"):
        generate_rf_equalization_receipt(object())


def test_generate_receipt_rejects_mismatched_rf_hash() -> None:
    satellite = _satellite_artifact()
    artifact = run_rf_equalization(satellite)
    tampered = replace(artifact, rf_equalization_hash="deadbeef")

    with pytest.raises(ValueError, match="must match stable_hash"):
        generate_rf_equalization_receipt(tampered)


def test_satellite_identity_controls_rf_identity() -> None:
    satellite = _satellite_artifact()

    baseline = run_rf_equalization(satellite)
    with_fixtures = run_rf_equalization(
        satellite,
        float_fixture=(4.0, 1.0, 2.0),
        int_fixture=(4, 1, 2),
        str_fixture=("gamma", "alpha"),
    )

    assert baseline.to_canonical_bytes() == with_fixtures.to_canonical_bytes()
    assert baseline.rf_equalization_hash == with_fixtures.rf_equalization_hash


def test_perfect_orbital_continuity_produces_max_scores() -> None:
    satellite = _satellite_artifact()
    normalized = _with_uniform_satellite_scores(satellite, 1.0)

    artifact = run_rf_equalization(normalized)

    assert artifact.equalization_integrity_score == 1.0
    assert artifact.compensation_stability_score == 1.0


def test_severe_rf_degradation_lowers_scores_deterministically() -> None:
    satellite = _satellite_artifact()
    degraded = _with_uniform_satellite_scores(satellite, 0.0)

    artifact_a = run_rf_equalization(degraded)
    artifact_b = run_rf_equalization(degraded)

    assert artifact_a == artifact_b
    assert artifact_a.equalization_integrity_score < 1.0
    assert artifact_a.compensation_stability_score < 1.0
    assert artifact_a.reflection_resilience_score < 1.0
    assert artifact_a.overall_rf_score < 1.0


def test_deterministic_compensation_scenarios() -> None:
    satellite = _satellite_artifact()
    artifact = run_rf_equalization(satellite)

    scenarios = tuple(segment.compensation_scenario for segment in artifact.segments)
    assert scenarios == (
        "nominal_station",
        "atmospheric_shift",
        "ground_reflection",
        "horizon_occlusion",
        "polar_noise",
    )
