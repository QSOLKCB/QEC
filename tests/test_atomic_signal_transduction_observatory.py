from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.arithmetic_topology_correspondence_engine import build_arithmetic_topology_correspondence
from qec.analysis.atomic_signal_transduction_observatory import (
    ATOMIC_SIGNAL_TRANSDUCTION_OBSERVATORY_LAW,
    BOUNDED_OBSERVATORY_SCORE_RULE,
    DETERMINISTIC_OBSERVATORY_ORDERING_RULE,
    REPLAY_SAFE_OBSERVATORY_IDENTITY_RULE,
    export_atomic_signal_observatory_bytes,
    generate_atomic_signal_observatory_receipt,
    run_atomic_signal_transduction_observatory,
)
from qec.analysis.cross_modal_replay_certification import run_cross_modal_replay_certification
from qec.analysis.e8_symmetry_projection_layer import build_e8_symmetry_projection
from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.fragmentation_recovery_engine import recover_fragmented_compression_chain
from qec.analysis.hash_preserving_memory_compression import CompressedMemoryArtifact, compress_semantic_theme_memory
from qec.analysis.legacy_copper_noise_channel_battery import run_legacy_copper_noise_channel_battery
from qec.analysis.manifold_traversal_planner import build_manifold_traversal_plan
from qec.analysis.multimodal_feature_schema import MultimodalFeatureSchemaResult, build_multimodal_feature_schema
from qec.analysis.polytope_reasoning_engine import build_polytope_reasoning_engine
from qec.analysis.rf_equalization_and_ground_station_compensation import run_rf_equalization
from qec.analysis.satellite_signal_baseline_and_orbital_noise import run_satellite_signal_baseline
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes
from qec.analysis.spectral_reasoning_layer import build_spectral_reasoning_layer
from qec.analysis.telecom_line_recovery_and_sync import run_telecom_line_recovery
from qec.analysis.topological_graph_kernel import build_topological_graph_kernel
from qec.analysis.topology_divergence_battery import run_topology_divergence_battery


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:atomic-observatory-{i:03d}" if i % 2 == 0 else f"atomic-observatory-{i:03d}",
            "sequence_index": i,
            "source_id": "src-a" if i < 11 else "src-b",
            "provenance_id": "prov-a" if i % 3 == 0 else "prov-b",
            "state_token": "A" if i % 2 == 0 else "B",
            "task_completed": bool(i % 5 == 0),
            "is_reset": bool(i > 0 and i % 13 == 0),
        }
        for i in range(23)
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
    return build_multimodal_feature_schema(correspondence, float_payload={"zeta": 0.35, "alpha": 0.45})


@pytest.fixture(scope="module")
def lineage_artifacts() -> tuple:
    schema = _schema_artifact()
    spectral = build_spectral_reasoning_layer(schema)
    battery = run_legacy_copper_noise_channel_battery(
        spectral,
        attenuation_fixture=(1.0,),
        distortion_fixture=(1,),
        label_fixture=("fixture",),
    )
    telecom = run_telecom_line_recovery(battery)
    satellite = run_satellite_signal_baseline(telecom)
    rf = run_rf_equalization(satellite)
    certification = run_cross_modal_replay_certification(
        rf,
        schema_artifact=schema,
        spectral_artifact=spectral,
        telecom_artifact=telecom,
        satellite_artifact=satellite,
    )
    return schema, spectral, battery, telecom, satellite, rf, certification


def test_identical_input_produces_byte_identical_observatory_artifacts(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, certification = lineage_artifacts

    artifact_a = run_atomic_signal_transduction_observatory(certification)
    artifact_b = run_atomic_signal_transduction_observatory(certification)

    assert artifact_a == artifact_b
    assert artifact_a.to_canonical_bytes() == artifact_b.to_canonical_bytes()
    assert export_atomic_signal_observatory_bytes(artifact_a) == export_atomic_signal_observatory_bytes(artifact_b)
    assert artifact_a.stable_hash() == artifact_a.atomic_signal_observatory_hash


def test_deterministic_repeated_runs(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, certification = lineage_artifacts

    artifacts = [
        export_atomic_signal_observatory_bytes(
            run_atomic_signal_transduction_observatory(
                certification,
                observatory_profile="end_to_end_signal_window",
                score_fixture=(1.0, 0.0, 0.5),
            )
        )
        for _ in range(20)
    ]
    assert len(set(artifacts)) == 1


def test_stable_observation_ordering_and_stable_window_ordering(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, certification = lineage_artifacts
    artifact = run_atomic_signal_transduction_observatory(certification)

    observation_keys = tuple(
        (o.observation_index, o.observatory_profile, o.observation_id)
        for o in artifact.observations
    )
    window_keys = tuple((w.window_index, w.window_id, w.window_hash) for w in artifact.windows)

    assert observation_keys == tuple(sorted(observation_keys))
    assert window_keys == tuple(sorted(window_keys))
    assert artifact.observation_count == len(artifact.observations)
    assert artifact.window_count == len(artifact.windows)


def test_stable_hash_invariants(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, certification = lineage_artifacts
    artifact = run_atomic_signal_transduction_observatory(certification)

    assert artifact.law_invariants == (
        ATOMIC_SIGNAL_TRANSDUCTION_OBSERVATORY_LAW,
        DETERMINISTIC_OBSERVATORY_ORDERING_RULE,
        REPLAY_SAFE_OBSERVATORY_IDENTITY_RULE,
        BOUNDED_OBSERVATORY_SCORE_RULE,
    )
    assert artifact.stable_hash() == artifact.atomic_signal_observatory_hash
    for observation in artifact.observations:
        assert observation.observation_hash == observation.stable_hash()
        assert observation.observation_id == observation.observation_hash
    for window in artifact.windows:
        assert window.window_hash == window.stable_hash()


def test_fail_fast_malformed_upstream_input(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, certification = lineage_artifacts
    invalid = replace(certification, replay_certification_hash="bad-hash")

    with pytest.raises(ValueError, match="replay_certification_hash must match stable_hash"):
        run_atomic_signal_transduction_observatory(invalid)


def test_lineage_preservation_and_end_to_end_observatory_continuity(lineage_artifacts: tuple) -> None:
    schema, spectral, battery, telecom, satellite, rf, certification = lineage_artifacts

    artifact = run_atomic_signal_transduction_observatory(
        certification,
        observatory_profile="end_to_end_signal_window",
        schema_artifact=schema,
        spectral_artifact=spectral,
        telecom_artifact=telecom,
        satellite_artifact=satellite,
        rf_artifact=rf,
    )

    assert artifact.source_feature_schema_hash == schema.feature_schema_hash
    assert artifact.source_spectral_reasoning_hash == spectral.spectral_reasoning_hash
    assert artifact.source_copper_channel_battery_hash == battery.copper_channel_battery_hash
    assert artifact.source_telecom_recovery_hash == telecom.telecom_recovery_hash
    assert artifact.source_satellite_baseline_hash == satellite.satellite_baseline_hash
    assert artifact.source_rf_equalization_hash == rf.rf_equalization_hash
    assert artifact.source_replay_certification_hash == certification.replay_certification_hash
    assert artifact.overall_observatory_score == pytest.approx(certification.overall_certification_score)


def test_receipt_determinism(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, certification = lineage_artifacts
    artifact = run_atomic_signal_transduction_observatory(certification)

    receipt_a = generate_atomic_signal_observatory_receipt(artifact)
    receipt_b = generate_atomic_signal_observatory_receipt(artifact)

    assert receipt_a == receipt_b
    assert receipt_a.stable_hash() == receipt_a.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_bounded_score_checks(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, certification = lineage_artifacts
    artifact = run_atomic_signal_transduction_observatory(certification)

    scores = (
        artifact.transduction_integrity_score,
        artifact.window_consistency_score,
        artifact.observation_alignment_score,
        artifact.replay_visibility_score,
        artifact.overall_observatory_score,
    )
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_nan_infinity_rejection(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, certification = lineage_artifacts

    with pytest.raises(ValueError, match="finite"):
        run_atomic_signal_transduction_observatory(certification, score_fixture=(1.0, float("nan")))
    with pytest.raises(ValueError, match="finite"):
        run_atomic_signal_transduction_observatory(certification, score_fixture=(1.0, float("inf")))


def test_direct_lineage_mismatch_rejection(lineage_artifacts: tuple) -> None:
    schema, spectral, _, telecom, satellite, rf, certification = lineage_artifacts
    wrong_satellite = replace(satellite, source_telecom_recovery_hash="f" * 64)
    wrong_satellite = replace(wrong_satellite, satellite_baseline_hash=wrong_satellite.stable_hash())

    with pytest.raises(ValueError, match="direct lineage mismatch"):
        run_atomic_signal_transduction_observatory(
            certification,
            schema_artifact=schema,
            spectral_artifact=spectral,
            telecom_artifact=telecom,
            satellite_artifact=wrong_satellite,
            rf_artifact=rf,
        )


def test_forged_source_field_rejected(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, certification = lineage_artifacts
    # Forge source_feature_schema_hash while keeping ledger.lineage_chain unchanged,
    # then recompute replay_certification_hash so the hash self-consistency check passes.
    forged = replace(certification, source_feature_schema_hash="a" * 64)
    forged = replace(forged, replay_certification_hash=forged.stable_hash())

    with pytest.raises(ValueError, match="source fields.*inconsistent.*ledger lineage_chain"):
        run_atomic_signal_transduction_observatory(forged)
