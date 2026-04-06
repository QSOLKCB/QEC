from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.arithmetic_topology_correspondence_engine import build_arithmetic_topology_correspondence
from qec.analysis.e8_symmetry_projection_layer import build_e8_symmetry_projection
from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.fragmentation_recovery_engine import recover_fragmented_compression_chain
from qec.analysis.hash_preserving_memory_compression import CompressedMemoryArtifact, compress_semantic_theme_memory
from qec.analysis.legacy_copper_noise_channel_battery import (
    BOUNDED_CHANNEL_SCORE_RULE,
    DETERMINISTIC_SCENARIO_ORDERING_RULE,
    LEGACY_COPPER_CHANNEL_BATTERY_LAW,
    REPLAY_SAFE_CHANNEL_BATTERY_IDENTITY_RULE,
    export_copper_channel_battery_bytes,
    generate_copper_channel_battery_receipt,
    run_legacy_copper_noise_channel_battery,
)
from qec.analysis.manifold_traversal_planner import build_manifold_traversal_plan
from qec.analysis.multimodal_feature_schema import MultimodalFeatureSchemaResult, build_multimodal_feature_schema
from qec.analysis.polytope_reasoning_engine import build_polytope_reasoning_engine
from qec.analysis.semantic_theme_compaction import compact_episodic_memory_to_semantic_themes
from qec.analysis.spectral_reasoning_layer import build_spectral_reasoning_layer
from qec.analysis.topological_graph_kernel import build_topological_graph_kernel
from qec.analysis.topology_divergence_battery import run_topology_divergence_battery


def _records() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "record_id": f"theme:copper-{i:03d}" if i % 2 == 0 else f"copper-{i:03d}",
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


def _spectral_artifact():
    return build_spectral_reasoning_layer(_schema_artifact())


def test_identical_input_produces_byte_identical_battery_artifacts() -> None:
    spectral = _spectral_artifact()

    artifact_a = run_legacy_copper_noise_channel_battery(spectral)
    artifact_b = run_legacy_copper_noise_channel_battery(spectral)

    assert artifact_a == artifact_b
    assert artifact_a.to_canonical_bytes() == artifact_b.to_canonical_bytes()
    assert export_copper_channel_battery_bytes(artifact_a) == export_copper_channel_battery_bytes(artifact_b)
    assert artifact_a.stable_hash() == artifact_a.copper_channel_battery_hash


def test_deterministic_repeated_runs() -> None:
    spectral = _spectral_artifact()

    artifacts = [
        export_copper_channel_battery_bytes(
            run_legacy_copper_noise_channel_battery(
                spectral,
                attenuation_fixture=(3.0, 1.0, 2.0),
                distortion_fixture=(3, 1, 2),
                label_fixture=("z", "a"),
            )
        )
        for _ in range(25)
    ]
    assert len(set(artifacts)) == 1


def test_stable_scenario_ordering() -> None:
    spectral = _spectral_artifact()
    artifact = run_legacy_copper_noise_channel_battery(spectral)

    scenario_keys = tuple((scenario.scenario_index, scenario.scenario_name) for scenario in artifact.scenarios)
    assert scenario_keys == (
        (0, "low_noise"),
        (1, "medium_noise"),
        (2, "severe_noise"),
        (3, "burst_noise"),
        (4, "line_loss"),
    )


def test_stable_fixture_ordering() -> None:
    spectral = _spectral_artifact()
    artifact = run_legacy_copper_noise_channel_battery(spectral)

    fixture_keys = tuple((fixture.fixture_index, fixture.channel_family) for fixture in artifact.fixtures)
    assert fixture_keys == (
        (0, "pots"),
        (1, "dsl"),
        (2, "isdn"),
        (3, "t1"),
        (4, "t3"),
    )


def test_stable_hash_invariants() -> None:
    spectral = _spectral_artifact()
    artifact = run_legacy_copper_noise_channel_battery(spectral)

    assert artifact.law_invariants == (
        LEGACY_COPPER_CHANNEL_BATTERY_LAW,
        DETERMINISTIC_SCENARIO_ORDERING_RULE,
        REPLAY_SAFE_CHANNEL_BATTERY_IDENTITY_RULE,
        BOUNDED_CHANNEL_SCORE_RULE,
    )
    assert artifact.stable_hash() == artifact.copper_channel_battery_hash
    for fixture in artifact.fixtures:
        assert fixture.fixture_id == fixture.stable_hash()
        assert fixture.fixture_hash == fixture.stable_hash()


def test_fail_fast_malformed_spectral_input() -> None:
    spectral = _spectral_artifact()
    invalid = replace(spectral, spectral_reasoning_hash="bad-hash")

    with pytest.raises(ValueError, match="spectral_reasoning_hash must match stable_hash"):
        run_legacy_copper_noise_channel_battery(invalid)


def test_deterministic_degradation_curves() -> None:
    spectral = _spectral_artifact()
    artifact = run_legacy_copper_noise_channel_battery(spectral)

    low_noise = artifact.scenarios[0]
    line_loss = artifact.scenarios[-1]

    assert low_noise.attenuation_integrity_score > line_loss.attenuation_integrity_score
    assert low_noise.channel_distortion_score > line_loss.channel_distortion_score
    assert low_noise.burst_noise_resilience_score > line_loss.burst_noise_resilience_score
    assert low_noise.overall_channel_battery_score > line_loss.overall_channel_battery_score


def test_receipt_determinism() -> None:
    spectral = _spectral_artifact()
    artifact = run_legacy_copper_noise_channel_battery(spectral)

    receipt_a = generate_copper_channel_battery_receipt(artifact)
    receipt_b = generate_copper_channel_battery_receipt(artifact)

    assert receipt_a == receipt_b
    assert receipt_a.stable_hash() == receipt_a.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_lineage_hash_preservation() -> None:
    spectral = _spectral_artifact()
    artifact = run_legacy_copper_noise_channel_battery(spectral)

    assert artifact.source_feature_schema_hash == spectral.source_feature_schema_hash
    assert artifact.source_spectral_reasoning_hash == spectral.spectral_reasoning_hash


def test_bounded_score_checks() -> None:
    spectral = _spectral_artifact()
    artifact = run_legacy_copper_noise_channel_battery(spectral)

    scores = (
        artifact.attenuation_integrity_score,
        artifact.channel_distortion_score,
        artifact.burst_noise_resilience_score,
        artifact.fixture_consistency_score,
        artifact.overall_channel_battery_score,
    )
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_nan_infinity_rejection() -> None:
    spectral = _spectral_artifact()

    with pytest.raises(ValueError, match="finite"):
        run_legacy_copper_noise_channel_battery(spectral, attenuation_fixture=(1.0, float("nan")))

    with pytest.raises(ValueError, match="finite"):
        run_legacy_copper_noise_channel_battery(spectral, attenuation_fixture=(1.0, float("inf")))


def test_spectral_identity_controls_battery_identity() -> None:
    spectral = _spectral_artifact()

    baseline = run_legacy_copper_noise_channel_battery(spectral)
    with_fixtures = run_legacy_copper_noise_channel_battery(
        spectral,
        attenuation_fixture=(4.0, 1.0, 2.0),
        distortion_fixture=(4, 1, 2),
        label_fixture=("gamma", "alpha"),
    )

    assert baseline.to_canonical_bytes() == with_fixtures.to_canonical_bytes()
    assert baseline.copper_channel_battery_hash == with_fixtures.copper_channel_battery_hash


def test_generate_receipt_rejects_non_battery_artifact() -> None:
    with pytest.raises(ValueError, match="artifact must be a CopperChannelBatteryResult"):
        generate_copper_channel_battery_receipt(object())


def test_generate_receipt_rejects_mismatched_battery_hash() -> None:
    spectral = _spectral_artifact()
    artifact = run_legacy_copper_noise_channel_battery(spectral)
    invalid = replace(artifact, copper_channel_battery_hash="deadbeef")

    with pytest.raises(ValueError, match="copper_channel_battery_hash must match stable_hash"):
        generate_copper_channel_battery_receipt(invalid)


def test_export_rejects_non_battery_artifact() -> None:
    with pytest.raises(ValueError, match="artifact must be a CopperChannelBatteryResult"):
        export_copper_channel_battery_bytes(object())
