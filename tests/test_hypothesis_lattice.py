from __future__ import annotations

from dataclasses import replace

import pytest

from qec.analysis.arithmetic_topology_correspondence_engine import build_arithmetic_topology_correspondence
from qec.analysis.atomic_signal_transduction_observatory import run_atomic_signal_transduction_observatory
from qec.analysis.cross_modal_replay_certification import run_cross_modal_replay_certification
from qec.analysis.e8_symmetry_projection_layer import build_e8_symmetry_projection
from qec.analysis.episodic_memory_lifting import lift_raw_records_to_episodic_memory
from qec.analysis.fragmentation_recovery_engine import recover_fragmented_compression_chain
from qec.analysis.hash_preserving_memory_compression import CompressedMemoryArtifact, compress_semantic_theme_memory
from qec.analysis.hypothesis_lattice import (
    BOUNDED_LATTICE_SCORE_RULE,
    DETERMINISTIC_LATTICE_ORDERING_RULE,
    HYPOTHESIS_LATTICE_LAW,
    REPLAY_SAFE_LATTICE_IDENTITY_RULE,
    build_hypothesis_lattice,
    export_hypothesis_lattice_bytes,
    generate_hypothesis_lattice_receipt,
)
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
            "record_id": f"theme:hypothesis-lattice-{i:03d}" if i % 2 == 0 else f"hypothesis-lattice-{i:03d}",
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
    observatory = run_atomic_signal_transduction_observatory(
        certification,
        schema_artifact=schema,
        spectral_artifact=spectral,
        telecom_artifact=telecom,
        satellite_artifact=satellite,
        rf_artifact=rf,
    )
    return schema, spectral, battery, telecom, satellite, rf, certification, observatory


def test_identical_input_produces_byte_identical_lattice_artifacts(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, _, observatory = lineage_artifacts

    artifact_a = build_hypothesis_lattice(observatory)
    artifact_b = build_hypothesis_lattice(observatory)

    assert artifact_a == artifact_b
    assert artifact_a.to_canonical_bytes() == artifact_b.to_canonical_bytes()
    assert export_hypothesis_lattice_bytes(artifact_a) == export_hypothesis_lattice_bytes(artifact_b)
    assert artifact_a.stable_hash() == artifact_a.hypothesis_lattice_hash


def test_deterministic_repeated_runs(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, _, observatory = lineage_artifacts

    artifacts = [
        export_hypothesis_lattice_bytes(
            build_hypothesis_lattice(
                observatory,
                lattice_profile="full_chain_hypothesis_lattice",
                score_fixture=(1.0, 0.0, 0.5),
            )
        )
        for _ in range(20)
    ]
    assert len(set(artifacts)) == 1


def test_stable_node_ordering_and_stable_edge_ordering(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, _, observatory = lineage_artifacts
    artifact = build_hypothesis_lattice(observatory)

    node_keys = tuple((n.node_index, n.node_kind, n.node_id) for n in artifact.lattice.nodes)
    edge_keys = tuple((e.edge_index, e.from_node_id, e.to_node_id, e.edge_id) for e in artifact.lattice.edges)

    assert node_keys == tuple(sorted(node_keys))
    assert edge_keys == tuple(sorted(edge_keys))
    assert artifact.lattice.node_count == len(artifact.lattice.nodes)
    assert artifact.lattice.edge_count == len(artifact.lattice.edges)


def test_stable_hash_invariants(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, _, observatory = lineage_artifacts
    artifact = build_hypothesis_lattice(observatory)

    assert artifact.law_invariants == (
        HYPOTHESIS_LATTICE_LAW,
        DETERMINISTIC_LATTICE_ORDERING_RULE,
        REPLAY_SAFE_LATTICE_IDENTITY_RULE,
        BOUNDED_LATTICE_SCORE_RULE,
    )
    assert artifact.stable_hash() == artifact.hypothesis_lattice_hash
    assert artifact.lattice.lattice_hash == artifact.lattice.stable_hash()
    for node in artifact.lattice.nodes:
        assert node.node_hash == node.stable_hash()
        assert node.node_id == node.node_hash
    for edge in artifact.lattice.edges:
        assert edge.edge_hash == edge.stable_hash()
        assert edge.edge_id == edge.edge_hash


def test_fail_fast_malformed_upstream_input(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, _, observatory = lineage_artifacts
    invalid = replace(observatory, atomic_signal_observatory_hash="bad-hash")

    with pytest.raises(ValueError, match="atomic_signal_observatory_hash must match stable_hash"):
        build_hypothesis_lattice(invalid)


def test_lineage_preservation(lineage_artifacts: tuple) -> None:
    schema, spectral, battery, telecom, satellite, rf, certification, observatory = lineage_artifacts

    artifact = build_hypothesis_lattice(
        observatory,
        replay_certification_artifact=certification,
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
    assert artifact.source_atomic_signal_observatory_hash == observatory.atomic_signal_observatory_hash


def test_receipt_determinism(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, _, observatory = lineage_artifacts
    artifact = build_hypothesis_lattice(observatory)

    receipt_a = generate_hypothesis_lattice_receipt(artifact)
    receipt_b = generate_hypothesis_lattice_receipt(artifact)

    assert receipt_a == receipt_b
    assert receipt_a.stable_hash() == receipt_a.receipt_hash
    assert receipt_a.to_canonical_bytes() == receipt_b.to_canonical_bytes()


def test_bounded_score_checks(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, _, observatory = lineage_artifacts
    artifact = build_hypothesis_lattice(observatory)

    scores = (
        artifact.node_consistency_score,
        artifact.edge_integrity_score,
        artifact.lineage_reasoning_score,
        artifact.causal_alignment_score,
        artifact.overall_lattice_score,
    )
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_nan_infinity_rejection(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, _, observatory = lineage_artifacts

    with pytest.raises(ValueError, match="finite"):
        build_hypothesis_lattice(observatory, score_fixture=(1.0, float("nan")))
    with pytest.raises(ValueError, match="finite"):
        build_hypothesis_lattice(observatory, score_fixture=(1.0, float("inf")))


def test_disconnected_lattice_rejection(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, _, observatory = lineage_artifacts
    isolated_windows = observatory.windows[:1]
    disconnected = replace(observatory, windows=isolated_windows, window_count=len(isolated_windows))
    disconnected = replace(disconnected, atomic_signal_observatory_hash=disconnected.stable_hash())

    with pytest.raises(ValueError, match="disconnected lattice"):
        build_hypothesis_lattice(disconnected)


def test_direct_lineage_mismatch_rejection(lineage_artifacts: tuple) -> None:
    _, _, _, telecom, satellite, rf, certification, observatory = lineage_artifacts
    wrong_satellite = replace(satellite, source_telecom_recovery_hash="f" * 64)
    wrong_satellite = replace(wrong_satellite, satellite_baseline_hash=wrong_satellite.stable_hash())

    with pytest.raises(ValueError, match="direct lineage mismatch"):
        build_hypothesis_lattice(
            observatory,
            replay_certification_artifact=certification,
            telecom_artifact=telecom,
            satellite_artifact=wrong_satellite,
            rf_artifact=rf,
        )


def test_forged_observation_lineage_rejection(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, _, observatory = lineage_artifacts
    # Forge the first observation to carry a bad lineage_chain while keeping its
    # hash internally consistent — the observatory accepts the artifact at the
    # hash level but the hypothesis lattice must catch the cross-field mismatch.
    bad_obs = replace(observatory.observations[0], lineage_chain=("forged-lineage-hash",))
    new_hash = bad_obs.stable_hash()
    bad_obs = replace(bad_obs, observation_id=new_hash, observation_hash=new_hash)
    new_observations = (bad_obs,) + observatory.observations[1:]
    new_observations = tuple(
        sorted(
            new_observations,
            key=lambda o: (o.observation_index, o.observatory_profile, o.observation_id),
        )
    )
    forged = replace(observatory, observations=new_observations)
    forged = replace(forged, atomic_signal_observatory_hash=forged.stable_hash())

    with pytest.raises(ValueError, match="lineage_chain must match source fields"):
        build_hypothesis_lattice(forged)


def test_window_observation_ids_count_rejection(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, _, observatory = lineage_artifacts
    # Forge a window with 3 observation_ids (must be exactly 2).
    bad_window = replace(
        observatory.windows[0],
        observation_ids=observatory.windows[0].observation_ids + ("extra-observation-id",),
    )
    bad_window = replace(bad_window, window_hash=bad_window.stable_hash())
    new_windows = (bad_window,) + observatory.windows[1:]
    new_windows = tuple(
        sorted(
            new_windows,
            key=lambda w: (w.window_index, w.window_id, w.window_hash),
        )
    )
    forged = replace(observatory, windows=new_windows)
    forged = replace(forged, atomic_signal_observatory_hash=forged.stable_hash())

    with pytest.raises(ValueError, match="exactly 2 observation IDs"):
        build_hypothesis_lattice(forged)


def test_window_unknown_observation_id_rejection(lineage_artifacts: tuple) -> None:
    _, _, _, _, _, _, _, observatory = lineage_artifacts
    # Forge a window with a valid count but one unknown observation ID.
    orig_ids = observatory.windows[0].observation_ids
    bad_window = replace(
        observatory.windows[0],
        observation_ids=(orig_ids[0], "unknown-observation-id-" + "a" * 44),
    )
    bad_window = replace(bad_window, window_hash=bad_window.stable_hash())
    new_windows = (bad_window,) + observatory.windows[1:]
    new_windows = tuple(
        sorted(
            new_windows,
            key=lambda w: (w.window_index, w.window_id, w.window_hash),
        )
    )
    forged = replace(observatory, windows=new_windows)
    forged = replace(forged, atomic_signal_observatory_hash=forged.stable_hash())

    with pytest.raises(ValueError, match="references unknown observation IDs"):
        build_hypothesis_lattice(forged)
