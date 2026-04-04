"""Tests for v137.1.6 Unified Physics Simulation Orchestrator.

Coverage goals:
- cross-module replay integrity
- drift scoring
- symbolic memory trace consistency
- mismatch/divergence hardening
- determinism and randomness guards
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path

import pytest

from qec.analysis.unified_physics_simulation_orchestrator import (
    UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
    RuntimeStabilitySnapshot,
    StabilityConvergenceReport,
    RuntimeStabilityLedger,
    RepairSuggestion,
    RepairSuggestionBundle,
    RepairSuggestionLedger,
    audit_cross_module_replay_integrity,
    CompressedReplayBundle,
    DriftProvenanceLedger,
    ReplayHashChain,
    ReplaySnapshotDelta,
    attribute_drift_source,
    build_drift_provenance_report,
    build_replay_hash_chain,
    compare_replay_cycles,
    compress_replay_snapshots,
    compute_orchestrator_drift_score,
    decompress_replay_snapshots,
    export_drift_provenance_bundle,
    export_compressed_replay_bundle,
    validate_symbolic_memory_trace,
    export_repair_suggestion_bundle,
    generate_repair_suggestions,
    observe_runtime_stability,
    track_drift_convergence,
    analyze_repair_suggestion_stability,
    export_runtime_stability_bundle,
    verify_runtime_stability_roundtrip,
    map_drift_to_repair_actions,
    rank_repair_candidates,
    verify_drift_provenance_roundtrip,
    verify_repair_bundle_roundtrip,
    verify_replay_bundle_roundtrip,
)


def _fixture_inputs():
    beats = [1.0, 1.4, 2.2, 2.9, 3.7, 4.4]
    intensities = [0.5, 0.8, 1.1, 1.6, 1.9, 2.3]
    return beats, intensities


# A) API + version

def test_version_constant():
    assert UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION == "v137.1.6"


def test_audit_returns_artifact_with_expected_fields():
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(beats, intensities)
    assert artifact.replay_cycles == 8
    assert len(artifact.snapshots) == 8
    assert isinstance(artifact.stable_hash, str)
    assert isinstance(artifact.replay_identity, str)


# B) cross-module replay integrity

def test_100_run_replay_drift_soak_hash_identity():
    beats, intensities = _fixture_inputs()
    ref = audit_cross_module_replay_integrity(
        beats,
        intensities,
        replay_cycles=100,
        symbolic_trace="DEMO|E8|OURO|PHI",
    )
    for _ in range(100):
        got = audit_cross_module_replay_integrity(
            beats,
            intensities,
            replay_cycles=100,
            symbolic_trace="DEMO|E8|OURO|PHI",
        )
        assert got.stable_hash == ref.stable_hash
        assert got.composition_stable_hash == ref.composition_stable_hash
        assert got.simulation_stable_hash == ref.simulation_stable_hash
        assert got.sync_stable_hash == ref.sync_stable_hash


def test_cross_module_hashes_constant_across_cycles():
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=12)
    composition = {s.composition_stable_hash for s in artifact.snapshots}
    simulation = {s.simulation_stable_hash for s in artifact.snapshots}
    sync = {s.sync_stable_hash for s in artifact.snapshots}
    orchestrator = {s.orchestrator_stable_hash for s in artifact.snapshots}
    assert len(composition) == 1
    assert len(simulation) == 1
    assert len(sync) == 1
    assert len(orchestrator) == 1


@pytest.mark.parametrize(
    "field",
    ["composition_stable_hash", "simulation_stable_hash", "sync_stable_hash", "orchestrator_stable_hash"],
)
def test_cross_module_mismatch_fixtures_detected(field):
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=4)
    # All cycle hashes for this field are identical in a deterministic replay → zero drift.
    field_hashes = [getattr(s, field) for s in artifact.snapshots]
    assert compute_orchestrator_drift_score(field_hashes) == 0.0
    # Inject a single-hash mismatch and verify drift detection flags it.
    mismatch_hashes = list(field_hashes)
    mismatch_hashes[0] = "x" + mismatch_hashes[0][1:]
    assert compute_orchestrator_drift_score(mismatch_hashes) > 0.0


# C) drift score

@pytest.mark.parametrize(
    "hashes,expected",
    [
        (("a",), 0.0),
        (("a", "a"), 0.0),
        (("a", "b"), 1.0),
        (("a", "a", "b"), 0.5),
        (("a", "b", "c", "d"), 1.0),
    ],
)
def test_compute_orchestrator_drift_score_cases(hashes, expected):
    got = compute_orchestrator_drift_score(hashes)
    assert got == expected


def test_artifact_drift_score_is_bounded():
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=16)
    assert 0.0 <= artifact.orchestrator_drift_score <= 1.0


# D) symbolic memory trace consistency

@pytest.mark.parametrize(
    "symbolic_trace,expected",
    [
        ("A|B|C", True),
        ("C|B|A", True),
        ("A|A|B|C", True),
    ],
)
def test_validate_symbolic_memory_trace_divergence(symbolic_trace, expected):
    sync = {"symbolic_trace_timestamp_map": {"A": [1], "B": [2], "C": [3]}}
    assert validate_symbolic_memory_trace(symbolic_trace, sync) is expected


@pytest.mark.parametrize("symbolic_trace", ["A|B", "A|B|C|D", ""])
def test_validate_symbolic_memory_trace_divergence_raises(symbolic_trace):
    sync = {"symbolic_trace_timestamp_map": {"A": [1], "B": [2], "C": [3]}}
    with pytest.raises(ValueError, match="symbolic trace divergence|at least one token"):
        validate_symbolic_memory_trace(symbolic_trace, sync)


def test_validate_symbolic_memory_trace_invalid_map_raises():
    with pytest.raises(ValueError, match="must be a mapping"):
        validate_symbolic_memory_trace("A|B|C", {"symbolic_trace_timestamp_map": 1})


def test_symbolic_trace_valid_flag_round_trip():
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(
        beats,
        intensities,
        symbolic_trace="A|B|C",
        replay_cycles=3,
    )
    assert artifact.symbolic_trace_valid is True


# E) invariant stability bounds (high-count deterministic matrix)

@pytest.mark.parametrize("run", range(110))
def test_invariant_stability_bounds(run):
    beats, intensities = _fixture_inputs()
    trace = "PHI|E8|OURO|DEMO" if (run % 2 == 0) else "DEMO|OURO|E8|PHI"
    artifact = audit_cross_module_replay_integrity(
        beats,
        intensities,
        replay_cycles=5,
        start_tick=run % 3,
        ticks_per_segment=2 + (run % 3),
        symbolic_trace=trace,
    )
    assert 0.0 <= artifact.orchestrator_drift_score <= 1.0
    assert len(artifact.composition_stable_hash) == 64
    assert len(artifact.simulation_stable_hash) == 64
    assert len(artifact.sync_stable_hash) == 64


# F) AST/randomness and layer checks

def test_module_does_not_import_decoder_or_channel_layers():
    mod = importlib.import_module("qec.analysis.unified_physics_simulation_orchestrator")
    tree = ast.parse(inspect.getsource(mod))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith("qec.decoder") or name.startswith("qec.channel"):
                    bad.append(name)
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if name.startswith("qec.decoder") or name.startswith("qec.channel"):
                bad.append(name)
    assert bad == []


def test_ast_randomness_guard():
    path = Path("src/qec/analysis/unified_physics_simulation_orchestrator.py")
    src = path.read_text(encoding="utf-8")
    assert "random." not in src
    assert "np.random" not in src


def test_same_input_same_bytes():
    beats, intensities = _fixture_inputs()
    a = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=7)
    b = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=7)
    assert json.dumps(a.__dict__, sort_keys=True, default=str, separators=(",", ":")) == json.dumps(
        b.__dict__, sort_keys=True, default=str, separators=(",", ":")
    )


def test_replay_cycles_validation():
    beats, intensities = _fixture_inputs()
    with pytest.raises(ValueError, match="replay_cycles must be >= 1"):
        audit_cross_module_replay_integrity(beats, intensities, replay_cycles=0)


# G) replay compression layer

def test_compression_decompression_roundtrip_snapshots():
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=10)
    bundle = compress_replay_snapshots(artifact)
    rebuilt = decompress_replay_snapshots(bundle)
    assert rebuilt == artifact.snapshots


def test_byte_identical_reconstruction_from_export():
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=6)
    bundle = compress_replay_snapshots(artifact)
    rebuilt = decompress_replay_snapshots(bundle)
    lhs = json.dumps([x.__dict__ for x in artifact.snapshots], sort_keys=True, separators=(",", ":"))
    rhs = json.dumps([x.__dict__ for x in rebuilt], sort_keys=True, separators=(",", ":"))
    assert lhs.encode("utf-8") == rhs.encode("utf-8")


def test_hash_chain_determinism():
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=9)
    chain_a = build_replay_hash_chain(artifact.snapshots)
    chain_b = build_replay_hash_chain(artifact.snapshots)
    assert chain_a == chain_b
    assert chain_a.chain_digest == chain_b.chain_digest


def test_hash_chain_ordering_determinism_changes_when_reversed():
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=7)
    snaps = list(artifact.snapshots)
    varied = []
    for idx, snap in enumerate(snaps):
        varied.append(
            type(snap)(
                cycle_index=snap.cycle_index,
                composition_stable_hash=snap.composition_stable_hash,
                simulation_stable_hash=snap.simulation_stable_hash,
                sync_stable_hash=snap.sync_stable_hash,
                orchestrator_stable_hash=f"{idx:02d}{snap.orchestrator_stable_hash[2:]}",
            )
        )
    chain_a = build_replay_hash_chain(tuple(varied))
    chain_b = build_replay_hash_chain(tuple(reversed(varied)))
    assert chain_a.chain_digest != chain_b.chain_digest


def test_deduplication_correctness_tables_compacted():
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=12)
    bundle = compress_replay_snapshots(artifact)
    assert len(bundle.composition_hashes) == 1
    assert len(bundle.simulation_hashes) == 1
    assert len(bundle.sync_hashes) == 1
    assert len(bundle.orchestrator_hashes) == 1
    assert all(delta.composition_ref == 0 for delta in bundle.deltas)
    assert all(delta.simulation_ref == 0 for delta in bundle.deltas)
    assert all(delta.sync_ref == 0 for delta in bundle.deltas)
    assert all(delta.orchestrator_ref == 0 for delta in bundle.deltas)


def test_export_schema_invariant_fields_preserved():
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=8)
    bundle = compress_replay_snapshots(artifact)
    exported = export_compressed_replay_bundle(bundle)
    assert exported["symbolic_trace"] == artifact.symbolic_trace
    assert exported["symbolic_trace_valid"] == artifact.symbolic_trace_valid
    assert exported["orchestrator_drift_score"] == artifact.orchestrator_drift_score
    assert exported["composition_stable_hash"] == artifact.composition_stable_hash
    assert exported["simulation_stable_hash"] == artifact.simulation_stable_hash
    assert exported["sync_stable_hash"] == artifact.sync_stable_hash


def test_verify_replay_bundle_roundtrip_true():
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=11)
    assert verify_replay_bundle_roundtrip(artifact) is True


def test_100_run_roundtrip_stability():
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=5)
    reference_bundle = compress_replay_snapshots(artifact)
    ref_json = json.dumps(reference_bundle.to_dict(), sort_keys=True, separators=(",", ":"))
    for _ in range(100):
        bundle = compress_replay_snapshots(artifact)
        rebuilt = decompress_replay_snapshots(bundle)
        assert rebuilt == artifact.snapshots
        assert json.dumps(bundle.to_dict(), sort_keys=True, separators=(",", ":")) == ref_json


def test_compressed_bundle_stable_hash_determinism():
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=13)
    a = compress_replay_snapshots(artifact)
    b = compress_replay_snapshots(artifact)
    assert a.stable_hash == b.stable_hash
    assert a.replay_identity == b.replay_identity


def test_cross_snapshot_delta_correctness():
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=4)
    bundle = compress_replay_snapshots(artifact)
    assert tuple(delta.cycle_index for delta in bundle.deltas) == (0, 1, 2, 3)
    assert all(isinstance(delta, ReplaySnapshotDelta) for delta in bundle.deltas)


def test_fail_fast_on_malformed_bundle_out_of_range_ref():
    bundle = CompressedReplayBundle(
        version=UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        replay_cycles=1,
        composition_hashes=("a" * 64,),
        simulation_hashes=("b" * 64,),
        sync_hashes=("c" * 64,),
        orchestrator_hashes=("d" * 64,),
        deltas=(ReplaySnapshotDelta(0, 1, 0, 0, 0),),
        symbolic_trace_valid=True,
        symbolic_trace="A|B|C",
        orchestrator_drift_score=0.0,
        composition_stable_hash="a" * 64,
        simulation_stable_hash="b" * 64,
        sync_stable_hash="c" * 64,
        hash_chain=ReplayHashChain(
            ordered_hashes=("d" * 64,),
            chain_hashes=("e" * 64,),
            chain_digest="e" * 64,
            stable_hash="f" * 64,
        ),
        stable_hash="1" * 64,
        replay_identity="1" * 64,
    )
    with pytest.raises(ValueError, match="out of range"):
        decompress_replay_snapshots(bundle)


def test_fail_fast_on_malformed_bundle_non_sequential_cycle():
    bundle = CompressedReplayBundle(
        version=UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        replay_cycles=1,
        composition_hashes=("a" * 64,),
        simulation_hashes=("b" * 64,),
        sync_hashes=("c" * 64,),
        orchestrator_hashes=("d" * 64,),
        deltas=(ReplaySnapshotDelta(2, 0, 0, 0, 0),),
        symbolic_trace_valid=True,
        symbolic_trace="A|B|C",
        orchestrator_drift_score=0.0,
        composition_stable_hash="a" * 64,
        simulation_stable_hash="b" * 64,
        sync_stable_hash="c" * 64,
        hash_chain=ReplayHashChain(
            ordered_hashes=("d" * 64,),
            chain_hashes=("e" * 64,),
            chain_digest="e" * 64,
            stable_hash="f" * 64,
        ),
        stable_hash="1" * 64,
        replay_identity="1" * 64,
    )
    with pytest.raises(ValueError, match="non-sequential"):
        decompress_replay_snapshots(bundle)


def test_ast_randomness_guard_extended():
    path = Path("src/qec/analysis/unified_physics_simulation_orchestrator.py")
    src = path.read_text(encoding="utf-8")
    assert "numpy.random" not in src
    assert "uuid" not in src
    assert "time.time" not in src


def _build_bundle(*, replay_cycles: int = 6, symbolic_trace: str = "A|B|C"):
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(
        beats,
        intensities,
        replay_cycles=replay_cycles,
        symbolic_trace=symbolic_trace,
    )
    return compress_replay_snapshots(artifact)


def _symbolic_map(seed: int = 0):
    return {
        "A": [1 + seed, 4 + seed],
        "B": [2 + seed, 5 + seed],
        "C": [3 + seed, 6 + seed],
    }


def test_attribute_drift_source_single_module():
    got = attribute_drift_source(
        composition_match=False,
        simulation_match=True,
        synchronization_match=True,
        orchestrator_match=True,
        symbolic_trace_match=True,
    )
    assert got == "composition"


def test_attribute_drift_source_cross_module():
    got = attribute_drift_source(
        composition_match=False,
        simulation_match=True,
        synchronization_match=False,
        orchestrator_match=True,
        symbolic_trace_match=True,
    )
    assert got == "cross-module"


def test_compare_replay_cycles_same_input_zero_drift():
    bundle = _build_bundle(replay_cycles=3)
    snapshot = decompress_replay_snapshots(bundle)[0]
    report = compare_replay_cycles(
        snapshot,
        snapshot,
        cycle_index=0,
        chain_digest_anchor=bundle.hash_chain.chain_digest,
        delta_table_reference="delta-ref",
        symbolic_trace_anchor="trace-anchor",
        symbolic_trace_match=True,
    )
    assert report.record.bounded_drift_score == 0.0
    assert report.drift_source == "cross-module"


def test_build_drift_provenance_report_stable_100_runs():
    ref = _build_bundle(replay_cycles=8)
    cand = _build_bundle(replay_cycles=8)
    baseline = build_drift_provenance_report(
        ref,
        cand,
        reference_symbolic_trace_timestamp_map=_symbolic_map(0),
        candidate_symbolic_trace_timestamp_map=_symbolic_map(0),
    )
    for _ in range(100):
        got = build_drift_provenance_report(
            ref,
            cand,
            reference_symbolic_trace_timestamp_map=_symbolic_map(0),
            candidate_symbolic_trace_timestamp_map=_symbolic_map(0),
        )
        assert got.stable_hash == baseline.stable_hash
        assert got.to_canonical_json() == baseline.to_canonical_json()


def test_build_drift_provenance_report_single_cycle_mismatch_attribution():
    ref = _build_bundle(replay_cycles=4)
    snapshots = list(decompress_replay_snapshots(ref))
    snapshots[2] = type(snapshots[2])(
        cycle_index=snapshots[2].cycle_index,
        composition_stable_hash=("f" + snapshots[2].composition_stable_hash[1:]),
        simulation_stable_hash=snapshots[2].simulation_stable_hash,
        sync_stable_hash=snapshots[2].sync_stable_hash,
        orchestrator_stable_hash=snapshots[2].orchestrator_stable_hash,
    )
    modified_chain = build_replay_hash_chain(tuple(snapshots))
    candidate = CompressedReplayBundle(
        version=ref.version,
        replay_cycles=ref.replay_cycles,
        composition_hashes=(snapshots[0].composition_stable_hash, snapshots[2].composition_stable_hash),
        simulation_hashes=ref.simulation_hashes,
        sync_hashes=ref.sync_hashes,
        orchestrator_hashes=ref.orchestrator_hashes,
        deltas=(
            ReplaySnapshotDelta(0, 0, 0, 0, 0),
            ReplaySnapshotDelta(1, 0, 0, 0, 0),
            ReplaySnapshotDelta(2, 1, 0, 0, 0),
            ReplaySnapshotDelta(3, 0, 0, 0, 0),
        ),
        symbolic_trace_valid=ref.symbolic_trace_valid,
        symbolic_trace=ref.symbolic_trace,
        orchestrator_drift_score=ref.orchestrator_drift_score,
        composition_stable_hash=ref.composition_stable_hash,
        simulation_stable_hash=ref.simulation_stable_hash,
        sync_stable_hash=ref.sync_stable_hash,
        hash_chain=modified_chain,
        stable_hash=ref.stable_hash,
        replay_identity=ref.replay_identity,
    )
    report = build_drift_provenance_report(
        ref,
        candidate,
        reference_symbolic_trace_timestamp_map=_symbolic_map(0),
        candidate_symbolic_trace_timestamp_map=_symbolic_map(0),
    )
    assert report.first_divergence_point == 2
    assert report.cycle_reports[2].drift_source == "composition"


def test_symbolic_trace_divergence_detection_in_provenance():
    ref = _build_bundle(replay_cycles=5)
    cand = _build_bundle(replay_cycles=5)
    ledger = build_drift_provenance_report(
        ref,
        cand,
        reference_symbolic_trace_timestamp_map=_symbolic_map(0),
        candidate_symbolic_trace_timestamp_map=_symbolic_map(10),
    )
    assert ledger.cycle_reports[0].symbolic_trace_match is False
    assert ledger.cycle_reports[0].drift_source == "symbolic_trace"


def test_export_drift_provenance_bundle_byte_identical_roundtrip():
    ref = _build_bundle(replay_cycles=5)
    cand = _build_bundle(replay_cycles=5)
    ledger = build_drift_provenance_report(
        ref,
        cand,
        reference_symbolic_trace_timestamp_map=_symbolic_map(0),
        candidate_symbolic_trace_timestamp_map=_symbolic_map(0),
    )
    lhs = json.dumps(export_drift_provenance_bundle(ledger), sort_keys=True, separators=(",", ":"))
    rhs = json.dumps(export_drift_provenance_bundle(ledger), sort_keys=True, separators=(",", ":"))
    assert lhs.encode("utf-8") == rhs.encode("utf-8")
    assert verify_drift_provenance_roundtrip(
        ledger,
        reference_bundle=ref,
        candidate_bundle=cand,
        candidate_symbolic_trace_timestamp_map=_symbolic_map(0),
    ) is True


def test_provenance_fail_fast_mismatched_cycle_counts():
    ref = _build_bundle(replay_cycles=4)
    cand = _build_bundle(replay_cycles=5)
    with pytest.raises(ValueError, match="mismatched cycle counts"):
        build_drift_provenance_report(
            ref,
            cand,
            reference_symbolic_trace_timestamp_map=_symbolic_map(0),
            candidate_symbolic_trace_timestamp_map=_symbolic_map(0),
        )


def test_provenance_fail_fast_invalid_hash_chain_reference():
    ref = _build_bundle(replay_cycles=4)
    cand = _build_bundle(replay_cycles=4)
    ledger = build_drift_provenance_report(
        ref,
        cand,
        reference_symbolic_trace_timestamp_map=_symbolic_map(0),
        candidate_symbolic_trace_timestamp_map=_symbolic_map(0),
    )
    bad = DriftProvenanceLedger(
        version=ledger.version,
        replay_cycles=ledger.replay_cycles,
        first_divergence_point=ledger.first_divergence_point,
        cycle_reports=ledger.cycle_reports,
        chain_digest_anchor="",
        delta_table_reference=ledger.delta_table_reference,
        symbolic_trace_anchor=ledger.symbolic_trace_anchor,
        stable_hash=ledger.stable_hash,
        replay_identity=ledger.replay_identity,
    )
    with pytest.raises(ValueError, match="invalid hash-chain reference"):
        verify_drift_provenance_roundtrip(
            bad,
            reference_bundle=ref,
            candidate_bundle=cand,
            candidate_symbolic_trace_timestamp_map=_symbolic_map(0),
        )


def test_provenance_fail_fast_invalid_symbolic_trace_map():
    ref = _build_bundle(replay_cycles=4)
    cand = _build_bundle(replay_cycles=4)
    ledger = build_drift_provenance_report(
        ref,
        cand,
        reference_symbolic_trace_timestamp_map=_symbolic_map(0),
        candidate_symbolic_trace_timestamp_map=_symbolic_map(0),
    )
    with pytest.raises(ValueError, match="invalid symbolic trace map"):
        verify_drift_provenance_roundtrip(
            ledger,
            reference_bundle=ref,
            candidate_bundle=cand,
            candidate_symbolic_trace_timestamp_map={"A": "bad", "B": [2], "C": [3]},
        )


def test_provenance_fail_fast_tampered_stable_hash():
    ref = _build_bundle(replay_cycles=4)
    cand = _build_bundle(replay_cycles=4)
    ledger = build_drift_provenance_report(
        ref,
        cand,
        reference_symbolic_trace_timestamp_map=_symbolic_map(0),
        candidate_symbolic_trace_timestamp_map=_symbolic_map(0),
    )
    tampered = DriftProvenanceLedger(
        version=ledger.version,
        replay_cycles=ledger.replay_cycles,
        first_divergence_point=ledger.first_divergence_point,
        cycle_reports=ledger.cycle_reports,
        chain_digest_anchor=ledger.chain_digest_anchor,
        delta_table_reference=ledger.delta_table_reference,
        symbolic_trace_anchor=ledger.symbolic_trace_anchor,
        stable_hash="tampered_hash_value",
        replay_identity=ledger.replay_identity,
    )
    with pytest.raises(ValueError, match="malformed provenance bundle"):
        verify_drift_provenance_roundtrip(
            tampered,
            reference_bundle=ref,
            candidate_bundle=cand,
            candidate_symbolic_trace_timestamp_map=_symbolic_map(0),
        )


def _build_provenance_ledger(*, replay_cycles: int = 6, seed_shift: int = 0):
    ref = _build_bundle(replay_cycles=replay_cycles)
    cand = _build_bundle(replay_cycles=replay_cycles)
    return build_drift_provenance_report(
        ref,
        cand,
        reference_symbolic_trace_timestamp_map=_symbolic_map(0),
        candidate_symbolic_trace_timestamp_map=_symbolic_map(seed_shift),
    )


def test_deterministic_repair_ranking_stability_100_run_soak():
    ledger = _build_provenance_ledger(replay_cycles=5, seed_shift=0)
    a = generate_repair_suggestions(ledger)
    for _ in range(100):
        b = generate_repair_suggestions(ledger)
        assert a.stable_hash == b.stable_hash
        assert a.to_canonical_json() == b.to_canonical_json()
        assert a.suggestion_bundle.suggestions == b.suggestion_bundle.suggestions


@pytest.mark.parametrize(
    "source_module,actions",
    [
        ("composition", ("resync composition clock", "rebuild composition snapshot delta", "verify composition hash chain")),
        ("simulation", ("rebuild simulation tick graph", "re-run state propagation audit", "verify simulation hash integrity")),
        ("synchronization", ("rebuild sync timestamp map", "verify delta anchor consistency", "revalidate symbolic trace timestamps")),
        ("orchestrator", ("rebuild cycle ledger", "recompute replay identity", "re-run provenance verification")),
        ("symbolic_trace", ("rebuild symbolic token map", "verify timestamp provenance", "compare trace divergence anchors")),
        ("cross-module", ("full replay audit recommendation", "drift provenance rerun", "cycle anchor verification")),
    ],
)
def test_map_drift_to_repair_actions_per_module(source_module, actions):
    assert map_drift_to_repair_actions(source_module) == actions


def test_map_drift_to_repair_actions_invalid_module():
    with pytest.raises(ValueError, match="invalid source module"):
        map_drift_to_repair_actions("decoder")


def test_generate_repair_suggestions_advisory_only_invariant():
    ledger = _build_provenance_ledger(replay_cycles=4)
    repairs = generate_repair_suggestions(ledger)
    assert repairs.advisory_only is True
    assert repairs.suggestion_bundle.advisory_only is True
    assert all(item.advisory_only is True for item in repairs.suggestion_bundle.suggestions)


@pytest.mark.parametrize("seed_shift", [0, 5, 10])
def test_generate_repair_suggestions_contains_required_fields(seed_shift):
    ledger = _build_provenance_ledger(replay_cycles=4, seed_shift=seed_shift)
    repairs = generate_repair_suggestions(ledger)
    assert isinstance(repairs, RepairSuggestionLedger)
    assert repairs.first_divergence_cycle == ledger.first_divergence_point
    assert repairs.provenance_stable_hash == ledger.stable_hash
    for suggestion in repairs.suggestion_bundle.suggestions:
        assert suggestion.source_module in {
            "composition",
            "simulation",
            "synchronization",
            "orchestrator",
            "symbolic_trace",
            "cross-module",
        }
        assert isinstance(suggestion.repair_action, str) and suggestion.repair_action != ""
        assert 0.0 <= suggestion.deterministic_rank_score <= 1.0
        assert suggestion.provenance_cycle_reference >= 0
        assert suggestion.first_divergence_anchor == ledger.first_divergence_point


def test_rank_repair_candidates_stable_ordering():
    items = (
        RepairSuggestion("simulation", "b", 2, 0.5, 1, 1, True, "c" * 64, "c" * 64),
        RepairSuggestion("simulation", "a", 2, 0.5, 1, 1, True, "b" * 64, "b" * 64),
        RepairSuggestion("composition", "z", 1, 0.9, 1, 1, True, "a" * 64, "a" * 64),
    )
    ranked = rank_repair_candidates(items)
    assert [x.repair_action for x in ranked] == ["z", "a", "b"]


def test_export_repair_suggestion_bundle_roundtrip_byte_identical():
    ledger = _build_provenance_ledger(replay_cycles=5)
    repairs = generate_repair_suggestions(ledger)
    exported = export_repair_suggestion_bundle(repairs.suggestion_bundle)
    lhs = json.dumps(exported, sort_keys=True, separators=(",", ":")).encode("utf-8")
    rhs = json.dumps(repairs.suggestion_bundle.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert lhs == rhs
    assert verify_repair_bundle_roundtrip(repairs.suggestion_bundle) is True


def test_fail_fast_malformed_provenance_ledger_invalid_cycle_anchor():
    ledger = _build_provenance_ledger(replay_cycles=4)
    bad = DriftProvenanceLedger(
        version=ledger.version,
        replay_cycles=ledger.replay_cycles,
        first_divergence_point=ledger.replay_cycles + 1,
        cycle_reports=ledger.cycle_reports,
        chain_digest_anchor=ledger.chain_digest_anchor,
        delta_table_reference=ledger.delta_table_reference,
        symbolic_trace_anchor=ledger.symbolic_trace_anchor,
        stable_hash=ledger.stable_hash,
        replay_identity=ledger.replay_identity,
    )
    with pytest.raises(ValueError, match="invalid cycle anchor"):
        generate_repair_suggestions(bad)


def test_fail_fast_malformed_provenance_ledger_invalid_source_module():
    ledger = _build_provenance_ledger(replay_cycles=4)
    report = ledger.cycle_reports[0]
    bad_report = type(report)(
        cycle_index=report.cycle_index,
        composition_match=report.composition_match,
        simulation_match=report.simulation_match,
        synchronization_match=report.synchronization_match,
        orchestrator_match=report.orchestrator_match,
        symbolic_trace_match=report.symbolic_trace_match,
        drift_source="decoder",
        record=report.record,
        stable_hash=report.stable_hash,
        replay_identity=report.replay_identity,
    )
    bad = DriftProvenanceLedger(
        version=ledger.version,
        replay_cycles=ledger.replay_cycles,
        first_divergence_point=ledger.first_divergence_point,
        cycle_reports=(bad_report,) + ledger.cycle_reports[1:],
        chain_digest_anchor=ledger.chain_digest_anchor,
        delta_table_reference=ledger.delta_table_reference,
        symbolic_trace_anchor=ledger.symbolic_trace_anchor,
        stable_hash=ledger.stable_hash,
        replay_identity=ledger.replay_identity,
    )
    with pytest.raises(ValueError, match="invalid source module"):
        generate_repair_suggestions(bad)


def test_fail_fast_malformed_provenance_ledger_invalid_drift_score():
    ledger = _build_provenance_ledger(replay_cycles=4)
    report = ledger.cycle_reports[0]
    record = report.record
    bad_record = type(record)(
        cycle_index=record.cycle_index,
        source_module=record.source_module,
        prior_hash=record.prior_hash,
        divergent_hash=record.divergent_hash,
        chain_digest_anchor=record.chain_digest_anchor,
        delta_table_reference=record.delta_table_reference,
        symbolic_trace_anchor=record.symbolic_trace_anchor,
        bounded_drift_score=1.5,
        stable_hash=record.stable_hash,
        replay_identity=record.replay_identity,
    )
    bad_report = type(report)(
        cycle_index=report.cycle_index,
        composition_match=report.composition_match,
        simulation_match=report.simulation_match,
        synchronization_match=report.synchronization_match,
        orchestrator_match=report.orchestrator_match,
        symbolic_trace_match=report.symbolic_trace_match,
        drift_source=report.drift_source,
        record=bad_record,
        stable_hash=report.stable_hash,
        replay_identity=report.replay_identity,
    )
    bad = DriftProvenanceLedger(
        version=ledger.version,
        replay_cycles=ledger.replay_cycles,
        first_divergence_point=ledger.first_divergence_point,
        cycle_reports=(bad_report,) + ledger.cycle_reports[1:],
        chain_digest_anchor=ledger.chain_digest_anchor,
        delta_table_reference=ledger.delta_table_reference,
        symbolic_trace_anchor=ledger.symbolic_trace_anchor,
        stable_hash=ledger.stable_hash,
        replay_identity=ledger.replay_identity,
    )
    with pytest.raises(ValueError, match="invalid drift score"):
        generate_repair_suggestions(bad)


def test_fail_fast_corrupted_repair_bundle():
    ledger = _build_provenance_ledger(replay_cycles=4)
    repairs = generate_repair_suggestions(ledger)
    bad = RepairSuggestionBundle(
        version=repairs.suggestion_bundle.version,
        suggestions=repairs.suggestion_bundle.suggestions,
        advisory_only=True,
        stable_hash="0" * 64,
        replay_identity=repairs.suggestion_bundle.replay_identity,
    )
    with pytest.raises(ValueError, match="corrupted repair bundle"):
        export_repair_suggestion_bundle(bad)


def test_ast_entropy_and_layer_purity_guards_for_repair_layer():
    """Guard deterministic purity using AST checks instead of source substrings."""

    def _attribute_path(node: ast.AST) -> str | None:
        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
        return None

    path = Path("src/qec/analysis/unified_physics_simulation_orchestrator.py")
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    import_aliases: dict[str, str] = {}
    imported_names: set[str] = set()
    attribute_paths: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
                import_aliases[alias.asname or alias.name] = alias.name
            continue

        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            for alias in node.names:
                full_name = f"{node.module}.{alias.name}"
                imported_names.add(full_name)
                import_aliases[alias.asname or alias.name] = full_name
            continue

        if isinstance(node, ast.Attribute):
            path_value = _attribute_path(node)
            if path_value is None:
                continue
            root, _, remainder = path_value.partition(".")
            normalized_root = import_aliases.get(root, root)
            normalized_path = (
                f"{normalized_root}.{remainder}" if remainder else normalized_root
            )
            attribute_paths.add(normalized_path)

    assert "random" not in imported_names
    assert "uuid" not in imported_names
    assert "qec.decoder" not in imported_names
    assert "numpy.random" not in imported_names

    assert all(
        not (path_value == "random" or path_value.startswith("random."))
        for path_value in attribute_paths
    )
    assert all(
        not (path_value == "uuid" or path_value.startswith("uuid."))
        for path_value in attribute_paths
    )
    assert all(
        not (
            path_value == "numpy.random"
            or path_value.startswith("numpy.random.")
        )
        for path_value in attribute_paths
    )
    assert all(
        not (path_value == "time.time" or path_value.startswith("time.time."))
        for path_value in attribute_paths
    )
    assert all(
        not (
            path_value == "qec.decoder" or path_value.startswith("qec.decoder.")
        )
        for path_value in attribute_paths
    )


def _build_runtime_stability_ledger(*, replay_cycles: int = 8, soak_window: int = 8, seed_shift: int = 0):
    provenance = _build_provenance_ledger(replay_cycles=replay_cycles, seed_shift=seed_shift)
    repairs = generate_repair_suggestions(provenance)
    return observe_runtime_stability(provenance, repairs, soak_window=soak_window)


def test_200_run_soak_stability_runtime_observability():
    ledger = _build_runtime_stability_ledger(replay_cycles=8, soak_window=8, seed_shift=0)
    baseline = ledger.to_canonical_json()
    for _ in range(200):
        got = _build_runtime_stability_ledger(replay_cycles=8, soak_window=8, seed_shift=0)
        assert got.to_canonical_json() == baseline
        assert got.stable_hash == ledger.stable_hash


def test_same_input_convergence_score_is_one():
    ledger = _build_runtime_stability_ledger(replay_cycles=6, soak_window=6, seed_shift=0)
    assert all(snapshot.convergence_score == 1.0 for snapshot in ledger.convergence_report.snapshots)
    assert ledger.convergence_report.mean_convergence_score == 1.0


def test_drift_divergence_lowers_convergence_score():
    ledger = _build_runtime_stability_ledger(replay_cycles=6, soak_window=6, seed_shift=10)
    assert any(snapshot.convergence_score < 1.0 for snapshot in ledger.convergence_report.snapshots)
    assert ledger.convergence_report.mean_convergence_score < 1.0


def test_advisory_rank_stability_is_deterministic():
    provenance = _build_provenance_ledger(replay_cycles=6, seed_shift=0)
    repairs = generate_repair_suggestions(provenance)
    a = analyze_repair_suggestion_stability(repairs)
    b = analyze_repair_suggestion_stability(repairs)
    assert a == b
    assert 0.0 <= a <= 1.0


def test_runtime_stability_export_roundtrip_byte_identical():
    ledger = _build_runtime_stability_ledger(replay_cycles=6, soak_window=6, seed_shift=0)
    exported = export_runtime_stability_bundle(ledger)
    lhs = json.dumps(exported, sort_keys=True, separators=(",", ":")).encode("utf-8")
    rhs = json.dumps(ledger.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert lhs == rhs
    assert verify_runtime_stability_roundtrip(ledger) is True


def test_runtime_observatory_only_invariant_true():
    ledger = _build_runtime_stability_ledger(replay_cycles=4, soak_window=4, seed_shift=0)
    assert ledger.observatory_only is True
    assert ledger.convergence_report.observatory_only is True
    assert all(snapshot.observatory_only is True for snapshot in ledger.convergence_report.snapshots)


@pytest.mark.parametrize("soak_window", [0, -1])
def test_runtime_fail_fast_invalid_soak_window(soak_window):
    provenance = _build_provenance_ledger(replay_cycles=4, seed_shift=0)
    repairs = generate_repair_suggestions(provenance)
    with pytest.raises(ValueError, match="invalid soak window"):
        observe_runtime_stability(provenance, repairs, soak_window=soak_window)


def test_runtime_fail_fast_malformed_stability_bundle():
    ledger = _build_runtime_stability_ledger(replay_cycles=4, soak_window=4, seed_shift=0)
    bad = RuntimeStabilityLedger(
        version=ledger.version,
        soak_window=ledger.soak_window,
        convergence_report=ledger.convergence_report,
        advisory_stability_score=ledger.advisory_stability_score,
        provenance_stability_score=ledger.provenance_stability_score,
        stable_hash="0" * 64,
        replay_identity=ledger.replay_identity,
        observatory_only=True,
    )
    with pytest.raises(ValueError, match="malformed stability bundle"):
        verify_runtime_stability_roundtrip(bad)


def test_runtime_fail_fast_invalid_convergence_score():
    ledger = _build_runtime_stability_ledger(replay_cycles=4, soak_window=4, seed_shift=0)
    with pytest.raises(ValueError, match="invalid convergence score"):
        StabilityConvergenceReport(
            soak_window=ledger.convergence_report.soak_window,
            snapshots=ledger.convergence_report.snapshots,
            mean_drift_score=ledger.convergence_report.mean_drift_score,
            mean_convergence_score=1.5,
            stable_hash=ledger.convergence_report.stable_hash,
            replay_identity=ledger.convergence_report.replay_identity,
            observatory_only=True,
        )


def test_runtime_fail_fast_corrupted_replay_anchor():
    ledger = _build_runtime_stability_ledger(replay_cycles=4, soak_window=4, seed_shift=0)
    snap = ledger.convergence_report.snapshots[0]
    with pytest.raises(ValueError, match="corrupted replay anchor"):
        RuntimeStabilitySnapshot(
            cycle_index=snap.cycle_index,
            soak_window=snap.soak_window,
            drift_score=snap.drift_score,
            convergence_score=snap.convergence_score,
            replay_identity=snap.replay_identity,
            advisory_rank_drift=snap.advisory_rank_drift,
            provenance_source_drift=snap.provenance_source_drift,
            first_divergence_anchor=snap.first_divergence_anchor,
            chain_digest_anchor="",
            symbolic_trace_stability=snap.symbolic_trace_stability,
            stable_hash=snap.stable_hash,
            observatory_only=True,
        )


def test_runtime_fail_fast_invalid_advisory_ranking_state():
    with pytest.raises(ValueError, match="invalid advisory ranking state"):
        analyze_repair_suggestion_stability(
            RepairSuggestionLedger(
                version=UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
                provenance_stable_hash="1" * 64,
                first_divergence_cycle=0,
                suggestion_bundle=RepairSuggestionBundle(
                    version=UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
                    suggestions=tuple(),
                    advisory_only=True,
                    stable_hash="2" * 64,
                    replay_identity="2" * 64,
                ),
                advisory_only=True,
                stable_hash="3" * 64,
                replay_identity="3" * 64,
            )
        )


def test_runtime_snapshot_output_required_fields():
    ledger = _build_runtime_stability_ledger(replay_cycles=5, soak_window=5, seed_shift=0)
    snapshot = ledger.convergence_report.snapshots[0].to_dict()
    for key in (
        "cycle_index",
        "soak_window",
        "drift_score",
        "convergence_score",
        "replay_identity",
        "advisory_rank_drift",
        "provenance_source_drift",
        "first_divergence_anchor",
        "chain_digest_anchor",
    ):
        assert key in snapshot
