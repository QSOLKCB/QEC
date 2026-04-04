"""Tests for v137.1.2 Unified Physics Simulation Orchestrator.

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
    verify_drift_provenance_roundtrip,
    verify_replay_bundle_roundtrip,
)


def _fixture_inputs():
    beats = [1.0, 1.4, 2.2, 2.9, 3.7, 4.4]
    intensities = [0.5, 0.8, 1.1, 1.6, 1.9, 2.3]
    return beats, intensities


# A) API + version

def test_version_constant():
    assert UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION == "v137.1.4"


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
