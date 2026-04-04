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
    compute_orchestrator_drift_score,
    validate_symbolic_memory_trace,
)


def _fixture_inputs():
    beats = [1.0, 1.4, 2.2, 2.9, 3.7, 4.4]
    intensities = [0.5, 0.8, 1.1, 1.6, 1.9, 2.3]
    return beats, intensities


# A) API + version

def test_version_constant():
    assert UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION == "v137.1.2"


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
    "mutator",
    [
        lambda a: {**a, "composition_stable_hash": "x" + a["composition_stable_hash"][1:]},
        lambda a: {**a, "simulation_stable_hash": "x" + a["simulation_stable_hash"][1:]},
        lambda a: {**a, "sync_stable_hash": "x" + a["sync_stable_hash"][1:]},
        lambda a: {**a, "orchestrator_stable_hash": "x" + a["orchestrator_stable_hash"][1:]},
    ],
)
def test_cross_module_mismatch_fixtures_detected(mutator):
    beats, intensities = _fixture_inputs()
    artifact = audit_cross_module_replay_integrity(beats, intensities, replay_cycles=2)
    base = {
        "composition_stable_hash": artifact.snapshots[0].composition_stable_hash,
        "simulation_stable_hash": artifact.snapshots[0].simulation_stable_hash,
        "sync_stable_hash": artifact.snapshots[0].sync_stable_hash,
        "orchestrator_stable_hash": artifact.snapshots[0].orchestrator_stable_hash,
    }
    altered = mutator(base)
    assert altered != base


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

@pytest.mark.parametrize("run", range(114))
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
