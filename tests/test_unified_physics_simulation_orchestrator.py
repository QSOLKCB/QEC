"""Tests for v137.1.1 unified physics simulation orchestrator hardening."""

from __future__ import annotations

import ast
import inspect
import json

import qec.analysis.unified_physics_simulation_orchestrator as orchestrator
from qec.analysis.unified_physics_simulation_orchestrator import (
    UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
    build_unified_physics_simulation_orchestrator,
    export_unified_physics_simulation_bundle,
)


def _fixture_rows():
    frames = (
        {"frame_index": 2, "tick": 11, "energy": 0.5, "phi_shell": 1.0, "physics_mode": "A"},
        {"frame_index": 0, "tick": 10, "energy": 0.7, "phi_shell": 1.6, "physics_mode": "B"},
        {"frame_index": 1, "tick": 10, "energy": 0.6, "phi_shell": 2.6, "physics_mode": "C"},
    )
    states = (
        {"tick_index": 2, "source_tick": 11, "particle_energy": 0.5, "transition_energy": 0.1, "feedback_term": 0.2},
        {"tick_index": 0, "source_tick": 10, "particle_energy": 0.7, "transition_energy": 0.2, "feedback_term": 0.3},
        {"tick_index": 1, "source_tick": 10, "particle_energy": 0.6, "transition_energy": 0.3, "feedback_term": 0.4},
    )
    sync_rows = (
        {"pair_index": 2, "invariant_tick": 11, "phi_shell_timing_alignment": 0.9, "e8_transition_timing_consistency": 0.8, "timestamp_token": "F2|S2|T11"},
        {"pair_index": 0, "invariant_tick": 10, "phi_shell_timing_alignment": 0.8, "e8_transition_timing_consistency": 0.9, "timestamp_token": "F0|S0|T10"},
        {"pair_index": 1, "invariant_tick": 10, "phi_shell_timing_alignment": 0.85, "e8_transition_timing_consistency": 0.95, "timestamp_token": "F1|S1|T10"},
    )
    return frames, states, sync_rows


def test_version_exact():
    assert UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION == "v137.1.1"


def test_fail_fast_frames_states_length_mismatch():
    frames, states, sync_rows = _fixture_rows()
    try:
        build_unified_physics_simulation_orchestrator(frames[:-1], states, sync_rows)
        assert False
    except ValueError as exc:
        assert "frames/states length mismatch" in str(exc)


def test_fail_fast_frames_sync_rows_length_mismatch():
    frames, states, sync_rows = _fixture_rows()
    try:
        build_unified_physics_simulation_orchestrator(frames, states, sync_rows[:-1])
        assert False
    except ValueError as exc:
        assert "frames/sync_rows length mismatch" in str(exc)


def test_100_run_ledger_hash_identity_replay_audit():
    frames, states, sync_rows = _fixture_rows()
    ref = build_unified_physics_simulation_orchestrator(frames, states, sync_rows).stable_hash
    for _ in range(100):
        got = build_unified_physics_simulation_orchestrator(frames, states, sync_rows).stable_hash
        assert got == ref


def test_replay_identity_stability_replay_audit():
    frames, states, sync_rows = _fixture_rows()
    ref = build_unified_physics_simulation_orchestrator(frames, states, sync_rows)
    for _ in range(40):
        got = build_unified_physics_simulation_orchestrator(frames, states, sync_rows)
        assert got.replay_identity == ref.replay_identity
        assert got.replay_identity == got.stable_hash


def test_canonical_json_byte_identity_replay_audit():
    frames, states, sync_rows = _fixture_rows()
    ref = build_unified_physics_simulation_orchestrator(frames, states, sync_rows).to_canonical_json().encode("utf-8")
    for _ in range(100):
        got = build_unified_physics_simulation_orchestrator(frames, states, sync_rows).to_canonical_json().encode("utf-8")
        assert got == ref


def test_export_bundle_stability_replay_audit():
    frames, states, sync_rows = _fixture_rows()
    ref = json.dumps(
        export_unified_physics_simulation_bundle(
            build_unified_physics_simulation_orchestrator(frames, states, sync_rows)
        ),
        sort_keys=True,
        separators=(",", ":"),
    )
    for _ in range(100):
        got = json.dumps(
            export_unified_physics_simulation_bundle(
                build_unified_physics_simulation_orchestrator(frames, states, sync_rows)
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        assert got == ref


def test_invariant_scores_reused_in_symbolic_trace():
    frames, states, sync_rows = _fixture_rows()
    ledger = build_unified_physics_simulation_orchestrator(frames, states, sync_rows)
    for name, score in ledger.invariant_scores.items():
        assert f"{name}={score:.6f}" in ledger.symbolic_trace


def test_dynamic_test_count_guard():
    source = inspect.getsource(__import__(__name__))
    assert source.count("def test_") >= 10


def test_ast_randomness_import_guard():
    tree = ast.parse(inspect.getsource(orchestrator))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    for forbidden in ("random", "secrets", "uuid", "time"):
        assert forbidden not in imports
