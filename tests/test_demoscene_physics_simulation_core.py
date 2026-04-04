"""Tests for v137.0.17 Demoscene Physics Simulation Core.

Groups:
A) dataclass + export
B) determinism + stable hash
C) invariant bounds
D) monotonic tick evolution
E) architecture purity
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import sys
import inspect
import json
from pathlib import Path

import pytest

import qec.analysis.demoscene_physics_simulation_core as core
from qec.analysis.physics_music_video_composition_engine import (
    build_music_video_composition,
    extract_audio_timeline_cues,
    extract_visual_scene_cues,
)
from qec.analysis.demoscene_physics_simulation_core import (
    DEMOSCENE_PHYSICS_SIMULATION_VERSION,
    DEMOSCENE_RUNTIME_TICK_FIELD,
    E8_TRANSITION_TRIALITY_CORE,
    OUROBOROS_RUNTIME_FEEDBACK,
    PHI_STATE_PROPAGATION_LOCK,
    PhysicsSimulationDecision,
    PhysicsSimulationLedger,
    PhysicsSimulationState,
    PhysicsSimulationTick,
    TRANSITION_MODES,
    build_runtime_simulation,
    build_simulation_ledger,
    build_simulation_ticks,
    compute_transition_field,
    export_simulation_bundle,
    propagate_physics_state,
)


def _make_frames():
    audio = extract_audio_timeline_cues([1.0, 1.3, 1.8, 2.4, 3.9, 4.3, 6.2, 6.7], start_tick=3)
    visual = extract_visual_scene_cues([0.4, 0.7, 1.4, 2.1, 1.2, 1.0, 1.8, 2.0], start_tick=3)
    return build_music_video_composition(audio, visual, ticks_per_segment=2).frames


def _make_runtime() -> PhysicsSimulationLedger:
    return build_runtime_simulation(_make_frames())


# A) dataclass + export
@pytest.mark.parametrize(
    "obj,field,value",
    [
        (lambda: build_simulation_ticks(_make_frames())[0], "energy", 99.0),
        (lambda: propagate_physics_state(build_simulation_ticks(_make_frames()))[0], "particle_energy", 99.0),
        (lambda: compute_transition_field(propagate_physics_state(build_simulation_ticks(_make_frames())))[0], "transition_gain", 99.0),
        (lambda: _make_runtime(), "runtime_hash", "x"),
    ],
)
def test_frozen_dataclass_immutability(obj, field, value):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj(), field, value)


@pytest.mark.parametrize(
    "cls,fields",
    [
        (PhysicsSimulationTick, {"tick_index", "source_tick", "stable_hash", "replay_identity", "version"}),
        (PhysicsSimulationState, {"tick_index", "particle_energy", "feedback_term", "stable_hash", "replay_identity", "version"}),
        (PhysicsSimulationDecision, {"transition_mode", "transition_gain", "bounded", "stable_hash", "replay_identity", "version"}),
        (PhysicsSimulationLedger, {"ticks", "states", "decisions", "runtime_hash", "invariant_scores", "symbolic_trace", "stable_hash", "replay_identity", "version"}),
    ],
)
def test_dataclass_fields_present(cls, fields):
    assert fields.issubset({f.name for f in dataclasses.fields(cls)})


@pytest.mark.parametrize(
    "builder",
    [
        lambda: build_simulation_ticks(_make_frames())[0],
        lambda: propagate_physics_state(build_simulation_ticks(_make_frames()))[0],
        lambda: compute_transition_field(propagate_physics_state(build_simulation_ticks(_make_frames())))[0],
        lambda: _make_runtime(),
    ],
)
def test_canonical_json_export(builder):
    obj = builder()
    txt = obj.to_canonical_json()
    assert isinstance(txt, str)
    assert json.loads(txt)


def test_export_simulation_bundle_schema():
    payload = export_simulation_bundle(_make_runtime())
    required = {"ticks", "states", "decisions", "runtime_hash", "invariant_scores", "symbolic_trace", "stable_hash", "replay_identity", "version"}
    assert required.issubset(set(payload.keys()))


@pytest.mark.parametrize("k", ["tick_index", "source_tick", "physics_mode", "transition_seed", "stable_hash"])
def test_tick_export_schema(k):
    assert k in export_simulation_bundle(_make_runtime())["ticks"][0]


@pytest.mark.parametrize("k", ["particle_energy", "resonance_wave", "mesh_displacement", "transition_energy", "feedback_term"])
def test_state_export_schema(k):
    assert k in export_simulation_bundle(_make_runtime())["states"][0]


@pytest.mark.parametrize("k", ["transition_mode", "transition_gain", "bounded", "stable_hash"])
def test_decision_export_schema(k):
    assert k in export_simulation_bundle(_make_runtime())["decisions"][0]


# B) determinism + stable hash

def test_same_input_same_hash():
    assert _make_runtime().stable_hash == _make_runtime().stable_hash


def test_50_run_determinism_hash_soak():
    ref = _make_runtime().stable_hash
    for _ in range(50):
        assert _make_runtime().stable_hash == ref


def test_50_run_determinism_export_json_soak():
    ref = json.dumps(export_simulation_bundle(_make_runtime()), sort_keys=True)
    for _ in range(50):
        got = json.dumps(export_simulation_bundle(_make_runtime()), sort_keys=True)
        assert got == ref


@pytest.mark.parametrize("builder", [
    lambda: build_simulation_ticks(_make_frames())[0],
    lambda: propagate_physics_state(build_simulation_ticks(_make_frames()))[0],
    lambda: compute_transition_field(propagate_physics_state(build_simulation_ticks(_make_frames())))[0],
    lambda: _make_runtime(),
])
def test_stable_hash_identity(builder):
    obj = builder()
    assert len(obj.stable_hash) == 64
    assert obj.replay_identity == obj.stable_hash


@pytest.mark.parametrize("delta", [0.05, 0.1, 0.2, 0.4])
def test_input_change_changes_hash(delta):
    f1 = list(_make_frames())
    f2 = list(_make_frames())
    d = f2[0].__dict__.copy()
    d["energy"] = d["energy"] + delta
    f2[0] = type(f2[0])(**d)
    r1 = build_runtime_simulation(f1)
    r2 = build_runtime_simulation(f2)
    assert r1.stable_hash != r2.stable_hash


@pytest.mark.parametrize("permute", [False, True])
def test_sorting_is_stable_and_deterministic(permute):
    frames = list(_make_frames())
    if permute:
        frames = list(reversed(frames))
    r1 = build_runtime_simulation(frames)
    r2 = build_runtime_simulation(frames)
    assert r1.stable_hash == r2.stable_hash


def test_build_simulation_ticks_canonical_ordering_scrambled_input():
    scrambled = [
        {"tick": 8, "frame_index": 2, "energy": 1.0, "phi_shell": 1.0, "physics_mode": "TRIALITY_SWEEP"},
        {"tick": 7, "frame_index": 3, "energy": 1.0, "phi_shell": 1.0, "physics_mode": "TRIALITY_SWEEP"},
        {"tick": 8, "frame_index": 1, "energy": 1.0, "phi_shell": 1.0, "physics_mode": "TRIALITY_SWEEP"},
        {"tick": 7, "frame_index": 0, "energy": 1.0, "phi_shell": 1.0, "physics_mode": "TRIALITY_SWEEP"},
    ]
    ticks = build_simulation_ticks(scrambled)
    ordering = [(t.source_tick, t.frame_index) for t in ticks]
    assert ordering == [(7, 0), (7, 3), (8, 1), (8, 2)]


def test_score_phi_lock_handles_tick_state_length_mismatch():
    ticks = build_simulation_ticks(_make_frames())
    states = propagate_physics_state(ticks[:-1])
    score = core._score_phi_lock(ticks, states)
    assert 0.0 <= score <= 1.0


# C) invariant bounds
@pytest.mark.parametrize("name", [
    DEMOSCENE_RUNTIME_TICK_FIELD,
    PHI_STATE_PROPAGATION_LOCK,
    E8_TRANSITION_TRIALITY_CORE,
    OUROBOROS_RUNTIME_FEEDBACK,
])
def test_invariant_score_bounds(name):
    score = _make_runtime().invariant_scores[name]
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("token", [
    DEMOSCENE_RUNTIME_TICK_FIELD,
    PHI_STATE_PROPAGATION_LOCK,
    E8_TRANSITION_TRIALITY_CORE,
    OUROBOROS_RUNTIME_FEEDBACK,
])
def test_symbolic_trace_contains_tokens(token):
    assert token in _make_runtime().symbolic_trace


@pytest.mark.parametrize("name", [
    DEMOSCENE_RUNTIME_TICK_FIELD,
    PHI_STATE_PROPAGATION_LOCK,
    E8_TRANSITION_TRIALITY_CORE,
    OUROBOROS_RUNTIME_FEEDBACK,
])
def test_export_invariant_roundtrip(name):
    runtime = _make_runtime()
    assert export_simulation_bundle(runtime)["invariant_scores"][name] == runtime.invariant_scores[name]


# D) monotonic tick evolution
@pytest.mark.parametrize("run", range(6))
def test_tick_indices_monotonic(run):
    ticks = build_simulation_ticks(_make_frames())
    assert [t.tick_index for t in ticks] == list(range(len(ticks)))


@pytest.mark.parametrize("run", range(6))
def test_source_ticks_monotonic(run):
    ticks = build_simulation_ticks(_make_frames())
    vals = [t.source_tick for t in ticks]
    assert vals == sorted(vals)


@pytest.mark.parametrize("run", range(6))
def test_state_chain_length_matches_ticks(run):
    ticks = build_simulation_ticks(_make_frames())
    states = propagate_physics_state(ticks)
    assert len(states) == len(ticks)


@pytest.mark.parametrize("run", range(6))
def test_decision_chain_length_matches_states(run):
    ticks = build_simulation_ticks(_make_frames())
    states = propagate_physics_state(ticks)
    decisions = compute_transition_field(states)
    assert len(decisions) == len(states)


@pytest.mark.parametrize("run", range(6))
def test_transition_modes_cycle_deterministically(run):
    decisions = compute_transition_field(propagate_physics_state(build_simulation_ticks(_make_frames())))
    for i, dec in enumerate(decisions):
        assert dec.transition_mode == TRANSITION_MODES[i % len(TRANSITION_MODES)]


@pytest.mark.parametrize("run", range(6))
def test_transition_gain_non_negative(run):
    decisions = compute_transition_field(propagate_physics_state(build_simulation_ticks(_make_frames())))
    assert all(d.transition_gain >= 0.0 for d in decisions)


# E) architecture purity

def test_no_decoder_imports():
    tree = ast.parse(inspect.getsource(core))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("qec.decoder")
            assert not node.module.startswith("src.qec.decoder")


def test_layer4_path_purity():
    assert "qec/analysis/" in core.__file__.replace("\\", "/")


def test_controlled_reload_no_decoder_leak():
    before = set(sys.modules.keys())
    importlib.reload(core)
    after = set(sys.modules.keys())
    assert not any(name.startswith("qec.decoder") for name in (after - before))


@pytest.mark.parametrize("forbidden", ["random", "secrets", "uuid", "time"])
def test_no_randomness_import(forbidden):
    tree = ast.parse(inspect.getsource(core))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert forbidden not in imports


def test_version_exact():
    assert DEMOSCENE_PHYSICS_SIMULATION_VERSION == "v137.0.17"


def test_module_file_exists():
    assert Path(core.__file__).exists()


def test_build_simulation_ledger_function_roundtrip():
    ticks = build_simulation_ticks(_make_frames())
    states = propagate_physics_state(ticks)
    decisions = compute_transition_field(states)
    ledger = build_simulation_ledger(ticks, states, decisions)
    assert ledger.runtime_hash
    assert len(ledger.ticks) == len(ticks)
