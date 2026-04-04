"""Tests for v137.1.0 Unified Physics-Simulation Orchestrator.

Groups:
A) orchestration correctness
B) determinism + stable hash
C) subsystem synchronization
D) symbolic trace memory
E) architecture purity
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path

import pytest

from qec.analysis.demoscene_physics_simulation_core import build_runtime_simulation
from qec.analysis.multimodal_theory_synchronization import build_multimodal_sync_ledger
from qec.analysis.physics_music_video_composition_engine import (
    build_music_video_composition,
    extract_audio_timeline_cues,
    extract_visual_scene_cues,
)
from qec.analysis.unified_physics_simulation_orchestrator import (
    DEMOSCENE_MASTER_CLOCK,
    E8_RUNTIME_TRIALITY_ORCHESTRATION,
    OUROBOROS_MEMORY_FEEDBACK,
    PHYSICS_ORCHESTRATION_LOCK,
    UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
    build_orchestrator_ledger,
    build_orchestrator_state_graph,
    build_symbolic_memory_trace,
    export_orchestrator_bundle,
    orchestrate_multimodal_runtime,
    synchronize_subsystem_decisions,
)


def _build_inputs():
    audio = extract_audio_timeline_cues([0.8, 1.1, 1.7, 2.2, 2.8, 3.3], start_tick=3)
    visual = extract_visual_scene_cues([0.5, 0.9, 1.4, 1.9, 1.6, 1.2], start_tick=3)
    composition = build_music_video_composition(audio, visual, ticks_per_segment=2)
    sim_ledger = build_runtime_simulation(composition.frames).to_dict()
    sync_ledger = build_multimodal_sync_ledger(composition.frames, sim_ledger["ticks"], composition.symbolic_trace)
    return composition, sim_ledger, sync_ledger


# A) orchestration correctness

def test_version_constant():
    assert UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION == "v137.1.0"


def test_orchestrate_multimodal_runtime_non_empty():
    composition, sim, sync = _build_inputs()
    ledger = orchestrate_multimodal_runtime(composition.frames, sim, sync)
    assert len(ledger.states) > 0


@pytest.mark.parametrize("name", [
    PHYSICS_ORCHESTRATION_LOCK,
    E8_RUNTIME_TRIALITY_ORCHESTRATION,
    OUROBOROS_MEMORY_FEEDBACK,
    DEMOSCENE_MASTER_CLOCK,
])
def test_required_theory_invariant_names_present(name):
    composition, sim, sync = _build_inputs()
    ledger = orchestrate_multimodal_runtime(composition.frames, sim, sync)
    assert name in ledger.invariant_scores


@pytest.mark.parametrize("field", [
    "states", "decisions", "trace_frames", "invariant_scores", "symbolic_trace", "stable_hash", "replay_identity", "version",
])
def test_export_orchestrator_bundle_schema(field):
    composition, sim, sync = _build_inputs()
    payload = export_orchestrator_bundle(orchestrate_multimodal_runtime(composition.frames, sim, sync))
    assert field in payload


# B) determinism + stable hash

@pytest.mark.parametrize("run", range(40))
def test_40_run_stable_hash_soak(run):
    composition, sim, sync = _build_inputs()
    ref = orchestrate_multimodal_runtime(composition.frames, sim, sync).stable_hash
    got = orchestrate_multimodal_runtime(composition.frames, sim, sync).stable_hash
    assert got == ref


@pytest.mark.parametrize("run", range(12))
def test_12_run_same_input_same_bytes(run):
    composition, sim, sync = _build_inputs()
    a = export_orchestrator_bundle(orchestrate_multimodal_runtime(composition.frames, sim, sync))
    b = export_orchestrator_bundle(orchestrate_multimodal_runtime(composition.frames, sim, sync))
    assert json.dumps(a, sort_keys=True, separators=(",", ":")) == json.dumps(b, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("variant", ["A|B|C", "A|B|D", "X|Y|Z", "Q|E8|M", "P|O|L"])
def test_input_change_changes_hash(variant):
    composition, sim, sync = _build_inputs()
    base = orchestrate_multimodal_runtime(composition.frames, sim, sync)
    sync2 = dict(sync)
    sync2["synchronized_rows"] = [dict(row) for row in sync["synchronized_rows"]]
    if sync2["synchronized_rows"]:
        sync2["synchronized_rows"][0]["timestamp_token"] = variant
    alt = orchestrate_multimodal_runtime(composition.frames, sim, sync2)
    assert base.stable_hash != alt.stable_hash


def test_state_graph_length_mismatch_raises():
    composition, sim, sync = _build_inputs()
    with pytest.raises(ValueError, match="length mismatch"):
        build_orchestrator_state_graph(composition.frames, sim["states"][:-1], sync["synchronized_rows"])


# C) subsystem synchronization

@pytest.mark.parametrize("run", range(15))
def test_synchronize_subsystem_decisions_repeatable(run):
    composition, sim, sync = _build_inputs()
    states = build_orchestrator_state_graph(composition.frames, sim["states"], sync["synchronized_rows"])
    d1 = synchronize_subsystem_decisions(states, sim["decisions"])
    d2 = synchronize_subsystem_decisions(states, sim["decisions"])
    assert tuple(x.stable_hash for x in d1) == tuple(x.stable_hash for x in d2)


@pytest.mark.parametrize("index", range(8))
def test_decision_deterministic_rank_sequence(index):
    composition, sim, sync = _build_inputs()
    states = build_orchestrator_state_graph(composition.frames, sim["states"], sync["synchronized_rows"])
    decisions = synchronize_subsystem_decisions(states, sim["decisions"])
    if index < len(decisions):
        assert decisions[index].deterministic_rank == index
    else:
        assert len(decisions) <= index


@pytest.mark.parametrize("run", range(6))
def test_state_graph_sorted_by_tick_and_index(run):
    composition, sim, sync = _build_inputs()
    states = build_orchestrator_state_graph(composition.frames, sim["states"], sync["synchronized_rows"])
    ordered = sorted((s.tick, s.state_index) for s in states)
    assert [(s.tick, s.state_index) for s in states] == ordered


def test_synchronize_subsystem_decisions_length_mismatch_raises():
    composition, sim, sync = _build_inputs()
    states = build_orchestrator_state_graph(composition.frames, sim["states"], sync["synchronized_rows"])
    with pytest.raises(ValueError, match="length mismatch"):
        synchronize_subsystem_decisions(states[:-1], sim["decisions"])


# D) symbolic trace memory

@pytest.mark.parametrize("run", range(10))
def test_symbolic_memory_trace_repeatable(run):
    composition, sim, sync = _build_inputs()
    states = build_orchestrator_state_graph(composition.frames, sim["states"], sync["synchronized_rows"])
    decisions = synchronize_subsystem_decisions(states, sim["decisions"])
    t1 = build_symbolic_memory_trace(states, decisions)
    t2 = build_symbolic_memory_trace(states, decisions)
    assert tuple(x.stable_hash for x in t1) == tuple(x.stable_hash for x in t2)


@pytest.mark.parametrize("run", range(6))
def test_memory_scalar_non_negative(run):
    composition, sim, sync = _build_inputs()
    states = build_orchestrator_state_graph(composition.frames, sim["states"], sync["synchronized_rows"])
    decisions = synchronize_subsystem_decisions(states, sim["decisions"])
    trace = build_symbolic_memory_trace(states, decisions)
    assert all(frame.memory_scalar >= 0.0 for frame in trace)


@pytest.mark.parametrize("run", range(6))
def test_symbolic_token_contains_sync_marker(run):
    composition, sim, sync = _build_inputs()
    states = build_orchestrator_state_graph(composition.frames, sim["states"], sync["synchronized_rows"])
    decisions = synchronize_subsystem_decisions(states, sim["decisions"])
    trace = build_symbolic_memory_trace(states, decisions)
    assert all("|" in frame.symbolic_token for frame in trace)


@pytest.mark.parametrize("run", range(6))
def test_build_orchestrator_ledger_roundtrip_dict(run):
    composition, sim, sync = _build_inputs()
    states = build_orchestrator_state_graph(composition.frames, sim["states"], sync["synchronized_rows"])
    decisions = synchronize_subsystem_decisions(states, sim["decisions"])
    trace = build_symbolic_memory_trace(states, decisions)
    ledger = build_orchestrator_ledger(states, decisions, trace)
    payload = ledger.to_dict()
    assert payload["stable_hash"] == ledger.stable_hash


def test_build_symbolic_memory_trace_length_mismatch_raises():
    composition, sim, sync = _build_inputs()
    states = build_orchestrator_state_graph(composition.frames, sim["states"], sync["synchronized_rows"])
    decisions = synchronize_subsystem_decisions(states, sim["decisions"])
    with pytest.raises(ValueError, match="length mismatch"):
        build_symbolic_memory_trace(states[:-1], decisions)


# E) architecture purity

def test_layer4_module_does_not_import_decoder_or_channel_layers():
    mod = importlib.import_module("qec.analysis.unified_physics_simulation_orchestrator")
    src = inspect.getsource(mod)
    tree = ast.parse(src)

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


def test_orchestrator_module_has_no_random_usage():
    mod_path = Path("src/qec/analysis/unified_physics_simulation_orchestrator.py")
    src = mod_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    aliases = {}
    forbidden = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name
                aliases[local] = alias.name
                if alias.name == "random" or alias.name == "numpy.random":
                    forbidden.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "random" or module == "numpy.random":
                forbidden.append(module)
            for alias in node.names:
                local = alias.asname or alias.name
                aliases[local] = f"{module}.{alias.name}" if module else alias.name
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                base = node.value.id
                target = aliases.get(base, base)
                if (target == "numpy" and node.attr == "random") or target == "numpy.random":
                    forbidden.append(f"{target}.{node.attr}")

    assert forbidden == []


def test_orchestrator_total_test_count_guard():
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    count = sum(
        1
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    )
    assert count >= 1
