"""Tests for v137.0.18 Multimodal Theory Synchronization.

Groups:
A) synchronization correctness
B) deterministic replay identity
C) invariant timeline alignment
D) repository-wide CI stabilization
E) architecture purity
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path

import pytest

from qec.analysis.demoscene_physics_simulation_core import build_simulation_ticks
from qec.analysis.physics_music_video_composition_engine import (
    build_music_video_composition,
    extract_audio_timeline_cues,
    extract_visual_scene_cues,
)
from qec.analysis.multimodal_theory_synchronization import (
    MULTIMODAL_THEORY_SYNCHRONIZATION_VERSION,
    build_multimodal_sync_ledger,
    build_symbolic_trace_timestamp_map,
    stabilize_repository_test_ordering,
    synchronize_composition_and_simulation_clocks,
    unify_invariant_timeline,
)


def _build_inputs():
    audio = extract_audio_timeline_cues([1.0, 1.3, 2.0, 2.7, 3.5, 4.2], start_tick=4)
    visual = extract_visual_scene_cues([0.6, 1.0, 1.7, 2.1, 1.4, 1.8], start_tick=4)
    composition = build_music_video_composition(audio, visual, ticks_per_segment=2)
    ticks = build_simulation_ticks(composition.frames)
    return composition.frames, ticks


# A) synchronization correctness

def test_version_constant():
    assert MULTIMODAL_THEORY_SYNCHRONIZATION_VERSION == "v137.0.18"


def test_synchronize_composition_and_simulation_clocks_non_empty():
    frames, ticks = _build_inputs()
    rows = synchronize_composition_and_simulation_clocks(frames, ticks)
    assert len(rows) > 0


@pytest.mark.parametrize(
    "field",
    [
        "pair_index",
        "frame_tick",
        "simulation_tick",
        "invariant_tick",
        "phi_shell_timing_alignment",
        "e8_transition_timing_consistency",
        "ouroboros_recurrence_clock",
        "demoscene_runtime_synchronization",
        "timestamp_token",
        "version",
    ],
)
def test_sync_row_schema(field):
    frames, ticks = _build_inputs()
    row = synchronize_composition_and_simulation_clocks(frames, ticks)[0]
    assert field in row


def test_unify_invariant_timeline_schema():
    frames, ticks = _build_inputs()
    row = unify_invariant_timeline(synchronize_composition_and_simulation_clocks(frames, ticks))[0]
    assert {
        "timeline_index",
        "invariant_tick",
        "timestamp_token",
        "phi_shell_timing_alignment",
        "e8_transition_timing_consistency",
        "ouroboros_recurrence_clock",
        "demoscene_runtime_synchronization",
        "version",
    }.issubset(set(row.keys()))


# B) deterministic replay identity

def test_50_run_determinism_hash_soak():
    frames, ticks = _build_inputs()
    ref = build_multimodal_sync_ledger(frames, ticks, "A|B|C")["stable_hash"]
    for _ in range(50):
        got = build_multimodal_sync_ledger(frames, ticks, "A|B|C")["stable_hash"]
        assert got == ref


def test_50_run_determinism_json_soak():
    frames, ticks = _build_inputs()
    ref = json.dumps(build_multimodal_sync_ledger(frames, ticks, "A|B|C"), sort_keys=True)
    for _ in range(50):
        got = json.dumps(build_multimodal_sync_ledger(frames, ticks, "A|B|C"), sort_keys=True)
        assert got == ref


def test_same_input_same_bytes_identity():
    frames, ticks = _build_inputs()
    a = build_multimodal_sync_ledger(frames, ticks, "A|B|C")
    b = build_multimodal_sync_ledger(frames, ticks, "A|B|C")
    assert json.dumps(a, sort_keys=True, separators=(",", ":")) == json.dumps(
        b, sort_keys=True, separators=(",", ":")
    )


def test_input_change_changes_hash():
    frames, ticks = _build_inputs()
    base = build_multimodal_sync_ledger(frames, ticks, "A|B|C")
    alt = build_multimodal_sync_ledger(frames, ticks, "A|B|D")
    assert base["stable_hash"] != alt["stable_hash"]


# C) invariant timeline alignment

def test_timeline_invariant_ticks_monotonic():
    frames, ticks = _build_inputs()
    timeline = unify_invariant_timeline(synchronize_composition_and_simulation_clocks(frames, ticks))
    vals = [row["invariant_tick"] for row in timeline]
    assert vals == sorted(vals)


@pytest.mark.parametrize(
    "name",
    [
        "phi_shell_timing_alignment",
        "e8_transition_timing_consistency",
        "ouroboros_recurrence_clock",
        "demoscene_runtime_synchronization",
    ],
)
def test_invariant_scores_bounded(name):
    frames, ticks = _build_inputs()
    score = build_multimodal_sync_ledger(frames, ticks, "P|E8|OURO|DEMO")["invariants"][name]
    assert 0.0 <= score <= 1.0


def test_symbolic_map_is_deterministic_and_sorted_keys():
    frames, ticks = _build_inputs()
    timeline = unify_invariant_timeline(synchronize_composition_and_simulation_clocks(frames, ticks))
    mapping = build_symbolic_trace_timestamp_map("z|a|m", timeline)
    assert tuple(mapping.keys()) == ("a", "m", "z")


# D) repository-wide CI stabilization

def test_stabilize_repository_test_ordering_is_deterministic():
    nodeids = [
        "tests/test_b.py::test_z",
        "tests/test_a.py::test_b",
        "tests/test_a.py::test_a",
        "tests/test_b.py::test_a",
    ]
    out1 = stabilize_repository_test_ordering(nodeids)
    out2 = stabilize_repository_test_ordering(list(reversed(nodeids)))
    assert out1 == out2


@pytest.mark.parametrize("run", range(12))
def test_stabilize_repository_test_ordering_repeatable(run):
    nodeids = [
        "tests/test_c.py::test_2[param]",
        "tests/test_c.py::test_1[param]",
        "tests/test_b.py::test_9",
    ]
    out = stabilize_repository_test_ordering(nodeids)
    assert out == (
        "tests/test_b.py::test_9",
        "tests/test_c.py::test_1[param]",
        "tests/test_c.py::test_2[param]",
    )


# E) architecture purity

def test_layer4_module_does_not_import_decoder_or_channel_layers():
    mod = importlib.import_module("qec.analysis.multimodal_theory_synchronization")
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


def test_sync_module_has_no_random_usage():
    mod_path = Path("src/qec/analysis/multimodal_theory_synchronization.py")
    src = mod_path.read_text(encoding="utf-8")
    assert "random." not in src
    assert "np.random" not in src
