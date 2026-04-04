"""Tests for v137.0.16 Physics-Aware Music Video Composition Engine.

Scope:
- dataclass + export schema
- determinism + stable hashing
- invariant bounds and symbolic trace
- ordering + monotonicity
- architecture purity
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

import qec.analysis.physics_music_video_composition_engine as engine
from qec.analysis.physics_music_video_composition_engine import (
    AudioTimelineCue,
    MusicVideoComposition,
    PHYSICS_VIDEO_COMPOSITION_VERSION,
    PHI_SHELLS,
    PhysicsVideoLedger,
    VALID_PHYSICS_MODES,
    VideoFrame,
    build_music_video_composition,
    build_physics_scene_segments,
    build_physics_video_ledger,
    build_video_frame_timeline,
    export_physics_video_bundle,
    export_physics_video_ledger,
    extract_audio_timeline_cues,
    extract_visual_scene_cues,
)


def _make_inputs() -> tuple[tuple[AudioTimelineCue, ...], tuple]:
    audio = extract_audio_timeline_cues(
        [1.0, 1.2, 1.9, 2.4, 3.9, 4.1, 6.7, 6.9], start_tick=2
    )
    visual = extract_visual_scene_cues(
        [0.3, 0.8, 1.5, 2.0, 1.1, 0.9, 1.7, 2.2], start_tick=2
    )
    return audio, visual


def _make_composition(ticks_per_segment: int = 3) -> MusicVideoComposition:
    audio, visual = _make_inputs()
    return build_music_video_composition(
        audio, visual, ticks_per_segment=ticks_per_segment
    )


# ---------------------------------------------------------------------------
# GROUP A — dataclass + export tests
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _FrozenSentinel:
    value: int


@pytest.mark.parametrize(
    "obj,field,value",
    [
        (lambda: extract_audio_timeline_cues([1.0])[0], "energy", 9.0),
        (lambda: extract_visual_scene_cues([1.0])[0], "intensity", 9.0),
        (
            lambda: build_physics_scene_segments(
                extract_audio_timeline_cues([1.0]), extract_visual_scene_cues([1.0])
            )[0],
            "energy",
            9.0,
        ),
        (
            lambda: build_video_frame_timeline(
                build_physics_scene_segments(
                    extract_audio_timeline_cues([1.0, 2.0]),
                    extract_visual_scene_cues([0.1, 0.2]),
                )
            )[0],
            "energy",
            9.0,
        ),
        (lambda: _make_composition(), "frame_count", 999),
        (lambda: build_physics_video_ledger((_make_composition(),)), "composition_count", 2),
    ],
)
def test_frozen_dataclass_immutability(obj, field, value) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj(), field, value)


@pytest.mark.parametrize(
    "cls,expected_fields",
    [
        (AudioTimelineCue, {"cue_index", "tick", "energy", "phi_shell", "stable_hash", "version"}),
        (VideoFrame, {"frame_index", "tick", "energy", "phi_shell", "physics_mode", "stable_hash", "version"}),
        (MusicVideoComposition, {
            "segments", "frames", "segment_count", "frame_count",
            "phi_timing_lock_score", "triality_mesh_score", "ouroboros_loop_score",
            "demoscene_clock_score", "symbolic_trace", "stable_hash", "version"
        }),
        (PhysicsVideoLedger, {"compositions", "composition_count", "stable_hash", "version"}),
    ],
)
def test_dataclass_field_integrity(cls, expected_fields) -> None:
    names = {f.name for f in dataclasses.fields(cls)}
    assert expected_fields.issubset(names)


def test_export_physics_video_bundle_schema_keys() -> None:
    bundle = export_physics_video_bundle(_make_composition())
    required = {
        "demoscene_clock_score", "frame_count", "frames", "ouroboros_loop_score",
        "phi_timing_lock_score", "segment_count", "segments", "stable_hash",
        "symbolic_trace", "triality_mesh_score", "version",
    }
    assert required.issubset(set(bundle.keys()))


@pytest.mark.parametrize(
    "frame_key",
    ["energy", "frame_index", "physics_mode", "stable_hash"],
)
def test_export_bundle_frame_schema(frame_key: str) -> None:
    frame = export_physics_video_bundle(_make_composition())["frames"][0]
    assert frame_key in frame


@pytest.mark.parametrize(
    "segment_key",
    [
        "end_tick", "energy", "physics_mode", "stable_hash", "uff_restore_term",
    ],
)
def test_export_bundle_segment_schema(segment_key: str) -> None:
    segment = export_physics_video_bundle(_make_composition())["segments"][0]
    assert segment_key in segment


def test_export_physics_video_ledger_schema_keys() -> None:
    ledger = build_physics_video_ledger((_make_composition(), _make_composition(2)))
    exported = export_physics_video_ledger(ledger)
    required = {
        "composition_count", "compositions", "stable_hash", "version"
    }
    assert required.issubset(set(exported.keys()))


def test_canonical_json_stability_bundle() -> None:
    payload = export_physics_video_bundle(_make_composition())
    j1 = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    j2 = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert j1 == j2


def test_canonical_json_stability_ledger() -> None:
    ledger = build_physics_video_ledger((_make_composition(), _make_composition(2)))
    payload = export_physics_video_ledger(ledger)
    j1 = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    j2 = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert j1 == j2


# ---------------------------------------------------------------------------
# GROUP B — determinism + stable hash
# ---------------------------------------------------------------------------


def test_same_input_same_hash_composition() -> None:
    c1 = _make_composition()
    c2 = _make_composition()
    assert c1.stable_hash == c2.stable_hash


def test_20_run_determinism_hash() -> None:
    ref = _make_composition().stable_hash
    for _ in range(20):
        assert _make_composition().stable_hash == ref


def test_20_run_determinism_export_json() -> None:
    ref = json.dumps(export_physics_video_bundle(_make_composition()), sort_keys=True)
    for _ in range(20):
        got = json.dumps(export_physics_video_bundle(_make_composition()), sort_keys=True)
        assert got == ref


@pytest.mark.slow
def test_100_run_determinism_hash_soak() -> None:
    ref = _make_composition().stable_hash
    for _ in range(100):
        assert _make_composition().stable_hash == ref


@pytest.mark.slow
def test_100_run_determinism_export_json_soak() -> None:
    ref = json.dumps(export_physics_video_bundle(_make_composition()), sort_keys=True)
    for _ in range(100):
        got = json.dumps(export_physics_video_bundle(_make_composition()), sort_keys=True)
        assert got == ref


def test_ledger_hash_identity_same_compositions() -> None:
    c = _make_composition()
    l1 = build_physics_video_ledger((c, c))
    l2 = build_physics_video_ledger((c, c))
    assert l1.stable_hash == l2.stable_hash


@pytest.mark.parametrize(
    "builder",
    [
        lambda: extract_audio_timeline_cues([1.0])[0],
        lambda: extract_visual_scene_cues([1.0])[0],
        lambda: build_physics_scene_segments(
            extract_audio_timeline_cues([1.0]), extract_visual_scene_cues([1.0])
        )[0],
        lambda: build_video_frame_timeline(
            build_physics_scene_segments(
                extract_audio_timeline_cues([1.0]), extract_visual_scene_cues([1.0])
            )
        )[0],
        lambda: _make_composition(),
    ],
)
def test_stable_hash_field_presence(builder) -> None:
    assert hasattr(builder(), "stable_hash")
    assert isinstance(builder().stable_hash, str)
    assert len(builder().stable_hash) == 64


@pytest.mark.parametrize(
    "audio_a,audio_b",
    [
        ([1.0, 1.2, 1.9], [1.0, 1.2, 2.0]),
        ([1.0, 1.2, 1.9], [1.0, 1.4, 1.9]),
        ([1.0, 1.2, 1.9], [0.9, 1.2, 1.9]),
        ([1.0, 1.2, 1.9], [1.0, 1.2, 1.9, 2.4]),
    ],
)
def test_different_input_different_hash(audio_a, audio_b) -> None:
    v = [0.2, 0.4, 0.6, 0.8]
    c1 = build_music_video_composition(
        extract_audio_timeline_cues(audio_a), extract_visual_scene_cues(v[: len(audio_a)]), ticks_per_segment=2
    )
    c2 = build_music_video_composition(
        extract_audio_timeline_cues(audio_b), extract_visual_scene_cues(v[: len(audio_b)]), ticks_per_segment=2
    )
    assert c1.stable_hash != c2.stable_hash


@pytest.mark.parametrize("ticks_per_segment", [1, 3, 5])
def test_hash_changes_with_timing_partition(ticks_per_segment: int) -> None:
    c = _make_composition(ticks_per_segment=ticks_per_segment)
    if ticks_per_segment != 3:
        assert c.stable_hash != _make_composition(3).stable_hash


# ---------------------------------------------------------------------------
# GROUP C — invariant score bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ticks_per_segment", [1, 3, 6])
def test_phi_video_timing_lock_bounds(ticks_per_segment: int) -> None:
    c = _make_composition(ticks_per_segment)
    assert 0.0 <= c.phi_timing_lock_score <= 1.0


@pytest.mark.parametrize("ticks_per_segment", [1, 3, 6])
def test_e8_scene_triality_mesh_bounds(ticks_per_segment: int) -> None:
    c = _make_composition(ticks_per_segment)
    assert 0.0 <= c.triality_mesh_score <= 1.0


@pytest.mark.parametrize("ticks_per_segment", [1, 3, 6])
def test_ouroboros_frame_loop_bounds(ticks_per_segment: int) -> None:
    c = _make_composition(ticks_per_segment)
    assert 0.0 <= c.ouroboros_loop_score <= 1.0


@pytest.mark.parametrize("ticks_per_segment", [1, 3, 6])
def test_demoscene_runtime_clock_bounds(ticks_per_segment: int) -> None:
    c = _make_composition(ticks_per_segment)
    assert 0.0 <= c.demoscene_clock_score <= 1.0


@pytest.mark.parametrize(
    "token",
    [
        "PHI_VIDEO_TIMING_LOCK",
        "E8_SCENE_TRIALITY_MESH",
        "OUROBOROS_FRAME_LOOP",
        "DEMOSCENE_RUNTIME_CLOCK",
    ],
)
def test_symbolic_trace_contains_all_invariants(token: str) -> None:
    assert token in _make_composition().symbolic_trace


@pytest.mark.parametrize("ticks_per_segment", [1, 3, 6])
def test_scores_export_roundtrip_consistency(ticks_per_segment: int) -> None:
    c = _make_composition(ticks_per_segment)
    b = export_physics_video_bundle(c)
    assert b["phi_timing_lock_score"] == c.phi_timing_lock_score
    assert b["triality_mesh_score"] == c.triality_mesh_score
    assert b["ouroboros_loop_score"] == c.ouroboros_loop_score
    assert b["demoscene_clock_score"] == c.demoscene_clock_score


# ---------------------------------------------------------------------------
# GROUP D — ordering + monotonicity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ticks_per_segment", [1, 3, 5])
def test_tick_monotonicity(ticks_per_segment: int) -> None:
    ticks = [f.tick for f in _make_composition(ticks_per_segment).frames]
    assert ticks == sorted(ticks)


@pytest.mark.parametrize("ticks_per_segment", [1, 3, 5])
def test_frame_index_monotonicity(ticks_per_segment: int) -> None:
    indices = [f.frame_index for f in _make_composition(ticks_per_segment).frames]
    assert indices == list(range(len(indices)))


@pytest.mark.parametrize("ticks_per_segment", [1, 3, 5])
def test_segment_ordering_non_overlapping(ticks_per_segment: int) -> None:
    segments = _make_composition(ticks_per_segment).segments
    for i in range(1, len(segments)):
        assert segments[i - 1].end_tick < segments[i].start_tick


@pytest.mark.parametrize("segment_index", range(10))
def test_deterministic_mode_cycling(segment_index: int) -> None:
    expected = VALID_PHYSICS_MODES[segment_index % len(VALID_PHYSICS_MODES)]
    audio = extract_audio_timeline_cues([1.0] * (segment_index + 1))
    visual = extract_visual_scene_cues([1.0] * (segment_index + 1))
    segs = build_physics_scene_segments(audio, visual, ticks_per_segment=1)
    assert segs[segment_index].physics_mode == expected


@pytest.mark.parametrize("energy", [0.0, 0.9, 1.7, 4.0, 7.0])
def test_phi_shell_quantization_consistency(energy: float) -> None:
    cues_a = extract_audio_timeline_cues([energy])
    cues_b = extract_audio_timeline_cues([energy])
    assert cues_a[0].phi_shell == cues_b[0].phi_shell
    assert cues_a[0].phi_shell in PHI_SHELLS


@pytest.mark.parametrize("ticks_per_segment", [1, 3, 5])
def test_segment_ticks_cover_frame_ticks(ticks_per_segment: int) -> None:
    c = _make_composition(ticks_per_segment)
    frame_ticks = {f.tick for f in c.frames}
    seg_ticks = set()
    for seg in c.segments:
        seg_ticks.update(range(seg.start_tick, seg.end_tick + 1))
    assert frame_ticks == seg_ticks


# ---------------------------------------------------------------------------
# GROUP E — architecture purity
# ---------------------------------------------------------------------------


def test_no_import_from_decoder_in_source() -> None:
    src = inspect.getsource(engine)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("qec.decoder")
            assert not node.module.startswith("src.qec.decoder")


def test_layer4_isolation_by_path() -> None:
    assert "qec/analysis/" in engine.__file__.replace("\\", "/")


def test_controlled_import_diff_purity_no_new_decoder_modules() -> None:
    before = set(sys.modules.keys())
    importlib.reload(engine)
    after = set(sys.modules.keys())
    new_modules = after - before
    assert not any(name.startswith("qec.decoder") for name in new_modules)


@pytest.mark.parametrize("forbidden", ["random", "secrets", "uuid", "time"])
def test_no_randomness_imports(forbidden: str) -> None:
    src = inspect.getsource(engine)
    tree = ast.parse(src)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0])
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])
    assert forbidden not in imported_modules


def test_import_idempotency() -> None:
    mod = importlib.reload(engine)
    c1 = mod.build_music_video_composition(*_make_inputs(), ticks_per_segment=3)
    c2 = mod.build_music_video_composition(*_make_inputs(), ticks_per_segment=3)
    assert c1.stable_hash == c2.stable_hash


def test_version_exact() -> None:
    assert PHYSICS_VIDEO_COMPOSITION_VERSION == "v137.0.16"


def test_module_file_exists() -> None:
    assert Path(engine.__file__).exists()
