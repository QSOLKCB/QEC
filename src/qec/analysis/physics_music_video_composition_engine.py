"""v137.0.16 — Physics-Aware Music Video Composition Engine.
 
Theory-coupled composition engine absorbing real semantics from:
 
  Sound_as_a_Fractal_Golden_E8_Dimension_i.pdf
  Unified_Field_Fidelity_Bridging_Computat.docx
 
Core theory constructs absorbed:
 
  1. PHI_VIDEO_TIMING_LOCK — phi-shell quantized keyframe timing
     All frame timestamps snap to the golden-ratio shell progression
     (1.0, PHI, PHI+1, 2*PHI+1, 3*PHI+2).  Linear interpolation is
     forbidden.  Bounded timing precision.
 
  2. E8_SCENE_TRIALITY_MESH — E8-coupled scene segment classification
     Scene segments are classified into one of five triality axis modes
     via (segment_index % 5).  Mirrors the E8 root-system triality.
 
  3. OUROBOROS_FRAME_LOOP — cyclic self-similarity in frame timeline
     First-half and second-half frame energy vectors are compared via
     cosine similarity.  High loopback = strong ouroboros cycle.
 
  4. DEMOSCENE_RUNTIME_CLOCK — deterministic tick-based timeline
     All frame indices are integer ticks.  No floating-point time.
     No wall-clock dependency.  Fully replay-safe.
 
Physics modes:
 
  TRIALITY_SWEEP       — E8 triality progression across segments
  PHI_SHELL_EXPANSION  — phi-shell depth expansion over time
  OUROBOROS_LOOPBACK   — cyclic return to initial state
  RESONANCE_COLLAPSE   — energy convergence to attractor
  DEMOSCENE_TRANSITION — demoscene-style hard cut between modes
 
Pipeline law:
 
  audio cues + visual cues
  -> physics scene segments
  -> video frame timeline
  -> music video composition
  -> canonical JSON + SHA-256 ledger
 
Layer 4 — Analysis.
Does not import or modify decoder internals.
Fully deterministic: no randomness, no global state, no input mutation.
 
Theory Upgrade Source:
- file: Sound_as_a_Fractal_Golden_E8_Dimension_i.pdf,
        Unified_Field_Fidelity_Bridging_Computat.docx
- concept: phi-shell timing quantization, E8 triality scene mesh,
           ouroboros frame loopback, UFF restore operator
- implementation: deterministic composition pipeline with frozen
                  dataclasses, SHA-256 ledger, symbolic traces
- invariant tested: PHI_VIDEO_TIMING_LOCK, E8_SCENE_TRIALITY_MESH,
                    OUROBOROS_FRAME_LOOP, DEMOSCENE_RUNTIME_CLOCK
"""
 
from __future__ import annotations
 
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple
 
import numpy as np
 
# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
 
PHYSICS_VIDEO_COMPOSITION_VERSION: str = "v137.0.16"
 
# ---------------------------------------------------------------------------
# Constants — theory-coupled
# ---------------------------------------------------------------------------
 
PHI: float = 1.618033988749895
 
# Phi-shell timing progression: each shell = sum of two preceding
PHI_SHELLS: Tuple[float, ...] = (1.0, 1.618, 2.618, 4.236, 6.854)
 
# E8 triality axis classes (mod-5 mapping)
TRIALITY_SWEEP: str = "TRIALITY_SWEEP"
PHI_SHELL_EXPANSION: str = "PHI_SHELL_EXPANSION"
OUROBOROS_LOOPBACK: str = "OUROBOROS_LOOPBACK"
RESONANCE_COLLAPSE: str = "RESONANCE_COLLAPSE"
DEMOSCENE_TRANSITION: str = "DEMOSCENE_TRANSITION"
 
VALID_PHYSICS_MODES: Tuple[str, ...] = (
    TRIALITY_SWEEP,
    PHI_SHELL_EXPANSION,
    OUROBOROS_LOOPBACK,
    RESONANCE_COLLAPSE,
    DEMOSCENE_TRANSITION,
)
 
# Scene classification
SCENE_INTRO: str = "SCENE_INTRO"
SCENE_BUILD: str = "SCENE_BUILD"
SCENE_CLIMAX: str = "SCENE_CLIMAX"
SCENE_DECAY: str = "SCENE_DECAY"
SCENE_OUTRO: str = "SCENE_OUTRO"
 
VALID_SCENE_CLASSES: Tuple[str, ...] = (
    SCENE_INTRO,
    SCENE_BUILD,
    SCENE_CLIMAX,
    SCENE_DECAY,
    SCENE_OUTRO,
)
 
# Precision for deterministic hashing
FLOAT_PRECISION: int = 12
 
# UFF restore operator constants
UFF_PHASE_OFFSET: float = math.pi / 2.0
 
# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------
 
 
@dataclass(frozen=True)
class AudioTimelineCue:
    """A single audio cue extracted from beat/energy data."""
    cue_index: int
    tick: int
    energy: float
    phi_shell: float
    stable_hash: str
    version: str = PHYSICS_VIDEO_COMPOSITION_VERSION
 
 
@dataclass(frozen=True)
class VisualSceneCue:
    """A single visual cue extracted from scene/image data."""
    cue_index: int
    tick: int
    intensity: float
    scene_class: str
    stable_hash: str
    version: str = PHYSICS_VIDEO_COMPOSITION_VERSION
 
 
@dataclass(frozen=True)
class PhysicsSceneSegment:
    """A physics-classified scene segment with triality mode."""
    segment_index: int
    start_tick: int
    end_tick: int
    physics_mode: str
    energy: float
    phi_timing: float
    uff_restore_term: float
    symbolic_trace: str
    stable_hash: str
    version: str = PHYSICS_VIDEO_COMPOSITION_VERSION
 
 
@dataclass(frozen=True)
class VideoFrame:
    """A single deterministic video frame in the composition timeline."""
    frame_index: int
    tick: int
    energy: float
    phi_shell: float
    physics_mode: str
    stable_hash: str
    version: str = PHYSICS_VIDEO_COMPOSITION_VERSION
 
 
@dataclass(frozen=True)
class MusicVideoComposition:
    """Complete music video composition with theory-coupled invariants."""
    segments: Tuple[PhysicsSceneSegment, ...]
    frames: Tuple[VideoFrame, ...]
    segment_count: int
    frame_count: int
    phi_timing_lock_score: float
    triality_mesh_score: float
    ouroboros_loop_score: float
    demoscene_clock_score: float
    symbolic_trace: str
    stable_hash: str
    version: str = PHYSICS_VIDEO_COMPOSITION_VERSION
 
 
@dataclass(frozen=True)
class PhysicsVideoLedger:
    """Ledger aggregating multiple compositions."""
    compositions: Tuple[MusicVideoComposition, ...]
    composition_count: int
    stable_hash: str
    version: str = PHYSICS_VIDEO_COMPOSITION_VERSION
 
 
# ---------------------------------------------------------------------------
# Deterministic hashing helpers
# ---------------------------------------------------------------------------
 
 
def _round(value: float) -> float:
    """Round to canonical precision for deterministic hashing."""
    return round(value, FLOAT_PRECISION)
 
 
def _canonical_json(obj: Any) -> str:
    """Produce canonical JSON: sorted keys, compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)
 
 
def _stable_hash_dict(d: Dict[str, Any]) -> str:
    """SHA-256 of canonical JSON."""
    raw = _canonical_json(d)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
 
 
# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
 
 
def _validate_positive_int(value: Any, field_name: str) -> int:
    """Validate a positive integer (not bool)."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be int, got bool")
    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")
    return value
 
 
def _validate_non_negative_int(value: Any, field_name: str) -> int:
    """Validate a non-negative integer (not bool)."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be int, got bool")
    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be int, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0, got {value}")
    return value
 
 
def _validate_non_negative_float(value: Any, field_name: str) -> float:
    """Validate a non-negative finite float."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{field_name} must be numeric, got {type(value).__name__}"
        )
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError(f"{field_name} must be finite, got {fv}")
    if fv < 0.0:
        raise ValueError(f"{field_name} must be >= 0, got {fv}")
    return fv
 
 
# ---------------------------------------------------------------------------
# Theory-coupled scoring functions
# ---------------------------------------------------------------------------
 
 
def _quantize_to_phi_shell(value: float) -> float:
    """Quantize a positive value to the nearest phi-shell.
 
    PHI_VIDEO_TIMING_LOCK: all timing values snap to the canonical
    phi-shell progression.  Linear interpolation is forbidden.
    """
    if value <= 0.0:
        return PHI_SHELLS[0]
    best = PHI_SHELLS[0]
    best_dist = abs(value - best)
    for shell in PHI_SHELLS[1:]:
        dist = abs(value - shell)
        if dist < best_dist:
            best = shell
            best_dist = dist
    return best
 
 
def _classify_physics_mode(segment_index: int) -> str:
    """E8_SCENE_TRIALITY_MESH: classify segment via mod-5 mapping."""
    return VALID_PHYSICS_MODES[segment_index % 5]
 
 
def _classify_scene(cue_index: int) -> str:
    """Classify scene cue via mod-5 scene mapping."""
    return VALID_SCENE_CLASSES[cue_index % 5]
 
 
def _compute_uff_restore_term(
    span_energy: float,
    phase_offset: float = UFF_PHASE_OFFSET,
) -> float:
    """OUROBOROS_FEEDBACK_LOOP: UFF restore operator.
 
    From theory: nabla^2 T + (phi + psi)^2 T = 0
    Implementation: restore = span_energy + ((PHI + phase_offset)^2) * 0.01
    Pure function of span_energy and phase_offset.
    """
    coupling = (PHI + phase_offset) ** 2
    return _round(span_energy + coupling * 0.01)
 
 
def _compute_phi_timing_lock_score(
    frames: Tuple[VideoFrame, ...],
) -> float:
    """PHI_VIDEO_TIMING_LOCK: measure how well frame energies align
    with phi-shell quantization.
 
    For each frame, compute distance from energy to nearest phi-shell.
    Perfect alignment scores 1.0.  Bounded [0, 1].
    """
    if len(frames) == 0:
        return 0.0
    total = 0.0
    for f in frames:
        shell = _quantize_to_phi_shell(abs(f.energy) if f.energy > 0 else 1.0)
        dist = abs(f.energy - shell) if f.energy > 0 else 0.0
        max_range = PHI_SHELLS[-1]
        closeness = 1.0 - min(dist / max_range, 1.0)
        total += closeness
    return max(0.0, min(1.0, total / len(frames)))
 
 
def _compute_triality_mesh_score(
    segments: Tuple[PhysicsSceneSegment, ...],
) -> float:
    """E8_SCENE_TRIALITY_MESH: uniformity of physics mode distribution.
 
    Segments classified by mod-5 should distribute uniformly across
    the five triality modes.  Perfect uniformity = 1.0.
    """
    if len(segments) == 0:
        return 0.0
    counts = [0] * 5
    for s in segments:
        idx = VALID_PHYSICS_MODES.index(s.physics_mode)
        counts[idx] += 1
    n = len(segments)
    expected = n / 5.0
    max_dev = expected * 4.0
    if max_dev < 1e-12:
        return 1.0
    actual_dev = sum(abs(c - expected) for c in counts)
    return max(0.0, min(1.0, 1.0 - actual_dev / max_dev))
 
 
def _compute_ouroboros_loop_score(
    frames: Tuple[VideoFrame, ...],
) -> float:
    """OUROBOROS_FRAME_LOOP: cyclic self-similarity of frame energies.
 
    Cosine similarity between first-half and second-half energy vectors.
    High similarity = strong ouroboros cycle.  Bounded [0, 1].
    """
    if len(frames) < 4:
        return 0.0
    energies = np.array([f.energy for f in frames], dtype=np.float64)
    mid = len(energies) // 2
    first_half = energies[:mid]
    second_half = energies[mid: mid + len(first_half)]
    if len(second_half) == 0:
        return 0.0
    dot = float(np.dot(first_half, second_half))
    norm_a = float(np.linalg.norm(first_half))
    norm_b = float(np.linalg.norm(second_half))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    cosine = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))
 
 
def _compute_demoscene_clock_score(
    frames: Tuple[VideoFrame, ...],
) -> float:
    """DEMOSCENE_RUNTIME_CLOCK: monotonicity and regularity of ticks.
 
    All ticks must be monotonically non-decreasing.  Score measures
    regularity of tick spacing.  Perfect uniformity = 1.0.
    """
    if len(frames) < 2:
        return 1.0 if len(frames) == 1 else 0.0
    ticks = np.array([f.tick for f in frames], dtype=np.float64)
    diffs = np.diff(ticks)
    # Check monotonicity
    if np.any(diffs < 0):
        return 0.0
    mean_diff = float(np.mean(diffs))
    if mean_diff < 1e-12:
        return 1.0
    std_diff = float(np.std(diffs))
    cv = std_diff / mean_diff
    return max(0.0, min(1.0, 1.0 - cv / 2.0))
 
 
# ---------------------------------------------------------------------------
# Cue extraction
# ---------------------------------------------------------------------------
 
 
def extract_audio_timeline_cues(
    beat_energies: Sequence[float],
    *,
    start_tick: int = 0,
) -> Tuple[AudioTimelineCue, ...]:
    """Extract deterministic audio timeline cues from beat energies.
 
    Each beat energy is quantized to the nearest phi-shell and assigned
    a monotonic tick.  Returns frozen, hashed cue objects.
    """
    _validate_non_negative_int(start_tick, "start_tick")
    cues = []
    for i, raw_energy in enumerate(beat_energies):
        energy = float(raw_energy)
        phi_shell = _quantize_to_phi_shell(abs(energy) if energy > 0 else 1.0)
        tick = start_tick + i
        payload = {
            "cue_index": i,
            "energy": _round(energy),
            "phi_shell": _round(phi_shell),
            "tick": tick,
            "version": PHYSICS_VIDEO_COMPOSITION_VERSION,
        }
        h = _stable_hash_dict(payload)
        cues.append(AudioTimelineCue(
            cue_index=i,
            tick=tick,
            energy=_round(energy),
            phi_shell=phi_shell,
            stable_hash=h,
        ))
    return tuple(cues)
 
 
def extract_visual_scene_cues(
    intensities: Sequence[float],
    *,
    start_tick: int = 0,
) -> Tuple[VisualSceneCue, ...]:
    """Extract deterministic visual scene cues from intensity values.
 
    Each intensity is classified into one of five scene classes via
    mod-5 mapping.  Returns frozen, hashed cue objects.
    """
    _validate_non_negative_int(start_tick, "start_tick")
    cues = []
    for i, raw_intensity in enumerate(intensities):
        intensity = float(raw_intensity)
        scene_class = _classify_scene(i)
        tick = start_tick + i
        payload = {
            "cue_index": i,
            "intensity": _round(intensity),
            "scene_class": scene_class,
            "tick": tick,
            "version": PHYSICS_VIDEO_COMPOSITION_VERSION,
        }
        h = _stable_hash_dict(payload)
        cues.append(VisualSceneCue(
            cue_index=i,
            tick=tick,
            intensity=_round(intensity),
            scene_class=scene_class,
            stable_hash=h,
        ))
    return tuple(cues)
 
 
# ---------------------------------------------------------------------------
# Scene segment builder
# ---------------------------------------------------------------------------
 
 
def build_physics_scene_segments(
    audio_cues: Tuple[AudioTimelineCue, ...],
    visual_cues: Tuple[VisualSceneCue, ...],
    *,
    ticks_per_segment: int = 4,
) -> Tuple[PhysicsSceneSegment, ...]:
    """Build physics-classified scene segments from audio + visual cues.
 
    Segments are built from contiguous tick windows.  Each segment gets
    a physics mode via E8 triality (mod-5) and a UFF restore term.
    """
    ticks_per_segment = _validate_positive_int(
        ticks_per_segment, "ticks_per_segment"
    )
    # Merge cue energies by tick
    tick_energies: Dict[int, float] = {}
    for ac in audio_cues:
        tick_energies[ac.tick] = tick_energies.get(ac.tick, 0.0) + ac.energy
    for vc in visual_cues:
        tick_energies[vc.tick] = tick_energies.get(vc.tick, 0.0) + vc.intensity
 
    if len(tick_energies) == 0:
        return ()
 
    sorted_ticks = sorted(tick_energies.keys())
    min_tick = sorted_ticks[0]
    max_tick = sorted_ticks[-1]
 
    segments = []
    seg_idx = 0
    t = min_tick
    while t <= max_tick:
        end_t = min(t + ticks_per_segment - 1, max_tick)
        seg_energy = 0.0
        count = 0
        for tick in range(t, end_t + 1):
            if tick in tick_energies:
                seg_energy += tick_energies[tick]
                count += 1
        avg_energy = _round(seg_energy / max(count, 1))
        physics_mode = _classify_physics_mode(seg_idx)
        phi_timing = _quantize_to_phi_shell(avg_energy if avg_energy > 0 else 1.0)
        uff_restore = _compute_uff_restore_term(avg_energy)
 
        trace_parts = [
            f"SEG={seg_idx}",
            f"MODE={physics_mode}",
            f"PHI={phi_timing:.6f}",
            f"UFF={uff_restore:.6f}",
        ]
        symbolic_trace = "|".join(trace_parts)
 
        payload = {
            "end_tick": end_t,
            "energy": avg_energy,
            "phi_timing": _round(phi_timing),
            "physics_mode": physics_mode,
            "segment_index": seg_idx,
            "start_tick": t,
            "symbolic_trace": symbolic_trace,
            "uff_restore_term": _round(uff_restore),
            "version": PHYSICS_VIDEO_COMPOSITION_VERSION,
        }
        h = _stable_hash_dict(payload)
 
        segments.append(PhysicsSceneSegment(
            segment_index=seg_idx,
            start_tick=t,
            end_tick=end_t,
            physics_mode=physics_mode,
            energy=avg_energy,
            phi_timing=phi_timing,
            uff_restore_term=uff_restore,
            symbolic_trace=symbolic_trace,
            stable_hash=h,
        ))
        seg_idx += 1
        t = end_t + 1
 
    return tuple(segments)
 
 
# ---------------------------------------------------------------------------
# Frame timeline builder
# ---------------------------------------------------------------------------
 
 
def build_video_frame_timeline(
    segments: Tuple[PhysicsSceneSegment, ...],
) -> Tuple[VideoFrame, ...]:
    """Build deterministic video frame timeline from scene segments.
 
    One frame per tick across all segments.  Each frame inherits the
    segment's physics mode and has phi-shell quantized energy.
    DEMOSCENE_RUNTIME_CLOCK: all ticks are integer, monotonic.
    """
    frames = []
    frame_idx = 0
    for seg in segments:
        for tick in range(seg.start_tick, seg.end_tick + 1):
            energy = _round(seg.energy)
            phi_shell = _quantize_to_phi_shell(
                abs(energy) if energy > 0 else 1.0
            )
            payload = {
                "energy": energy,
                "frame_index": frame_idx,
                "phi_shell": _round(phi_shell),
                "physics_mode": seg.physics_mode,
                "tick": tick,
                "version": PHYSICS_VIDEO_COMPOSITION_VERSION,
            }
            h = _stable_hash_dict(payload)
            frames.append(VideoFrame(
                frame_index=frame_idx,
                tick=tick,
                energy=energy,
                phi_shell=phi_shell,
                physics_mode=seg.physics_mode,
                stable_hash=h,
            ))
            frame_idx += 1
    return tuple(frames)
 
 
# ---------------------------------------------------------------------------
# Primary composition pipeline
# ---------------------------------------------------------------------------
 
 
def build_music_video_composition(
    audio_cues: Tuple[AudioTimelineCue, ...],
    visual_cues: Tuple[VisualSceneCue, ...],
    *,
    ticks_per_segment: int = 4,
) -> MusicVideoComposition:
    """Build a complete music video composition from cues.
 
    This is the primary pipeline entry point.  Composes:
      cues -> segments -> frames -> scored composition
 
    All four invariants are computed and embedded.
    """
    ticks_per_segment = _validate_positive_int(
        ticks_per_segment, "ticks_per_segment"
    )
    segments = build_physics_scene_segments(
        audio_cues, visual_cues, ticks_per_segment=ticks_per_segment,
    )
    frames = build_video_frame_timeline(segments)
 
    phi_timing = _compute_phi_timing_lock_score(frames)
    triality = _compute_triality_mesh_score(segments)
    ouroboros = _compute_ouroboros_loop_score(frames)
    demoscene = _compute_demoscene_clock_score(frames)
 
    symbolic_trace = (
        f"PHI_VIDEO_TIMING_LOCK={phi_timing:.6f}|"
        f"E8_SCENE_TRIALITY_MESH={triality:.6f}|"
        f"OUROBOROS_FRAME_LOOP={ouroboros:.6f}|"
        f"DEMOSCENE_RUNTIME_CLOCK={demoscene:.6f}"
    )
 
    payload = {
        "demoscene_clock_score": _round(demoscene),
        "frame_count": len(frames),
        "ouroboros_loop_score": _round(ouroboros),
        "phi_timing_lock_score": _round(phi_timing),
        "segment_count": len(segments),
        "segment_hashes": [s.stable_hash for s in segments],
        "symbolic_trace": symbolic_trace,
        "triality_mesh_score": _round(triality),
        "version": PHYSICS_VIDEO_COMPOSITION_VERSION,
    }
    h = _stable_hash_dict(payload)
 
    return MusicVideoComposition(
        segments=segments,
        frames=frames,
        segment_count=len(segments),
        frame_count=len(frames),
        phi_timing_lock_score=phi_timing,
        triality_mesh_score=triality,
        ouroboros_loop_score=ouroboros,
        demoscene_clock_score=demoscene,
        symbolic_trace=symbolic_trace,
        stable_hash=h,
    )
 
 
# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------
 
 
def export_physics_video_bundle(
    composition: MusicVideoComposition,
) -> Dict[str, Any]:
    """Export a composition as canonical JSON-safe dict."""
    return {
        "demoscene_clock_score": composition.demoscene_clock_score,
        "frame_count": composition.frame_count,
        "frames": [
            {
                "energy": f.energy,
                "frame_index": f.frame_index,
                "phi_shell": f.phi_shell,
                "physics_mode": f.physics_mode,
                "stable_hash": f.stable_hash,
                "tick": f.tick,
                "version": f.version,
            }
            for f in composition.frames
        ],
        "ouroboros_loop_score": composition.ouroboros_loop_score,
        "phi_timing_lock_score": composition.phi_timing_lock_score,
        "segment_count": composition.segment_count,
        "segments": [
            {
                "end_tick": s.end_tick,
                "energy": s.energy,
                "phi_timing": s.phi_timing,
                "physics_mode": s.physics_mode,
                "segment_index": s.segment_index,
                "start_tick": s.start_tick,
                "symbolic_trace": s.symbolic_trace,
                "stable_hash": s.stable_hash,
                "uff_restore_term": s.uff_restore_term,
                "version": s.version,
            }
            for s in composition.segments
        ],
        "stable_hash": composition.stable_hash,
        "symbolic_trace": composition.symbolic_trace,
        "triality_mesh_score": composition.triality_mesh_score,
        "version": composition.version,
    }
 
 
# ---------------------------------------------------------------------------
# Ledger functions
# ---------------------------------------------------------------------------
 
 
def build_physics_video_ledger(
    compositions: Tuple[MusicVideoComposition, ...],
) -> PhysicsVideoLedger:
    """Aggregate multiple compositions into a ledger."""
    payload = {
        "composition_count": len(compositions),
        "composition_hashes": [c.stable_hash for c in compositions],
    }
    h = _stable_hash_dict(payload)
    return PhysicsVideoLedger(
        compositions=compositions,
        composition_count=len(compositions),
        stable_hash=h,
    )
 
 
def export_physics_video_ledger(
    ledger: PhysicsVideoLedger,
) -> Dict[str, Any]:
    """Export the full ledger as canonical JSON-safe dict."""
    return {
        "composition_count": ledger.composition_count,
        "compositions": [
            export_physics_video_bundle(c) for c in ledger.compositions
        ],
        "stable_hash": ledger.stable_hash,
        "version": ledger.version,
    }
