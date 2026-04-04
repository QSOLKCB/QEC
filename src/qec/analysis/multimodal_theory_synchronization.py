"""v137.0.18 — Multimodal Theory Synchronization + CI Stabilization.

Deterministic Layer-4 synchronization substrate joining:
- v137.0.16 frame timelines
- v137.0.17 simulation ticks
- invariant clocks
- symbolic trace timestamps

Core law:
composition clock
+ simulation clock
+ invariant timeline
+ repository-wide CI integrity
= synchronized deterministic substrate
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

MULTIMODAL_THEORY_SYNCHRONIZATION_VERSION: str = "v137.0.18"
FLOAT_PRECISION: int = 12
PHI: float = 1.618033988749895


def _round(value: float) -> float:
    return round(float(value), FLOAT_PRECISION)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash_dict(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _extract_frame(frame: Any) -> Dict[str, Any]:
    if isinstance(frame, Mapping):
        return {
            "frame_index": int(frame.get("frame_index", 0)),
            "tick": int(frame.get("tick", 0)),
            "energy": float(frame.get("energy", 0.0)),
            "phi_shell": float(frame.get("phi_shell", 1.0)),
            "physics_mode": str(frame.get("physics_mode", "TRIALITY_SWEEP")),
        }
    return {
        "frame_index": int(getattr(frame, "frame_index", 0)),
        "tick": int(getattr(frame, "tick", 0)),
        "energy": float(getattr(frame, "energy", 0.0)),
        "phi_shell": float(getattr(frame, "phi_shell", 1.0)),
        "physics_mode": str(getattr(frame, "physics_mode", "TRIALITY_SWEEP")),
    }


def _extract_tick(tick: Any) -> Dict[str, Any]:
    if isinstance(tick, Mapping):
        return {
            "tick_index": int(tick.get("tick_index", 0)),
            "source_tick": int(tick.get("source_tick", 0)),
            "energy": float(tick.get("energy", 0.0)),
            "phi_shell": float(tick.get("phi_shell", 1.0)),
            "physics_mode": str(tick.get("physics_mode", "TRIALITY_SWEEP")),
        }
    return {
        "tick_index": int(getattr(tick, "tick_index", 0)),
        "source_tick": int(getattr(tick, "source_tick", 0)),
        "energy": float(getattr(tick, "energy", 0.0)),
        "phi_shell": float(getattr(tick, "phi_shell", 1.0)),
        "physics_mode": str(getattr(tick, "physics_mode", "TRIALITY_SWEEP")),
    }


def synchronize_composition_and_simulation_clocks(
    composition_frames: Sequence[Any],
    simulation_ticks: Sequence[Any],
) -> Tuple[Dict[str, Any], ...]:
    """Align v137.0.16 composition frames to v137.0.17 simulation ticks."""
    frames = [_extract_frame(frame) for frame in composition_frames]
    ticks = [_extract_tick(tick) for tick in simulation_ticks]
    ordered_frames = sorted(frames, key=lambda row: (row["tick"], row["frame_index"]))
    ordered_ticks = sorted(ticks, key=lambda row: (row["source_tick"], row["tick_index"]))
    pair_count = min(len(ordered_frames), len(ordered_ticks))

    aligned = []
    for idx in range(pair_count):
        frame = ordered_frames[idx]
        tick = ordered_ticks[idx]
        invariant_tick = int(max(frame["tick"], tick["source_tick"]))
        phi_alignment = 1.0 - min(abs(frame["phi_shell"] - tick["phi_shell"]) / (PHI + 1.0), 1.0)
        e8_consistency = 1.0 if frame["physics_mode"] == tick["physics_mode"] else 0.0
        ouroboros_clock = _round(float(idx + 1) / float(pair_count))
        aligned.append(
            {
                "pair_index": idx,
                "frame_tick": int(frame["tick"]),
                "simulation_tick": int(tick["source_tick"]),
                "invariant_tick": invariant_tick,
                "phi_shell_timing_alignment": _round(phi_alignment),
                "e8_transition_timing_consistency": _round(e8_consistency),
                "ouroboros_recurrence_clock": ouroboros_clock,
                "demoscene_runtime_synchronization": _round(1.0 if invariant_tick >= 0 else 0.0),
                "timestamp_token": f"F{frame['frame_index']}|S{tick['tick_index']}|T{invariant_tick}",
                "version": MULTIMODAL_THEORY_SYNCHRONIZATION_VERSION,
            }
        )
    return tuple(aligned)


def unify_invariant_timeline(
    synchronized_clock_rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], ...]:
    """Build a deterministic invariant timeline from synchronized rows."""
    ordered = sorted(
        synchronized_clock_rows,
        key=lambda row: (int(row["invariant_tick"]), int(row["pair_index"])),
    )
    timeline = []
    for idx, row in enumerate(ordered):
        timeline.append(
            {
                "timeline_index": idx,
                "invariant_tick": int(row["invariant_tick"]),
                "timestamp_token": str(row["timestamp_token"]),
                "phi_shell_timing_alignment": _round(float(row["phi_shell_timing_alignment"])),
                "e8_transition_timing_consistency": _round(float(row["e8_transition_timing_consistency"])),
                "ouroboros_recurrence_clock": _round(float(row["ouroboros_recurrence_clock"])),
                "demoscene_runtime_synchronization": _round(float(row["demoscene_runtime_synchronization"])),
                "version": MULTIMODAL_THEORY_SYNCHRONIZATION_VERSION,
            }
        )
    return tuple(timeline)


def build_symbolic_trace_timestamp_map(
    symbolic_trace: Iterable[str] | str,
    invariant_timeline: Sequence[Mapping[str, Any]],
) -> Dict[str, Tuple[int, ...]]:
    """Map symbolic trace tokens to invariant timeline ticks."""
    if isinstance(symbolic_trace, str):
        raw_tokens = tuple(tok.strip() for tok in symbolic_trace.split("|") if tok.strip())
    else:
        raw_tokens = tuple(str(tok).strip() for tok in symbolic_trace if str(tok).strip())

    ordered_tokens = tuple(sorted(raw_tokens))
    ticks = tuple(int(row["invariant_tick"]) for row in invariant_timeline)
    if len(ticks) == 0:
        return {token: tuple() for token in ordered_tokens}

    mapped: Dict[str, Tuple[int, ...]] = {}
    for idx, token in enumerate(ordered_tokens):
        offset = idx % len(ticks)
        rotated = ticks[offset:] + ticks[:offset]
        mapped[token] = tuple(rotated)
    return mapped


def stabilize_repository_test_ordering(test_nodeids: Sequence[str]) -> Tuple[str, ...]:
    """Deterministically normalize repository-wide pytest collection order."""
    indexed = [(str(nodeid), idx) for idx, nodeid in enumerate(test_nodeids)]

    def _key(item: Tuple[str, int]) -> Tuple[str, str, str, int]:
        nodeid, idx = item
        if "::" in nodeid:
            file_part, test_part = nodeid.split("::", 1)
        else:
            file_part, test_part = nodeid, ""
        return (file_part, test_part, nodeid, idx)

    ordered = sorted(indexed, key=_key)
    return tuple(nodeid for nodeid, _ in ordered)


def build_multimodal_sync_ledger(
    composition_frames: Sequence[Any],
    simulation_ticks: Sequence[Any],
    symbolic_trace: Iterable[str] | str,
) -> Dict[str, Any]:
    """Build synchronized deterministic ledger for multimodal theory state."""
    synchronized = synchronize_composition_and_simulation_clocks(composition_frames, simulation_ticks)
    timeline = unify_invariant_timeline(synchronized)
    timestamp_map = build_symbolic_trace_timestamp_map(symbolic_trace, timeline)

    phi_vals = np.asarray([row["phi_shell_timing_alignment"] for row in timeline], dtype=np.float64)
    e8_vals = np.asarray([row["e8_transition_timing_consistency"] for row in timeline], dtype=np.float64)
    ouro_vals = np.asarray([row["ouroboros_recurrence_clock"] for row in timeline], dtype=np.float64)
    demo_vals = np.asarray([row["demoscene_runtime_synchronization"] for row in timeline], dtype=np.float64)

    def _safe_mean(values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        return _round(float(np.mean(values, dtype=np.float64)))

    invariants = {
        "phi_shell_timing_alignment": _safe_mean(phi_vals),
        "e8_transition_timing_consistency": _safe_mean(e8_vals),
        "ouroboros_recurrence_clock": _safe_mean(ouro_vals),
        "demoscene_runtime_synchronization": _safe_mean(demo_vals),
    }
    payload = {
        "synchronized_rows": [dict(row) for row in synchronized],
        "invariant_timeline": [dict(row) for row in timeline],
        "symbolic_trace_timestamp_map": {k: list(v) for k, v in sorted(timestamp_map.items())},
        "invariants": invariants,
        "version": MULTIMODAL_THEORY_SYNCHRONIZATION_VERSION,
    }
    stable_hash = _stable_hash_dict(payload)
    return {
        **payload,
        "stable_hash": stable_hash,
        "replay_identity": stable_hash,
    }
