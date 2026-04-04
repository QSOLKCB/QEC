"""v137.1.2 — Unified Physics Simulation Orchestrator.

Layer-4 deterministic orchestration that couples:
- v137.0.16 composition engine
- v137.0.17 simulation core
- v137.0.18 multimodal synchronization

Release focus: cross-module replay integrity + drift audit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from qec.analysis.demoscene_physics_simulation_core import build_runtime_simulation
from qec.analysis.multimodal_theory_synchronization import build_multimodal_sync_ledger
from qec.analysis.physics_music_video_composition_engine import (
    build_music_video_composition,
    extract_audio_timeline_cues,
    extract_visual_scene_cues,
)

UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION: str = "v137.1.2"
FLOAT_PRECISION: int = 12


def _round(value: float) -> float:
    return round(float(value), FLOAT_PRECISION)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash_dict(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _trace_tokens(symbolic_trace: str) -> Tuple[str, ...]:
    return tuple(sorted(set(tok.strip() for tok in str(symbolic_trace).split("|") if tok.strip())))


@dataclass(frozen=True)
class UnifiedPhysicsReplaySnapshot:
    cycle_index: int
    composition_stable_hash: str
    simulation_stable_hash: str
    sync_stable_hash: str
    orchestrator_stable_hash: str


@dataclass(frozen=True)
class UnifiedPhysicsOrchestratorArtifact:
    snapshots: Tuple[UnifiedPhysicsReplaySnapshot, ...]
    replay_cycles: int
    composition_stable_hash: str
    simulation_stable_hash: str
    sync_stable_hash: str
    symbolic_trace_valid: bool
    symbolic_trace: str
    orchestrator_drift_score: float
    stable_hash: str
    replay_identity: str
    version: str = UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION


def compute_orchestrator_drift_score(stable_hashes: Sequence[str]) -> float:
    """Deterministically score replay drift in [0, 1]."""
    hashes = tuple(str(h) for h in stable_hashes)
    count = len(hashes)
    if count <= 1:
        return 0.0
    unique = len(set(hashes))
    score = float(unique - 1) / float(count - 1)
    return _round(max(0.0, min(1.0, score)))


def validate_symbolic_memory_trace(
    symbolic_trace: str,
    sync_ledger: Mapping[str, Any],
) -> bool:
    """Validate symbolic trace consistency with synchronization timestamp map."""
    token_map = sync_ledger.get("symbolic_trace_timestamp_map", {})
    if not isinstance(token_map, Mapping):
        raise ValueError("sync_ledger.symbolic_trace_timestamp_map must be a mapping")
    expected = _trace_tokens(symbolic_trace)
    if len(expected) == 0:
        raise ValueError("symbolic_trace must contain at least one token")
    observed = tuple(sorted(str(key) for key in token_map.keys()))
    if expected != observed:
        raise ValueError(
            "symbolic trace divergence: expected token set does not match synchronized token map"
        )
    return True


def _build_orchestrator_cycle(
    beat_energies: Sequence[float],
    intensities: Sequence[float],
    *,
    start_tick: int,
    ticks_per_segment: int,
    symbolic_trace: str,
) -> Dict[str, Any]:
    audio = extract_audio_timeline_cues(beat_energies, start_tick=start_tick)
    visual = extract_visual_scene_cues(intensities, start_tick=start_tick)
    composition = build_music_video_composition(
        audio,
        visual,
        ticks_per_segment=ticks_per_segment,
    )
    simulation = build_runtime_simulation(composition.frames)
    sync_ledger = build_multimodal_sync_ledger(
        composition.frames,
        simulation.ticks,
        symbolic_trace,
    )
    return {
        "composition": composition,
        "simulation": simulation,
        "sync_ledger": sync_ledger,
    }


def audit_cross_module_replay_integrity(
    beat_energies: Sequence[float],
    intensities: Sequence[float],
    *,
    start_tick: int = 0,
    ticks_per_segment: int = 4,
    symbolic_trace: str = "PHI|E8|OURO|DEMO",
    replay_cycles: int = 8,
) -> UnifiedPhysicsOrchestratorArtifact:
    """Run deterministic replay cycles and audit cross-module stable-hash integrity."""
    if replay_cycles < 1:
        raise ValueError(f"replay_cycles must be >= 1, got {replay_cycles}")

    snapshots = []
    symbolic_trace_ok = False
    for cycle_index in range(replay_cycles):
        cycle = _build_orchestrator_cycle(
            beat_energies,
            intensities,
            start_tick=start_tick,
            ticks_per_segment=ticks_per_segment,
            symbolic_trace=symbolic_trace,
        )
        composition_hash = cycle["composition"].stable_hash
        simulation_hash = cycle["simulation"].stable_hash
        sync_ledger = cycle["sync_ledger"]
        sync_hash = str(sync_ledger["stable_hash"])
        if cycle_index == 0:
            symbolic_trace_ok = validate_symbolic_memory_trace(symbolic_trace, sync_ledger)

        orchestrator_payload = {
            "composition_stable_hash": composition_hash,
            "simulation_stable_hash": simulation_hash,
            "sync_stable_hash": sync_hash,
            "symbolic_trace": str(symbolic_trace),
            "version": UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        }
        orchestrator_hash = _stable_hash_dict(orchestrator_payload)
        snapshots.append(
            UnifiedPhysicsReplaySnapshot(
                cycle_index=cycle_index,
                composition_stable_hash=composition_hash,
                simulation_stable_hash=simulation_hash,
                sync_stable_hash=sync_hash,
                orchestrator_stable_hash=orchestrator_hash,
            )
        )

    snap_tuple = tuple(snapshots)
    first = snap_tuple[0]
    drift_score = compute_orchestrator_drift_score(
        tuple(snapshot.orchestrator_stable_hash for snapshot in snap_tuple)
    )

    final_payload = {
        "replay_cycles": replay_cycles,
        "composition_stable_hash": first.composition_stable_hash,
        "simulation_stable_hash": first.simulation_stable_hash,
        "sync_stable_hash": first.sync_stable_hash,
        "symbolic_trace": str(symbolic_trace),
        "symbolic_trace_valid": bool(symbolic_trace_ok),
        "orchestrator_drift_score": _round(drift_score),
        "orchestrator_hashes": [snapshot.orchestrator_stable_hash for snapshot in snap_tuple],
        "version": UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
    }
    stable_hash = _stable_hash_dict(final_payload)

    return UnifiedPhysicsOrchestratorArtifact(
        snapshots=snap_tuple,
        replay_cycles=replay_cycles,
        composition_stable_hash=first.composition_stable_hash,
        simulation_stable_hash=first.simulation_stable_hash,
        sync_stable_hash=first.sync_stable_hash,
        symbolic_trace_valid=bool(symbolic_trace_ok),
        symbolic_trace=str(symbolic_trace),
        orchestrator_drift_score=_round(drift_score),
        stable_hash=stable_hash,
        replay_identity=stable_hash,
    )
