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
    stable_hash: str
    replay_identity: str
    symbolic_trace_valid: bool
    symbolic_trace: str
    orchestrator_drift_score: float
def _canonical_json_bytes(obj: Any) -> bytes:
    return _canonical_json(obj).encode("utf-8")


def _stable_hash_dict(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(dict(payload))).hexdigest()


def _extract_scalar_row(
    row: Any,
    *,
    int_fields: Sequence[str],
    float_fields: Sequence[str],
    str_fields: Sequence[str],
) -> Dict[str, Any]:
    """Shared deterministic extractor for mapping or object rows."""
    if isinstance(row, Mapping):
        getter = row.get
    else:
        getter = lambda key, default: getattr(row, key, default)

    out: Dict[str, Any] = {}
    for key in int_fields:
        out[key] = int(getter(key, 0))
    for key in float_fields:
        out[key] = _round(float(getter(key, 0.0)))
    for key in str_fields:
        out[key] = str(getter(key, ""))
    return out


def _validate_equal_lengths(frames: Sequence[Any], states: Sequence[Any], sync_rows: Sequence[Any]) -> None:
    f_len = len(frames)
    s_len = len(states)
    y_len = len(sync_rows)
    if f_len != s_len:
        raise ValueError(f"frames/states length mismatch: {f_len} != {s_len}")
    if f_len != y_len:
        raise ValueError(f"frames/sync_rows length mismatch: {f_len} != {y_len}")


@dataclass(frozen=True)
class UnifiedPhysicsSimulationLedger:
    frames: Tuple[Dict[str, Any], ...]
    states: Tuple[Dict[str, Any], ...]
    sync_rows: Tuple[Dict[str, Any], ...]
    invariant_scores: Dict[str, float]
    symbolic_trace: str
    stable_hash: str
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames": [dict(x) for x in self.frames],
            "states": [dict(x) for x in self.states],
            "sync_rows": [dict(x) for x in self.sync_rows],
            "invariant_scores": dict(self.invariant_scores),
            "symbolic_trace": self.symbolic_trace,
            "stable_hash": self.stable_hash,
            "replay_identity": self.replay_identity,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def _extract_fields(obj: Any, schema: Mapping[str, Tuple[Any, Any]]) -> Dict[str, Any]:
    if isinstance(obj, Mapping):
        return {name: caster(obj.get(name, default)) for name, (default, caster) in schema.items()}
    return {name: caster(getattr(obj, name, default)) for name, (default, caster) in schema.items()}


def _extract_frame(frame: Any) -> Dict[str, Any]:
    return _extract_fields(
        frame,
        {
            "frame_index": (0, int),
            "tick": (0, int),
            "energy": (0.0, float),
            "physics_mode": ("TRIALITY_SWEEP", str),
            "stable_hash": ("", str),
        },
    )


def _extract_state(state: Any) -> Dict[str, Any]:
    return _extract_fields(
        state,
        {
            "tick_index": (0, int),
            "source_tick": (0, int),
            "transition_energy": (0.0, float),
            "feedback_term": (0.0, float),
            "stable_hash": ("", str),
        },
    )


def _extract_decision(decision: Any) -> Dict[str, Any]:
    return _extract_fields(
        decision,
        {
            "tick_index": (0, int),
            "source_tick": (0, int),
            "transition_mode": ("TRIALITY_SWEEP", str),
            "transition_gain": (0.0, float),
        },
    )


def _extract_sync_row(row: Any) -> Dict[str, Any]:
    return _extract_fields(
        row,
        {
            "pair_index": (0, int),
            "invariant_tick": (0, int),
            "timestamp_token": ("", str),
            "phi_shell_timing_alignment": (0.0, float),
            "e8_transition_timing_consistency": (0.0, float),
            "ouroboros_recurrence_clock": (0.0, float),
            "demoscene_runtime_synchronization": (0.0, float),
        },
    )


@dataclass(frozen=True)
class OrchestratorState:
    state_index: int
    tick: int
    physics_mode: str
    energy: float
    frame_hash: str
    simulation_state_hash: str
    sync_token: str
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
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_index": self.state_index,
            "tick": self.tick,
            "physics_mode": self.physics_mode,
            "energy": self.energy,
            "frame_hash": self.frame_hash,
            "simulation_state_hash": self.simulation_state_hash,
            "sync_token": self.sync_token,
            "stable_hash": self.stable_hash,
            "replay_identity": self.replay_identity,
            "version": self.version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class OrchestratorDecision:
    state_index: int
    tick: int
    selected_mode: str
    deterministic_rank: int
    alignment_gain: float
    memory_gain: float
    stable_hash: str
    replay_identity: str
    version: str = UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_index": self.state_index,
            "tick": self.tick,
            "selected_mode": self.selected_mode,
            "deterministic_rank": self.deterministic_rank,
            "alignment_gain": self.alignment_gain,
            "memory_gain": self.memory_gain,
            "stable_hash": self.stable_hash,
            "replay_identity": self.replay_identity,
            "version": self.version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class OrchestratorTraceFrame:
    trace_index: int
    tick: int
    symbolic_token: str
    memory_scalar: float
    coupling_score: float
    stable_hash: str
    replay_identity: str
    version: str = UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_index": self.trace_index,
            "tick": self.tick,
            "symbolic_token": self.symbolic_token,
            "memory_scalar": self.memory_scalar,
            "coupling_score": self.coupling_score,
            "stable_hash": self.stable_hash,
            "replay_identity": self.replay_identity,
            "version": self.version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def build_unified_physics_simulation_orchestrator(
    frames: Sequence[Any],
    states: Sequence[Any],
    sync_rows: Sequence[Any],
) -> UnifiedPhysicsSimulationLedger:
    _validate_equal_lengths(frames, states, sync_rows)

    extracted_frames = tuple(
        _extract_scalar_row(
            row,
            int_fields=("frame_index", "tick"),
            float_fields=("energy", "phi_shell"),
            str_fields=("physics_mode",),
        )
        for row in frames
    )
    extracted_states = tuple(
        _extract_scalar_row(
            row,
            int_fields=("tick_index", "source_tick"),
            float_fields=("particle_energy", "transition_energy", "feedback_term"),
            str_fields=(),
        )
        for row in states
    )
    extracted_sync = tuple(
        _extract_scalar_row(
            row,
            int_fields=("pair_index", "invariant_tick"),
            float_fields=("phi_shell_timing_alignment", "e8_transition_timing_consistency"),
            str_fields=("timestamp_token",),
        )
        for row in sync_rows
    )

    ordered_frames = tuple(sorted(extracted_frames, key=lambda r: (r["tick"], r["frame_index"])))
    ordered_states = tuple(sorted(extracted_states, key=lambda r: (r["source_tick"], r["tick_index"])))
    ordered_sync = tuple(sorted(extracted_sync, key=lambda r: (r["invariant_tick"], r["pair_index"])))

    n = float(len(ordered_frames))
    if n == 0.0:
        drift_energy = 0.0
        drift_tick = 0.0
        drift_sync = 0.0
    else:
        drift_energy = _round(
            sum(abs(f["energy"] - s["particle_energy"]) for f, s in zip(ordered_frames, ordered_states)) / n
        )
        drift_tick = _round(
            sum(abs(float(f["tick"] - s["source_tick"])) for f, s in zip(ordered_frames, ordered_states)) / n
        )
        drift_sync = _round(
            sum(abs(float(f["tick"] - y["invariant_tick"])) for f, y in zip(ordered_frames, ordered_sync)) / n
        )

    rounded_invariants = {
        "energy_state_drift": _round(max(0.0, min(1.0, 1.0 - drift_energy))),
        "tick_state_drift": _round(max(0.0, min(1.0, 1.0 - drift_tick / 32.0))),
        "sync_tick_drift": _round(max(0.0, min(1.0, 1.0 - drift_sync / 32.0))),
    }
    symbolic_trace = "|".join(f"{k}={rounded_invariants[k]:.6f}" for k in sorted(rounded_invariants))

    payload = {
        "frames": [dict(x) for x in ordered_frames],
        "states": [dict(x) for x in ordered_states],
        "sync_rows": [dict(x) for x in ordered_sync],
        "invariant_scores": rounded_invariants,
        "symbolic_trace": symbolic_trace,
        "version": UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
    }
    stable_hash = _stable_hash_dict(payload)
    return UnifiedPhysicsSimulationLedger(
        frames=ordered_frames,
        states=ordered_states,
        sync_rows=ordered_sync,
        invariant_scores=rounded_invariants,
        symbolic_trace=symbolic_trace,
        stable_hash=stable_hash,
        replay_identity=stable_hash,
    )


def export_unified_physics_simulation_bundle(
    ledger: UnifiedPhysicsSimulationLedger,
) -> Dict[str, Any]:
    return ledger.to_dict()
