"""v137.0.17 — Demoscene Physics Simulation Core.

Deterministic Layer-4 runtime that replays composition frames from v137.0.16
into a replay-safe physics tick/state transition artifact.

Core law:
composition
-> simulation tick core
-> physics state evolution
-> transition propagation
-> replay-safe runtime artifact
-> canonical ledger
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

DEMOSCENE_PHYSICS_SIMULATION_VERSION: str = "v137.0.17"

FLOAT_PRECISION: int = 12
PHI: float = 1.618033988749895
PHI_SHELLS: Tuple[float, ...] = (1.0, 1.618, 2.618, 4.236, 6.854)

DEMOSCENE_RUNTIME_TICK_FIELD: str = "DEMOSCENE_RUNTIME_TICK_FIELD"
PHI_STATE_PROPAGATION_LOCK: str = "PHI_STATE_PROPAGATION_LOCK"
E8_TRANSITION_TRIALITY_CORE: str = "E8_TRANSITION_TRIALITY_CORE"
OUROBOROS_RUNTIME_FEEDBACK: str = "OUROBOROS_RUNTIME_FEEDBACK"

TRANSITION_MODES: Tuple[str, ...] = (
    "TRIALITY_SWEEP",
    "PHI_SHELL_EXPANSION",
    "OUROBOROS_LOOPBACK",
    "RESONANCE_COLLAPSE",
    "DEMOSCENE_TRANSITION",
)


def _round(value: float) -> float:
    return round(float(value), FLOAT_PRECISION)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash_dict(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _quantize_phi_shell(value: float) -> float:
    positive = max(float(value), 1e-12)
    best = PHI_SHELLS[0]
    best_dist = abs(positive - best)
    for shell in PHI_SHELLS[1:]:
        dist = abs(positive - shell)
        if dist < best_dist:
            best = shell
            best_dist = dist
    return float(best)


def _extract_frame_payload(frame: Any) -> Dict[str, Any]:
    if hasattr(frame, "tick"):
        return {
            "tick": int(getattr(frame, "tick")),
            "frame_index": int(getattr(frame, "frame_index", 0)),
            "energy": float(getattr(frame, "energy", 0.0)),
            "phi_shell": float(getattr(frame, "phi_shell", 1.0)),
            "physics_mode": str(getattr(frame, "physics_mode", TRANSITION_MODES[0])),
        }
    if isinstance(frame, Mapping):
        return {
            "tick": int(frame.get("tick", 0)),
            "frame_index": int(frame.get("frame_index", 0)),
            "energy": float(frame.get("energy", 0.0)),
            "phi_shell": float(frame.get("phi_shell", 1.0)),
            "physics_mode": str(frame.get("physics_mode", TRANSITION_MODES[0])),
        }
    raise TypeError("composition frame must be object-with-attrs or mapping")


@dataclass(frozen=True)
class PhysicsSimulationTick:
    tick_index: int
    source_tick: int
    frame_index: int
    energy: float
    phi_shell: float
    physics_mode: str
    transition_seed: float
    stable_hash: str
    replay_identity: str
    version: str = DEMOSCENE_PHYSICS_SIMULATION_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick_index": self.tick_index,
            "source_tick": self.source_tick,
            "frame_index": self.frame_index,
            "energy": self.energy,
            "phi_shell": self.phi_shell,
            "physics_mode": self.physics_mode,
            "transition_seed": self.transition_seed,
            "stable_hash": self.stable_hash,
            "replay_identity": self.replay_identity,
            "version": self.version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PhysicsSimulationState:
    tick_index: int
    source_tick: int
    particle_energy: float
    resonance_wave: float
    mesh_displacement: float
    transition_energy: float
    feedback_term: float
    stable_hash: str
    replay_identity: str
    version: str = DEMOSCENE_PHYSICS_SIMULATION_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick_index": self.tick_index,
            "source_tick": self.source_tick,
            "particle_energy": self.particle_energy,
            "resonance_wave": self.resonance_wave,
            "mesh_displacement": self.mesh_displacement,
            "transition_energy": self.transition_energy,
            "feedback_term": self.feedback_term,
            "stable_hash": self.stable_hash,
            "replay_identity": self.replay_identity,
            "version": self.version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PhysicsSimulationDecision:
    tick_index: int
    source_tick: int
    transition_mode: str
    transition_gain: float
    bounded: bool
    stable_hash: str
    replay_identity: str
    version: str = DEMOSCENE_PHYSICS_SIMULATION_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick_index": self.tick_index,
            "source_tick": self.source_tick,
            "transition_mode": self.transition_mode,
            "transition_gain": self.transition_gain,
            "bounded": self.bounded,
            "stable_hash": self.stable_hash,
            "replay_identity": self.replay_identity,
            "version": self.version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PhysicsSimulationLedger:
    ticks: Tuple[PhysicsSimulationTick, ...]
    states: Tuple[PhysicsSimulationState, ...]
    decisions: Tuple[PhysicsSimulationDecision, ...]
    runtime_hash: str
    invariant_scores: Dict[str, float]
    symbolic_trace: str
    stable_hash: str
    replay_identity: str
    version: str = DEMOSCENE_PHYSICS_SIMULATION_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticks": [t.to_dict() for t in self.ticks],
            "states": [s.to_dict() for s in self.states],
            "decisions": [d.to_dict() for d in self.decisions],
            "runtime_hash": self.runtime_hash,
            "invariant_scores": dict(self.invariant_scores),
            "symbolic_trace": self.symbolic_trace,
            "stable_hash": self.stable_hash,
            "replay_identity": self.replay_identity,
            "version": self.version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def build_simulation_ticks(composition_frames: Sequence[Any]) -> Tuple[PhysicsSimulationTick, ...]:
    frames = [_extract_frame_payload(frame) for frame in composition_frames]
    ordered = sorted(frames, key=lambda f: (f["tick"], f["frame_index"]))
    ticks = []
    for idx, frame in enumerate(ordered):
        mode = frame["physics_mode"]
        transition_seed = _round((frame["energy"] + PHI * (idx + 1)) / (frame["phi_shell"] + 1.0))
        payload = {
            "tick_index": idx,
            "source_tick": int(frame["tick"]),
            "frame_index": int(frame["frame_index"]),
            "energy": _round(frame["energy"]),
            "phi_shell": _round(_quantize_phi_shell(frame["phi_shell"])),
            "physics_mode": mode,
            "transition_seed": transition_seed,
            "version": DEMOSCENE_PHYSICS_SIMULATION_VERSION,
        }
        sh = _stable_hash_dict(payload)
        ticks.append(
            PhysicsSimulationTick(
                tick_index=idx,
                source_tick=payload["source_tick"],
                frame_index=payload["frame_index"],
                energy=payload["energy"],
                phi_shell=payload["phi_shell"],
                physics_mode=mode,
                transition_seed=transition_seed,
                stable_hash=sh,
                replay_identity=sh,
            )
        )
    return tuple(ticks)


def propagate_physics_state(ticks: Tuple[PhysicsSimulationTick, ...]) -> Tuple[PhysicsSimulationState, ...]:
    states = []
    prev_particle = np.float64(0.0)
    prev_wave = np.float64(0.0)
    for tick in ticks:
        e = np.float64(tick.energy)
        seed = np.float64(tick.transition_seed)
        phi_shell = np.float64(tick.phi_shell)
        particle = np.float64(0.6) * prev_particle + np.float64(0.4) * e
        resonance = np.float64(0.5) * prev_wave + np.float64(0.5) * (seed / (phi_shell + np.float64(1.0)))
        mesh = (particle - resonance) / (np.float64(PHI) + np.float64(1.0))
        transition = np.abs(mesh) + np.float64(0.25) * np.abs(e)
        feedback = (particle + resonance) / (np.float64(2.0) + phi_shell)
        payload = {
            "tick_index": tick.tick_index,
            "source_tick": tick.source_tick,
            "particle_energy": _round(float(particle)),
            "resonance_wave": _round(float(resonance)),
            "mesh_displacement": _round(float(mesh)),
            "transition_energy": _round(float(transition)),
            "feedback_term": _round(float(feedback)),
            "version": DEMOSCENE_PHYSICS_SIMULATION_VERSION,
        }
        sh = _stable_hash_dict(payload)
        states.append(
            PhysicsSimulationState(
                tick_index=tick.tick_index,
                source_tick=tick.source_tick,
                particle_energy=payload["particle_energy"],
                resonance_wave=payload["resonance_wave"],
                mesh_displacement=payload["mesh_displacement"],
                transition_energy=payload["transition_energy"],
                feedback_term=payload["feedback_term"],
                stable_hash=sh,
                replay_identity=sh,
            )
        )
        prev_particle = particle
        prev_wave = resonance
    return tuple(states)


def compute_transition_field(states: Tuple[PhysicsSimulationState, ...]) -> Tuple[PhysicsSimulationDecision, ...]:
    decisions = []
    for state in states:
        mode = TRANSITION_MODES[state.tick_index % len(TRANSITION_MODES)]
        gain = np.float64(state.transition_energy) + np.float64(0.5) * np.float64(abs(state.feedback_term))
        bounded = bool(gain <= np.float64(64.0))
        payload = {
            "tick_index": state.tick_index,
            "source_tick": state.source_tick,
            "transition_mode": mode,
            "transition_gain": _round(float(gain)),
            "bounded": bounded,
            "version": DEMOSCENE_PHYSICS_SIMULATION_VERSION,
        }
        sh = _stable_hash_dict(payload)
        decisions.append(
            PhysicsSimulationDecision(
                tick_index=state.tick_index,
                source_tick=state.source_tick,
                transition_mode=mode,
                transition_gain=payload["transition_gain"],
                bounded=bounded,
                stable_hash=sh,
                replay_identity=sh,
            )
        )
    return tuple(decisions)


def _score_tick_field(ticks: Tuple[PhysicsSimulationTick, ...]) -> float:
    if len(ticks) < 2:
        return 1.0 if len(ticks) == 1 else 0.0
    source = np.asarray([t.source_tick for t in ticks], dtype=np.float64)
    diffs = np.diff(source)
    if np.any(diffs < 0):
        return 0.0
    mean = float(np.mean(diffs))
    if mean <= 1e-12:
        return 1.0
    return max(0.0, min(1.0, 1.0 - float(np.std(diffs)) / (mean + 1e-12)))


def _score_phi_lock(ticks: Tuple[PhysicsSimulationTick, ...], states: Tuple[PhysicsSimulationState, ...]) -> float:
    if len(ticks) == 0:
        return 0.0
    closeness = []
    for tick, state in zip(ticks, states):
        anchor = _quantize_phi_shell(abs(state.particle_energy) if state.particle_energy > 0 else 1.0)
        dist = abs(anchor - tick.phi_shell)
        closeness.append(max(0.0, 1.0 - min(dist / PHI_SHELLS[-1], 1.0)))
    return _round(float(np.mean(np.asarray(closeness, dtype=np.float64))))


def _score_e8_triality(decisions: Tuple[PhysicsSimulationDecision, ...]) -> float:
    if len(decisions) == 0:
        return 0.0
    counts = np.zeros(len(TRANSITION_MODES), dtype=np.float64)
    for d in decisions:
        counts[TRANSITION_MODES.index(d.transition_mode)] += 1.0
    expected = float(len(decisions)) / float(len(TRANSITION_MODES))
    if expected <= 1e-12:
        return 1.0
    deviation = float(np.sum(np.abs(counts - expected)))
    max_dev = expected * float(len(TRANSITION_MODES) - 1)
    return _round(max(0.0, min(1.0, 1.0 - deviation / (max_dev + 1e-12))))


def _score_ouroboros_feedback(states: Tuple[PhysicsSimulationState, ...]) -> float:
    if len(states) < 4:
        return 0.0
    series = np.asarray([s.feedback_term for s in states], dtype=np.float64)
    half = len(series) // 2
    first = series[:half]
    second = series[half: half + len(first)]
    if len(second) == 0:
        return 0.0
    dot = float(np.dot(first, second))
    na = float(np.linalg.norm(first))
    nb = float(np.linalg.norm(second))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    cosine = dot / (na * nb)
    return _round(max(0.0, min(1.0, (cosine + 1.0) / 2.0)))


def build_simulation_ledger(
    ticks: Tuple[PhysicsSimulationTick, ...],
    states: Tuple[PhysicsSimulationState, ...],
    decisions: Tuple[PhysicsSimulationDecision, ...],
) -> PhysicsSimulationLedger:
    invariant_scores = {
        DEMOSCENE_RUNTIME_TICK_FIELD: _score_tick_field(ticks),
        PHI_STATE_PROPAGATION_LOCK: _score_phi_lock(ticks, states),
        E8_TRANSITION_TRIALITY_CORE: _score_e8_triality(decisions),
        OUROBOROS_RUNTIME_FEEDBACK: _score_ouroboros_feedback(states),
    }
    symbolic_trace = "|".join(
        f"{name}={invariant_scores[name]:.6f}"
        for name in (
            DEMOSCENE_RUNTIME_TICK_FIELD,
            PHI_STATE_PROPAGATION_LOCK,
            E8_TRANSITION_TRIALITY_CORE,
            OUROBOROS_RUNTIME_FEEDBACK,
        )
    )
    runtime_payload = {
        "tick_hashes": [t.stable_hash for t in ticks],
        "state_hashes": [s.stable_hash for s in states],
        "decision_hashes": [d.stable_hash for d in decisions],
        "version": DEMOSCENE_PHYSICS_SIMULATION_VERSION,
    }
    runtime_hash = _stable_hash_dict(runtime_payload)
    payload = {
        "runtime_hash": runtime_hash,
        "invariant_scores": {k: _round(v) for k, v in invariant_scores.items()},
        "symbolic_trace": symbolic_trace,
        "version": DEMOSCENE_PHYSICS_SIMULATION_VERSION,
    }
    sh = _stable_hash_dict(payload)
    return PhysicsSimulationLedger(
        ticks=ticks,
        states=states,
        decisions=decisions,
        runtime_hash=runtime_hash,
        invariant_scores={k: _round(v) for k, v in invariant_scores.items()},
        symbolic_trace=symbolic_trace,
        stable_hash=sh,
        replay_identity=sh,
    )


def build_runtime_simulation(composition_frames: Sequence[Any]) -> PhysicsSimulationLedger:
    ticks = build_simulation_ticks(composition_frames)
    states = propagate_physics_state(ticks)
    decisions = compute_transition_field(states)
    return build_simulation_ledger(ticks, states, decisions)


def export_simulation_bundle(ledger: PhysicsSimulationLedger) -> Dict[str, Any]:
    return ledger.to_dict()
