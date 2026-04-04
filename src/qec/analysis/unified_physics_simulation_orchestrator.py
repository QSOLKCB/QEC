"""v137.1.0 — Unified Physics-Simulation Orchestrator.

Layer-4 deterministic orchestration law:
composition
+ simulation
+ synchronization
+ symbolic memory trace
= unified deterministic orchestrator
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION: str = "v137.1.0"
FLOAT_PRECISION: int = 12

PHYSICS_ORCHESTRATION_LOCK: str = "PHYSICS_ORCHESTRATION_LOCK"
E8_RUNTIME_TRIALITY_ORCHESTRATION: str = "E8_RUNTIME_TRIALITY_ORCHESTRATION"
OUROBOROS_MEMORY_FEEDBACK: str = "OUROBOROS_MEMORY_FEEDBACK"
DEMOSCENE_MASTER_CLOCK: str = "DEMOSCENE_MASTER_CLOCK"


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
            "physics_mode": str(frame.get("physics_mode", "TRIALITY_SWEEP")),
            "stable_hash": str(frame.get("stable_hash", "")),
        }
    return {
        "frame_index": int(getattr(frame, "frame_index", 0)),
        "tick": int(getattr(frame, "tick", 0)),
        "energy": float(getattr(frame, "energy", 0.0)),
        "physics_mode": str(getattr(frame, "physics_mode", "TRIALITY_SWEEP")),
        "stable_hash": str(getattr(frame, "stable_hash", "")),
    }


def _extract_state(state: Any) -> Dict[str, Any]:
    if isinstance(state, Mapping):
        return {
            "tick_index": int(state.get("tick_index", 0)),
            "source_tick": int(state.get("source_tick", 0)),
            "transition_energy": float(state.get("transition_energy", 0.0)),
            "feedback_term": float(state.get("feedback_term", 0.0)),
            "stable_hash": str(state.get("stable_hash", "")),
        }
    return {
        "tick_index": int(getattr(state, "tick_index", 0)),
        "source_tick": int(getattr(state, "source_tick", 0)),
        "transition_energy": float(getattr(state, "transition_energy", 0.0)),
        "feedback_term": float(getattr(state, "feedback_term", 0.0)),
        "stable_hash": str(getattr(state, "stable_hash", "")),
    }


def _extract_decision(decision: Any) -> Dict[str, Any]:
    if isinstance(decision, Mapping):
        return {
            "tick_index": int(decision.get("tick_index", 0)),
            "source_tick": int(decision.get("source_tick", 0)),
            "transition_mode": str(decision.get("transition_mode", "TRIALITY_SWEEP")),
            "transition_gain": float(decision.get("transition_gain", 0.0)),
        }
    return {
        "tick_index": int(getattr(decision, "tick_index", 0)),
        "source_tick": int(getattr(decision, "source_tick", 0)),
        "transition_mode": str(getattr(decision, "transition_mode", "TRIALITY_SWEEP")),
        "transition_gain": float(getattr(decision, "transition_gain", 0.0)),
    }


def _extract_sync_row(row: Any) -> Dict[str, Any]:
    if isinstance(row, Mapping):
        return {
            "pair_index": int(row.get("pair_index", 0)),
            "invariant_tick": int(row.get("invariant_tick", 0)),
            "timestamp_token": str(row.get("timestamp_token", "")),
            "phi_shell_timing_alignment": float(row.get("phi_shell_timing_alignment", 0.0)),
            "e8_transition_timing_consistency": float(row.get("e8_transition_timing_consistency", 0.0)),
            "ouroboros_recurrence_clock": float(row.get("ouroboros_recurrence_clock", 0.0)),
            "demoscene_runtime_synchronization": float(row.get("demoscene_runtime_synchronization", 0.0)),
        }
    return {
        "pair_index": int(getattr(row, "pair_index", 0)),
        "invariant_tick": int(getattr(row, "invariant_tick", 0)),
        "timestamp_token": str(getattr(row, "timestamp_token", "")),
        "phi_shell_timing_alignment": float(getattr(row, "phi_shell_timing_alignment", 0.0)),
        "e8_transition_timing_consistency": float(getattr(row, "e8_transition_timing_consistency", 0.0)),
        "ouroboros_recurrence_clock": float(getattr(row, "ouroboros_recurrence_clock", 0.0)),
        "demoscene_runtime_synchronization": float(getattr(row, "demoscene_runtime_synchronization", 0.0)),
    }


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


@dataclass(frozen=True)
class OrchestratorLedger:
    states: Tuple[OrchestratorState, ...]
    decisions: Tuple[OrchestratorDecision, ...]
    trace_frames: Tuple[OrchestratorTraceFrame, ...]
    invariant_scores: Dict[str, float]
    symbolic_trace: str
    stable_hash: str
    replay_identity: str
    version: str = UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "states": [s.to_dict() for s in self.states],
            "decisions": [d.to_dict() for d in self.decisions],
            "trace_frames": [t.to_dict() for t in self.trace_frames],
            "invariant_scores": dict(self.invariant_scores),
            "symbolic_trace": self.symbolic_trace,
            "stable_hash": self.stable_hash,
            "replay_identity": self.replay_identity,
            "version": self.version,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def build_orchestrator_state_graph(
    composition_frames: Sequence[Any],
    simulation_states: Sequence[Any],
    synchronized_rows: Sequence[Any],
) -> Tuple[OrchestratorState, ...]:
    frames = sorted((_extract_frame(frame) for frame in composition_frames), key=lambda row: (row["tick"], row["frame_index"]))
    states = sorted((_extract_state(state) for state in simulation_states), key=lambda row: (row["source_tick"], row["tick_index"]))
    sync_rows = sorted((_extract_sync_row(row) for row in synchronized_rows), key=lambda row: (row["invariant_tick"], row["pair_index"]))

    pair_count = min(len(frames), len(states), len(sync_rows))
    graph = []
    for idx in range(pair_count):
        frame = frames[idx]
        state = states[idx]
        sync = sync_rows[idx]
        tick = int(max(frame["tick"], state["source_tick"], sync["invariant_tick"]))
        payload = {
            "state_index": idx,
            "tick": tick,
            "physics_mode": frame["physics_mode"],
            "energy": _round((float(frame["energy"]) + float(state["transition_energy"])) / 2.0),
            "frame_hash": frame["stable_hash"],
            "simulation_state_hash": state["stable_hash"],
            "sync_token": sync["timestamp_token"],
            "version": UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        }
        sh = _stable_hash_dict(payload)
        graph.append(
            OrchestratorState(
                state_index=idx,
                tick=tick,
                physics_mode=payload["physics_mode"],
                energy=payload["energy"],
                frame_hash=payload["frame_hash"],
                simulation_state_hash=payload["simulation_state_hash"],
                sync_token=payload["sync_token"],
                stable_hash=sh,
                replay_identity=sh,
            )
        )
    return tuple(graph)


def synchronize_subsystem_decisions(
    state_graph: Sequence[OrchestratorState],
    simulation_decisions: Sequence[Any],
) -> Tuple[OrchestratorDecision, ...]:
    raw_decisions = sorted((_extract_decision(decision) for decision in simulation_decisions), key=lambda row: (row["source_tick"], row["tick_index"]))
    pair_count = min(len(state_graph), len(raw_decisions))
    out = []
    for idx in range(pair_count):
        state = state_graph[idx]
        decision = raw_decisions[idx]
        alignment = _round((float(decision["transition_gain"]) + float(state.energy)) / (1.0 + float(idx + 1)))
        memory = _round((alignment + abs(float(state.energy))) / 2.0)
        payload = {
            "state_index": state.state_index,
            "tick": state.tick,
            "selected_mode": decision["transition_mode"],
            "deterministic_rank": idx,
            "alignment_gain": alignment,
            "memory_gain": memory,
            "version": UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        }
        sh = _stable_hash_dict(payload)
        out.append(
            OrchestratorDecision(
                state_index=state.state_index,
                tick=state.tick,
                selected_mode=decision["transition_mode"],
                deterministic_rank=idx,
                alignment_gain=alignment,
                memory_gain=memory,
                stable_hash=sh,
                replay_identity=sh,
            )
        )
    return tuple(out)


def build_symbolic_memory_trace(
    state_graph: Sequence[OrchestratorState],
    decisions: Sequence[OrchestratorDecision],
) -> Tuple[OrchestratorTraceFrame, ...]:
    pair_count = min(len(state_graph), len(decisions))
    traces = []
    prev_memory = 0.0
    for idx in range(pair_count):
        state = state_graph[idx]
        decision = decisions[idx]
        memory_scalar = _round(0.5 * prev_memory + 0.5 * decision.memory_gain)
        coupling = _round((memory_scalar + abs(state.energy)) / (1.0 + abs(float(idx))))
        payload = {
            "trace_index": idx,
            "tick": state.tick,
            "symbolic_token": f"{state.physics_mode}|{decision.selected_mode}|{state.sync_token}",
            "memory_scalar": memory_scalar,
            "coupling_score": coupling,
            "version": UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        }
        sh = _stable_hash_dict(payload)
        traces.append(
            OrchestratorTraceFrame(
                trace_index=idx,
                tick=state.tick,
                symbolic_token=payload["symbolic_token"],
                memory_scalar=memory_scalar,
                coupling_score=coupling,
                stable_hash=sh,
                replay_identity=sh,
            )
        )
        prev_memory = memory_scalar
    return tuple(traces)


def build_orchestrator_ledger(
    states: Tuple[OrchestratorState, ...],
    decisions: Tuple[OrchestratorDecision, ...],
    trace_frames: Tuple[OrchestratorTraceFrame, ...],
) -> OrchestratorLedger:
    phi_vals = np.asarray([float(d.alignment_gain) for d in decisions], dtype=np.float64)
    e8_vals = np.asarray([1.0 if s.physics_mode == d.selected_mode else 0.0 for s, d in zip(states, decisions)], dtype=np.float64)
    ouro_vals = np.asarray([float(t.memory_scalar) for t in trace_frames], dtype=np.float64)
    clock_vals = np.asarray([1.0 if t.tick >= 0 else 0.0 for t in trace_frames], dtype=np.float64)

    def _safe_mean(values: np.ndarray) -> float:
        if values.size == 0:
            return 0.0
        return _round(float(np.mean(values, dtype=np.float64)))

    invariant_scores = {
        PHYSICS_ORCHESTRATION_LOCK: _safe_mean(phi_vals),
        E8_RUNTIME_TRIALITY_ORCHESTRATION: _safe_mean(e8_vals),
        OUROBOROS_MEMORY_FEEDBACK: _safe_mean(ouro_vals),
        DEMOSCENE_MASTER_CLOCK: _safe_mean(clock_vals),
    }

    symbolic_trace = "|".join(
        f"{name}={invariant_scores[name]:.6f}"
        for name in (
            PHYSICS_ORCHESTRATION_LOCK,
            E8_RUNTIME_TRIALITY_ORCHESTRATION,
            OUROBOROS_MEMORY_FEEDBACK,
            DEMOSCENE_MASTER_CLOCK,
        )
    )
    payload = {
        "state_hashes": [s.stable_hash for s in states],
        "decision_hashes": [d.stable_hash for d in decisions],
        "trace_hashes": [t.stable_hash for t in trace_frames],
        "invariant_scores": {k: _round(v) for k, v in invariant_scores.items()},
        "symbolic_trace": symbolic_trace,
        "version": UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
    }
    sh = _stable_hash_dict(payload)
    return OrchestratorLedger(
        states=states,
        decisions=decisions,
        trace_frames=trace_frames,
        invariant_scores={k: _round(v) for k, v in invariant_scores.items()},
        symbolic_trace=symbolic_trace,
        stable_hash=sh,
        replay_identity=sh,
    )


def orchestrate_multimodal_runtime(
    composition_frames: Sequence[Any],
    simulation_ledger: Mapping[str, Any],
    synchronization_ledger: Mapping[str, Any],
) -> OrchestratorLedger:
    sim_states = simulation_ledger.get("states", ())
    sim_decisions = simulation_ledger.get("decisions", ())
    sync_rows = synchronization_ledger.get("synchronized_rows", ())

    state_graph = build_orchestrator_state_graph(composition_frames, sim_states, sync_rows)
    decisions = synchronize_subsystem_decisions(state_graph, sim_decisions)
    trace = build_symbolic_memory_trace(state_graph, decisions)
    return build_orchestrator_ledger(state_graph, decisions, trace)


def export_orchestrator_bundle(ledger: OrchestratorLedger) -> Dict[str, Any]:
    return ledger.to_dict()
