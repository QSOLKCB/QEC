"""v137.1.6 — Unified Physics Simulation Orchestrator.

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

UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION: str = "v137.1.6"
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


@dataclass(frozen=True)
class ReplaySnapshotDelta:
    cycle_index: int
    composition_ref: int
    simulation_ref: int
    sync_ref: int
    orchestrator_ref: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_index": int(self.cycle_index),
            "composition_ref": int(self.composition_ref),
            "simulation_ref": int(self.simulation_ref),
            "sync_ref": int(self.sync_ref),
            "orchestrator_ref": int(self.orchestrator_ref),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def compute_stable_hash(self) -> str:
        return _stable_hash_dict(self.to_dict())


@dataclass(frozen=True)
class ReplayHashChain:
    ordered_hashes: Tuple[str, ...]
    chain_hashes: Tuple[str, ...]
    chain_digest: str
    stable_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ordered_hashes": list(self.ordered_hashes),
            "chain_hashes": list(self.chain_hashes),
            "chain_digest": self.chain_digest,
            "stable_hash": self.stable_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class CompressedReplayBundle:
    version: str
    replay_cycles: int
    composition_hashes: Tuple[str, ...]
    simulation_hashes: Tuple[str, ...]
    sync_hashes: Tuple[str, ...]
    orchestrator_hashes: Tuple[str, ...]
    deltas: Tuple[ReplaySnapshotDelta, ...]
    symbolic_trace_valid: bool
    symbolic_trace: str
    orchestrator_drift_score: float
    composition_stable_hash: str
    simulation_stable_hash: str
    sync_stable_hash: str
    hash_chain: ReplayHashChain
    stable_hash: str
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "replay_cycles": int(self.replay_cycles),
            "composition_hashes": list(self.composition_hashes),
            "simulation_hashes": list(self.simulation_hashes),
            "sync_hashes": list(self.sync_hashes),
            "orchestrator_hashes": list(self.orchestrator_hashes),
            "deltas": [delta.to_dict() for delta in self.deltas],
            "symbolic_trace_valid": bool(self.symbolic_trace_valid),
            "symbolic_trace": self.symbolic_trace,
            "orchestrator_drift_score": _round(float(self.orchestrator_drift_score)),
            "composition_stable_hash": self.composition_stable_hash,
            "simulation_stable_hash": self.simulation_stable_hash,
            "sync_stable_hash": self.sync_stable_hash,
            "hash_chain": self.hash_chain.to_dict(),
            "stable_hash": self.stable_hash,
            "replay_identity": self.replay_identity,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def _dedupe_insertion_order(values: Sequence[str]) -> Tuple[str, ...]:
    seen: Dict[str, int] = {}
    ordered = []
    for value in values:
        key = str(value)
        if key not in seen:
            seen[key] = len(ordered)
            ordered.append(key)
    return tuple(ordered)


def build_replay_hash_chain(snapshots: Sequence[UnifiedPhysicsReplaySnapshot]) -> ReplayHashChain:
    ordered = tuple(str(snapshot.orchestrator_stable_hash) for snapshot in snapshots)
    rolling = "GENESIS"
    chain_hashes = []
    for idx, replay_hash in enumerate(ordered):
        rolling = hashlib.sha256(f"{idx}|{rolling}|{replay_hash}".encode("utf-8")).hexdigest()
        chain_hashes.append(rolling)
    chain_payload = {
        "ordered_hashes": list(ordered),
        "chain_hashes": list(chain_hashes),
        "chain_digest": rolling,
    }
    stable_hash = _stable_hash_dict(chain_payload)
    return ReplayHashChain(
        ordered_hashes=ordered,
        chain_hashes=tuple(chain_hashes),
        chain_digest=rolling,
        stable_hash=stable_hash,
    )


def compress_replay_snapshots(
    artifact: UnifiedPhysicsOrchestratorArtifact,
) -> CompressedReplayBundle:
    if int(artifact.replay_cycles) != len(artifact.snapshots):
        raise ValueError("artifact replay_cycles does not match snapshot length")
    if len(artifact.snapshots) == 0:
        raise ValueError("artifact snapshots must not be empty")

    composition_hashes = _dedupe_insertion_order(tuple(s.composition_stable_hash for s in artifact.snapshots))
    simulation_hashes = _dedupe_insertion_order(tuple(s.simulation_stable_hash for s in artifact.snapshots))
    sync_hashes = _dedupe_insertion_order(tuple(s.sync_stable_hash for s in artifact.snapshots))
    orchestrator_hashes = _dedupe_insertion_order(tuple(s.orchestrator_stable_hash for s in artifact.snapshots))

    composition_index = {value: idx for idx, value in enumerate(composition_hashes)}
    simulation_index = {value: idx for idx, value in enumerate(simulation_hashes)}
    sync_index = {value: idx for idx, value in enumerate(sync_hashes)}
    orchestrator_index = {value: idx for idx, value in enumerate(orchestrator_hashes)}

    deltas = tuple(
        ReplaySnapshotDelta(
            cycle_index=int(snapshot.cycle_index),
            composition_ref=int(composition_index[snapshot.composition_stable_hash]),
            simulation_ref=int(simulation_index[snapshot.simulation_stable_hash]),
            sync_ref=int(sync_index[snapshot.sync_stable_hash]),
            orchestrator_ref=int(orchestrator_index[snapshot.orchestrator_stable_hash]),
        )
        for snapshot in artifact.snapshots
    )

    hash_chain = build_replay_hash_chain(artifact.snapshots)
    bundle_payload = {
        "version": UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        "replay_cycles": int(artifact.replay_cycles),
        "composition_hashes": list(composition_hashes),
        "simulation_hashes": list(simulation_hashes),
        "sync_hashes": list(sync_hashes),
        "orchestrator_hashes": list(orchestrator_hashes),
        "deltas": [delta.to_dict() for delta in deltas],
        "symbolic_trace_valid": bool(artifact.symbolic_trace_valid),
        "symbolic_trace": str(artifact.symbolic_trace),
        "orchestrator_drift_score": _round(float(artifact.orchestrator_drift_score)),
        "composition_stable_hash": str(artifact.composition_stable_hash),
        "simulation_stable_hash": str(artifact.simulation_stable_hash),
        "sync_stable_hash": str(artifact.sync_stable_hash),
        "hash_chain": hash_chain.to_dict(),
    }
    stable_hash = _stable_hash_dict(bundle_payload)
    return CompressedReplayBundle(
        version=UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        replay_cycles=int(artifact.replay_cycles),
        composition_hashes=composition_hashes,
        simulation_hashes=simulation_hashes,
        sync_hashes=sync_hashes,
        orchestrator_hashes=orchestrator_hashes,
        deltas=deltas,
        symbolic_trace_valid=bool(artifact.symbolic_trace_valid),
        symbolic_trace=str(artifact.symbolic_trace),
        orchestrator_drift_score=_round(float(artifact.orchestrator_drift_score)),
        composition_stable_hash=str(artifact.composition_stable_hash),
        simulation_stable_hash=str(artifact.simulation_stable_hash),
        sync_stable_hash=str(artifact.sync_stable_hash),
        hash_chain=hash_chain,
        stable_hash=stable_hash,
        replay_identity=stable_hash,
    )


def decompress_replay_snapshots(
    bundle: CompressedReplayBundle,
) -> Tuple[UnifiedPhysicsReplaySnapshot, ...]:
    if str(bundle.version) != UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION:
        raise ValueError(
            "unsupported compressed bundle version: "
            f"expected {UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION}, "
            f"got {bundle.version}"
        )
    if int(bundle.replay_cycles) != len(bundle.deltas):
        raise ValueError("malformed compressed bundle: replay_cycles must equal delta count")
    if len(bundle.composition_hashes) == 0:
        raise ValueError("malformed compressed bundle: composition hash table is empty")
    if len(bundle.simulation_hashes) == 0:
        raise ValueError("malformed compressed bundle: simulation hash table is empty")
    if len(bundle.sync_hashes) == 0:
        raise ValueError("malformed compressed bundle: sync hash table is empty")
    if len(bundle.orchestrator_hashes) == 0:
        raise ValueError("malformed compressed bundle: orchestrator hash table is empty")

    snapshots = []
    expected_cycle = 0
    for delta in bundle.deltas:
        if int(delta.cycle_index) != expected_cycle:
            raise ValueError("malformed compressed bundle: non-sequential cycle index")
        if not (0 <= delta.composition_ref < len(bundle.composition_hashes)):
            raise ValueError("malformed compressed bundle: composition_ref out of range")
        if not (0 <= delta.simulation_ref < len(bundle.simulation_hashes)):
            raise ValueError("malformed compressed bundle: simulation_ref out of range")
        if not (0 <= delta.sync_ref < len(bundle.sync_hashes)):
            raise ValueError("malformed compressed bundle: sync_ref out of range")
        if not (0 <= delta.orchestrator_ref < len(bundle.orchestrator_hashes)):
            raise ValueError("malformed compressed bundle: orchestrator_ref out of range")
        snapshots.append(
            UnifiedPhysicsReplaySnapshot(
                cycle_index=expected_cycle,
                composition_stable_hash=str(bundle.composition_hashes[delta.composition_ref]),
                simulation_stable_hash=str(bundle.simulation_hashes[delta.simulation_ref]),
                sync_stable_hash=str(bundle.sync_hashes[delta.sync_ref]),
                orchestrator_stable_hash=str(bundle.orchestrator_hashes[delta.orchestrator_ref]),
            )
        )
        expected_cycle += 1
    return tuple(snapshots)


def export_compressed_replay_bundle(bundle: CompressedReplayBundle) -> Dict[str, Any]:
    return bundle.to_dict()


def verify_replay_bundle_roundtrip(
    artifact: UnifiedPhysicsOrchestratorArtifact,
) -> bool:
    compressed = compress_replay_snapshots(artifact)
    decompressed = decompress_replay_snapshots(compressed)
    if decompressed != artifact.snapshots:
        return False
    chain = build_replay_hash_chain(decompressed)
    if chain.chain_digest != compressed.hash_chain.chain_digest:
        return False
    exported = export_compressed_replay_bundle(compressed)
    return (
        str(exported.get("stable_hash", "")) == compressed.stable_hash
        and str(exported.get("replay_identity", "")) == compressed.replay_identity
    )


def _validate_symbolic_trace_timestamp_map(symbolic_trace_timestamp_map: Mapping[str, Sequence[int]]) -> Dict[str, Tuple[int, ...]]:
    if not isinstance(symbolic_trace_timestamp_map, Mapping):
        raise ValueError("invalid symbolic trace map: expected mapping")
    normalized: Dict[str, Tuple[int, ...]] = {}
    for original_key in sorted(symbolic_trace_timestamp_map.keys(), key=str):
        raw = symbolic_trace_timestamp_map[original_key]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("invalid symbolic trace map: timestamps must be integer sequences")
        normalized[str(original_key)] = tuple(sorted(int(x) for x in raw))
    return normalized


def _symbolic_trace_anchor(symbolic_trace_timestamp_map: Mapping[str, Sequence[int]]) -> str:
    normalized = _validate_symbolic_trace_timestamp_map(symbolic_trace_timestamp_map)
    payload = {key: list(value) for key, value in normalized.items()}
    return _stable_hash_dict(payload)


@dataclass(frozen=True)
class DriftProvenanceRecord:
    cycle_index: int
    source_module: str
    prior_hash: str
    divergent_hash: str
    chain_digest_anchor: str
    delta_table_reference: str
    symbolic_trace_anchor: str
    bounded_drift_score: float
    stable_hash: str
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_index": int(self.cycle_index),
            "source_module": str(self.source_module),
            "prior_hash": str(self.prior_hash),
            "divergent_hash": str(self.divergent_hash),
            "chain_digest_anchor": str(self.chain_digest_anchor),
            "delta_table_reference": str(self.delta_table_reference),
            "symbolic_trace_anchor": str(self.symbolic_trace_anchor),
            "bounded_drift_score": _round(float(self.bounded_drift_score)),
            "stable_hash": str(self.stable_hash),
            "replay_identity": str(self.replay_identity),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class DriftCycleReport:
    cycle_index: int
    composition_match: bool
    simulation_match: bool
    synchronization_match: bool
    orchestrator_match: bool
    symbolic_trace_match: bool
    drift_source: str
    record: DriftProvenanceRecord
    stable_hash: str
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_index": int(self.cycle_index),
            "composition_match": bool(self.composition_match),
            "simulation_match": bool(self.simulation_match),
            "synchronization_match": bool(self.synchronization_match),
            "orchestrator_match": bool(self.orchestrator_match),
            "symbolic_trace_match": bool(self.symbolic_trace_match),
            "drift_source": str(self.drift_source),
            "record": self.record.to_dict(),
            "stable_hash": str(self.stable_hash),
            "replay_identity": str(self.replay_identity),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class DriftProvenanceLedger:
    version: str
    replay_cycles: int
    first_divergence_point: int
    cycle_reports: Tuple[DriftCycleReport, ...]
    chain_digest_anchor: str
    delta_table_reference: str
    symbolic_trace_anchor: str
    stable_hash: str
    replay_identity: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": str(self.version),
            "replay_cycles": int(self.replay_cycles),
            "first_divergence_point": int(self.first_divergence_point),
            "cycle_reports": [x.to_dict() for x in self.cycle_reports],
            "chain_digest_anchor": str(self.chain_digest_anchor),
            "delta_table_reference": str(self.delta_table_reference),
            "symbolic_trace_anchor": str(self.symbolic_trace_anchor),
            "stable_hash": str(self.stable_hash),
            "replay_identity": str(self.replay_identity),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def attribute_drift_source(
    *,
    composition_match: bool,
    simulation_match: bool,
    synchronization_match: bool,
    orchestrator_match: bool,
    symbolic_trace_match: bool,
) -> str:
    mismatches = tuple(
        key
        for key, ok in (
            ("composition", composition_match),
            ("simulation", simulation_match),
            ("synchronization", synchronization_match),
            ("orchestrator", orchestrator_match),
            ("symbolic_trace", symbolic_trace_match),
        )
        if not ok
    )
    if len(mismatches) == 0:
        return "cross-module"
    if len(mismatches) == 1:
        return mismatches[0]
    return "cross-module"


def compare_replay_cycles(
    reference_snapshot: UnifiedPhysicsReplaySnapshot,
    candidate_snapshot: UnifiedPhysicsReplaySnapshot,
    *,
    cycle_index: int,
    chain_digest_anchor: str,
    delta_table_reference: str,
    symbolic_trace_anchor: str,
    symbolic_trace_match: bool,
) -> DriftCycleReport:
    composition_match = (
        str(reference_snapshot.composition_stable_hash) == str(candidate_snapshot.composition_stable_hash)
    )
    simulation_match = (
        str(reference_snapshot.simulation_stable_hash) == str(candidate_snapshot.simulation_stable_hash)
    )
    synchronization_match = str(reference_snapshot.sync_stable_hash) == str(candidate_snapshot.sync_stable_hash)
    orchestrator_match = (
        str(reference_snapshot.orchestrator_stable_hash) == str(candidate_snapshot.orchestrator_stable_hash)
    )
    drift_source = attribute_drift_source(
        composition_match=composition_match,
        simulation_match=simulation_match,
        synchronization_match=synchronization_match,
        orchestrator_match=orchestrator_match,
        symbolic_trace_match=symbolic_trace_match,
    )
    mismatch_count = (
        int(not composition_match)
        + int(not simulation_match)
        + int(not synchronization_match)
        + int(not orchestrator_match)
        + int(not symbolic_trace_match)
    )
    bounded_drift_score = _round(float(mismatch_count) / 5.0)
    prior_hash = str(reference_snapshot.orchestrator_stable_hash)
    divergent_hash = str(candidate_snapshot.orchestrator_stable_hash)
    record_payload = {
        "cycle_index": int(cycle_index),
        "source_module": drift_source,
        "prior_hash": prior_hash,
        "divergent_hash": divergent_hash,
        "chain_digest_anchor": str(chain_digest_anchor),
        "delta_table_reference": str(delta_table_reference),
        "symbolic_trace_anchor": str(symbolic_trace_anchor),
        "bounded_drift_score": bounded_drift_score,
    }
    record_hash = _stable_hash_dict(record_payload)
    record = DriftProvenanceRecord(
        cycle_index=int(cycle_index),
        source_module=drift_source,
        prior_hash=prior_hash,
        divergent_hash=divergent_hash,
        chain_digest_anchor=str(chain_digest_anchor),
        delta_table_reference=str(delta_table_reference),
        symbolic_trace_anchor=str(symbolic_trace_anchor),
        bounded_drift_score=bounded_drift_score,
        stable_hash=record_hash,
        replay_identity=record_hash,
    )
    report_payload = {
        "cycle_index": int(cycle_index),
        "composition_match": composition_match,
        "simulation_match": simulation_match,
        "synchronization_match": synchronization_match,
        "orchestrator_match": orchestrator_match,
        "symbolic_trace_match": bool(symbolic_trace_match),
        "drift_source": drift_source,
        "record": record.to_dict(),
    }
    report_hash = _stable_hash_dict(report_payload)
    return DriftCycleReport(
        cycle_index=int(cycle_index),
        composition_match=composition_match,
        simulation_match=simulation_match,
        synchronization_match=synchronization_match,
        orchestrator_match=orchestrator_match,
        symbolic_trace_match=bool(symbolic_trace_match),
        drift_source=drift_source,
        record=record,
        stable_hash=report_hash,
        replay_identity=report_hash,
    )


def build_drift_provenance_report(
    reference_bundle: CompressedReplayBundle,
    candidate_bundle: CompressedReplayBundle,
    *,
    reference_symbolic_trace_timestamp_map: Mapping[str, Sequence[int]],
    candidate_symbolic_trace_timestamp_map: Mapping[str, Sequence[int]],
) -> DriftProvenanceLedger:
    if int(reference_bundle.replay_cycles) != int(candidate_bundle.replay_cycles):
        raise ValueError("mismatched cycle counts")
    if str(reference_bundle.hash_chain.chain_digest) == "":
        raise ValueError("invalid hash-chain reference")
    if len(candidate_bundle.deltas) == 0:
        raise ValueError("missing delta anchor")
    reference_snapshots = decompress_replay_snapshots(reference_bundle)
    candidate_snapshots = decompress_replay_snapshots(candidate_bundle)
    if len(reference_snapshots) != len(candidate_snapshots):
        raise ValueError("mismatched cycle counts")

    reference_trace_anchor = _symbolic_trace_anchor(reference_symbolic_trace_timestamp_map)
    candidate_trace_anchor = _symbolic_trace_anchor(candidate_symbolic_trace_timestamp_map)
    symbolic_trace_match = reference_trace_anchor == candidate_trace_anchor

    delta_table_reference = _stable_hash_dict(
        {"deltas": [delta.to_dict() for delta in candidate_bundle.deltas], "replay_cycles": int(candidate_bundle.replay_cycles)}
    )
    chain_digest_anchor = str(reference_bundle.hash_chain.chain_digest)

    reports = []
    first_divergence_point = -1
    for cycle_index, (reference_snapshot, candidate_snapshot) in enumerate(zip(reference_snapshots, candidate_snapshots)):
        report = compare_replay_cycles(
            reference_snapshot,
            candidate_snapshot,
            cycle_index=cycle_index,
            chain_digest_anchor=chain_digest_anchor,
            delta_table_reference=delta_table_reference,
            symbolic_trace_anchor=candidate_trace_anchor,
            symbolic_trace_match=symbolic_trace_match,
        )
        if first_divergence_point < 0 and report.record.bounded_drift_score > 0.0:
            first_divergence_point = cycle_index
        reports.append(report)

    if first_divergence_point < 0:
        first_divergence_point = 0

    ledger_payload = {
        "version": UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        "replay_cycles": int(candidate_bundle.replay_cycles),
        "first_divergence_point": int(first_divergence_point),
        "cycle_reports": [report.to_dict() for report in reports],
        "chain_digest_anchor": chain_digest_anchor,
        "delta_table_reference": delta_table_reference,
        "symbolic_trace_anchor": candidate_trace_anchor,
    }
    ledger_hash = _stable_hash_dict(ledger_payload)
    return DriftProvenanceLedger(
        version=UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        replay_cycles=int(candidate_bundle.replay_cycles),
        first_divergence_point=int(first_divergence_point),
        cycle_reports=tuple(reports),
        chain_digest_anchor=chain_digest_anchor,
        delta_table_reference=delta_table_reference,
        symbolic_trace_anchor=candidate_trace_anchor,
        stable_hash=ledger_hash,
        replay_identity=ledger_hash,
    )


def export_drift_provenance_bundle(ledger: DriftProvenanceLedger) -> Dict[str, Any]:
    payload = ledger.to_dict()
    required_keys = {
        "version",
        "replay_cycles",
        "first_divergence_point",
        "cycle_reports",
        "chain_digest_anchor",
        "delta_table_reference",
        "symbolic_trace_anchor",
        "stable_hash",
        "replay_identity",
    }
    if tuple(sorted(payload.keys())) != tuple(sorted(required_keys)):
        raise ValueError("malformed provenance bundle")
    return payload


def verify_drift_provenance_roundtrip(
    ledger: DriftProvenanceLedger,
    *,
    reference_bundle: CompressedReplayBundle,
    candidate_bundle: CompressedReplayBundle,
    candidate_symbolic_trace_timestamp_map: Mapping[str, Sequence[int]],
) -> bool:
    exported = export_drift_provenance_bundle(ledger)
    recomputed_anchor = _symbolic_trace_anchor(candidate_symbolic_trace_timestamp_map)
    if str(exported.get("symbolic_trace_anchor", "")) != recomputed_anchor:
        raise ValueError("invalid symbolic trace map")
    if str(ledger.chain_digest_anchor) != str(reference_bundle.hash_chain.chain_digest):
        raise ValueError("invalid hash-chain reference")
    if str(ledger.delta_table_reference) == "":
        raise ValueError("missing delta anchor")
    if int(reference_bundle.replay_cycles) != int(candidate_bundle.replay_cycles):
        raise ValueError("mismatched cycle counts")
    # Recompute stable hash from canonical content payload (excluding stable_hash and
    # replay_identity) to verify ledger integrity non-self-referentially.
    content_payload = {k: v for k, v in exported.items() if k not in ("stable_hash", "replay_identity")}
    recomputed_hash = _stable_hash_dict(content_payload)
    if recomputed_hash != str(ledger.stable_hash):
        raise ValueError("malformed provenance bundle")
    return str(exported.get("replay_identity", "")) == str(ledger.replay_identity)


_REPAIR_ACTIONS_BY_SOURCE: Mapping[str, Tuple[str, ...]] = {
    "composition": (
        "resync composition clock",
        "rebuild composition snapshot delta",
        "verify composition hash chain",
    ),
    "simulation": (
        "rebuild simulation tick graph",
        "re-run state propagation audit",
        "verify simulation hash integrity",
    ),
    "synchronization": (
        "rebuild sync timestamp map",
        "verify delta anchor consistency",
        "revalidate symbolic trace timestamps",
    ),
    "orchestrator": (
        "rebuild cycle ledger",
        "recompute replay identity",
        "re-run provenance verification",
    ),
    "symbolic_trace": (
        "rebuild symbolic token map",
        "verify timestamp provenance",
        "compare trace divergence anchors",
    ),
    "cross-module": (
        "full replay audit recommendation",
        "drift provenance rerun",
        "cycle anchor verification",
    ),
}

_VALID_REPAIR_SOURCE_MODULES: Tuple[str, ...] = tuple(sorted(_REPAIR_ACTIONS_BY_SOURCE.keys()))


@dataclass(frozen=True)
class RepairSuggestion:
    source_module: str
    repair_action: str
    advisory_priority: int
    deterministic_rank_score: float
    provenance_cycle_reference: int
    first_divergence_anchor: int
    advisory_only: bool
    stable_hash: str
    replay_identity: str

    def __post_init__(self) -> None:
        if self.advisory_only is not True:
            raise ValueError("repair suggestions must remain advisory-only")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_module": str(self.source_module),
            "repair_action": str(self.repair_action),
            "advisory_priority": int(self.advisory_priority),
            "deterministic_rank_score": _round(float(self.deterministic_rank_score)),
            "provenance_cycle_reference": int(self.provenance_cycle_reference),
            "first_divergence_anchor": int(self.first_divergence_anchor),
            "advisory_only": bool(self.advisory_only),
            "stable_hash": str(self.stable_hash),
            "replay_identity": str(self.replay_identity),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class RepairSuggestionBundle:
    version: str
    suggestions: Tuple[RepairSuggestion, ...]
    advisory_only: bool
    stable_hash: str
    replay_identity: str

    def __post_init__(self) -> None:
        if self.advisory_only is not True:
            raise ValueError("repair suggestions must remain advisory-only")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": str(self.version),
            "suggestions": [item.to_dict() for item in self.suggestions],
            "advisory_only": bool(self.advisory_only),
            "stable_hash": str(self.stable_hash),
            "replay_identity": str(self.replay_identity),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class RepairSuggestionLedger:
    version: str
    provenance_stable_hash: str
    first_divergence_cycle: int
    suggestion_bundle: RepairSuggestionBundle
    advisory_only: bool
    stable_hash: str
    replay_identity: str

    def __post_init__(self) -> None:
        if self.advisory_only is not True:
            raise ValueError("repair suggestions must remain advisory-only")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": str(self.version),
            "provenance_stable_hash": str(self.provenance_stable_hash),
            "first_divergence_cycle": int(self.first_divergence_cycle),
            "suggestion_bundle": self.suggestion_bundle.to_dict(),
            "advisory_only": bool(self.advisory_only),
            "stable_hash": str(self.stable_hash),
            "replay_identity": str(self.replay_identity),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def map_drift_to_repair_actions(source_module: str) -> Tuple[str, ...]:
    source_key = str(source_module)
    if source_key not in _REPAIR_ACTIONS_BY_SOURCE:
        raise ValueError(f"invalid source module: {source_key}")
    return _REPAIR_ACTIONS_BY_SOURCE[source_key]


def rank_repair_candidates(candidates: Sequence[RepairSuggestion]) -> Tuple[RepairSuggestion, ...]:
    return tuple(
        sorted(
            tuple(candidates),
            key=lambda item: (
                -_round(float(item.deterministic_rank_score)),
                int(item.advisory_priority),
                int(item.provenance_cycle_reference),
                str(item.source_module),
                str(item.repair_action),
                str(item.stable_hash),
            ),
        )
    )


def _validate_repair_inputs(provenance_ledger: DriftProvenanceLedger) -> None:
    if not isinstance(provenance_ledger, DriftProvenanceLedger):
        raise ValueError("malformed provenance ledger")
    if int(provenance_ledger.replay_cycles) < 1:
        raise ValueError("malformed provenance ledger")
    if int(provenance_ledger.first_divergence_point) < 0 or int(provenance_ledger.first_divergence_point) >= int(
        provenance_ledger.replay_cycles
    ):
        raise ValueError("invalid cycle anchor")
    if str(provenance_ledger.chain_digest_anchor) == "":
        raise ValueError("invalid cycle anchor")
    if len(provenance_ledger.cycle_reports) != int(provenance_ledger.replay_cycles):
        raise ValueError("malformed provenance ledger")
    for report in provenance_ledger.cycle_reports:
        if str(report.drift_source) not in _VALID_REPAIR_SOURCE_MODULES:
            raise ValueError(f"invalid source module: {report.drift_source}")
        drift_score = float(report.record.bounded_drift_score)
        if drift_score < 0.0 or drift_score > 1.0:
            raise ValueError(f"invalid drift score: {drift_score}")


def generate_repair_suggestions(provenance_ledger: DriftProvenanceLedger) -> RepairSuggestionLedger:
    _validate_repair_inputs(provenance_ledger)
    suggestions = []
    for report in provenance_ledger.cycle_reports:
        source_module = str(report.drift_source)
        actions = map_drift_to_repair_actions(source_module)
        base_score = _round(float(report.record.bounded_drift_score))
        action_count = len(actions)
        priority = 1 if base_score >= 0.66 else (2 if base_score >= 0.33 else 3)
        for action_index, action in enumerate(actions):
            action_weight = 1.0 - (float(action_index) / float(action_count))
            rank_score = _round(max(0.0, min(1.0, (0.7 * base_score) + (0.3 * action_weight))))
            suggestion_payload = {
                "source_module": source_module,
                "repair_action": str(action),
                "advisory_priority": int(priority),
                "deterministic_rank_score": rank_score,
                "provenance_cycle_reference": int(report.cycle_index),
                "first_divergence_anchor": int(provenance_ledger.first_divergence_point),
                "advisory_only": True,
            }
            suggestion_hash = _stable_hash_dict(suggestion_payload)
            suggestions.append(
                RepairSuggestion(
                    source_module=source_module,
                    repair_action=str(action),
                    advisory_priority=int(priority),
                    deterministic_rank_score=rank_score,
                    provenance_cycle_reference=int(report.cycle_index),
                    first_divergence_anchor=int(provenance_ledger.first_divergence_point),
                    advisory_only=True,
                    stable_hash=suggestion_hash,
                    replay_identity=suggestion_hash,
                )
            )
    ranked = rank_repair_candidates(tuple(suggestions))
    bundle_payload = {
        "version": UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        "suggestions": [item.to_dict() for item in ranked],
        "advisory_only": True,
    }
    bundle_hash = _stable_hash_dict(bundle_payload)
    bundle = RepairSuggestionBundle(
        version=UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        suggestions=ranked,
        advisory_only=True,
        stable_hash=bundle_hash,
        replay_identity=bundle_hash,
    )
    ledger_payload = {
        "version": UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        "provenance_stable_hash": str(provenance_ledger.stable_hash),
        "first_divergence_cycle": int(provenance_ledger.first_divergence_point),
        "suggestion_bundle": bundle.to_dict(),
        "advisory_only": True,
    }
    ledger_hash = _stable_hash_dict(ledger_payload)
    return RepairSuggestionLedger(
        version=UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        provenance_stable_hash=str(provenance_ledger.stable_hash),
        first_divergence_cycle=int(provenance_ledger.first_divergence_point),
        suggestion_bundle=bundle,
        advisory_only=True,
        stable_hash=ledger_hash,
        replay_identity=ledger_hash,
    )


def export_repair_suggestion_bundle(bundle: RepairSuggestionBundle) -> Dict[str, Any]:
    if bundle.advisory_only is not True:
        raise ValueError("corrupted repair bundle")
    if not all(s.advisory_only is True for s in bundle.suggestions):
        raise ValueError("corrupted repair bundle")
    payload = bundle.to_dict()
    content_payload = {k: v for k, v in payload.items() if k not in ("stable_hash", "replay_identity")}
    recomputed_hash = _stable_hash_dict(content_payload)
    if recomputed_hash != str(bundle.stable_hash):
        raise ValueError("corrupted repair bundle")
    if bool(payload.get("advisory_only", False)) is not True:
        raise ValueError("corrupted repair bundle")
    return payload


def verify_repair_bundle_roundtrip(bundle: RepairSuggestionBundle) -> bool:
    """Verify bundle export survives a canonical JSON roundtrip unchanged."""
    exported = export_repair_suggestion_bundle(bundle)
    canonical_export = _canonical_json(exported)
    reparsed_payload = json.loads(canonical_export)
    if not isinstance(reparsed_payload, dict):
        raise ValueError("corrupted repair bundle")
    canonical_roundtrip = _canonical_json(reparsed_payload)
    if canonical_export != canonical_roundtrip:
        raise ValueError("corrupted repair bundle")
    return str(reparsed_payload.get("replay_identity", "")) == str(bundle.replay_identity)


def _validate_bounded_score(name: str, value: float) -> float:
    score = _round(float(value))
    if score < 0.0 or score > 1.0:
        raise ValueError(f"invalid {name}: {value}")
    return score


def _is_sha256_hex(value: str) -> bool:
    token = str(value)
    return len(token) == 64 and all(ch in "0123456789abcdef" for ch in token)


@dataclass(frozen=True)
class RuntimeStabilitySnapshot:
    cycle_index: int
    soak_window: int
    drift_score: float
    convergence_score: float
    replay_identity: str
    advisory_rank_drift: float
    provenance_source_drift: str
    first_divergence_anchor: int
    chain_digest_anchor: str
    symbolic_trace_stability: bool
    stable_hash: str
    observatory_only: bool = True

    def __post_init__(self) -> None:
        if self.observatory_only is not True:
            raise ValueError("runtime stability observatory must remain observatory-only")
        _validate_bounded_score("drift_score", self.drift_score)
        _validate_bounded_score("convergence_score", self.convergence_score)
        _validate_bounded_score("advisory_rank_drift", self.advisory_rank_drift)
        if int(self.soak_window) < 1:
            raise ValueError("invalid soak window")
        if int(self.first_divergence_anchor) < 0:
            raise ValueError("corrupted replay anchor")
        if str(self.provenance_source_drift) not in _VALID_REPAIR_SOURCE_MODULES:
            raise ValueError("invalid provenance drift source")
        if not _is_sha256_hex(str(self.chain_digest_anchor)):
            raise ValueError("corrupted replay anchor")
        if not _is_sha256_hex(str(self.stable_hash)):
            raise ValueError("malformed stability bundle")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_index": int(self.cycle_index),
            "soak_window": int(self.soak_window),
            "drift_score": _round(float(self.drift_score)),
            "convergence_score": _round(float(self.convergence_score)),
            "replay_identity": str(self.replay_identity),
            "advisory_rank_drift": _round(float(self.advisory_rank_drift)),
            "provenance_source_drift": str(self.provenance_source_drift),
            "first_divergence_anchor": int(self.first_divergence_anchor),
            "chain_digest_anchor": str(self.chain_digest_anchor),
            "symbolic_trace_stability": bool(self.symbolic_trace_stability),
            "stable_hash": str(self.stable_hash),
            "observatory_only": bool(self.observatory_only),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class StabilityConvergenceReport:
    soak_window: int
    snapshots: Tuple[RuntimeStabilitySnapshot, ...]
    mean_drift_score: float
    mean_convergence_score: float
    stable_hash: str
    replay_identity: str
    observatory_only: bool = True

    def __post_init__(self) -> None:
        if self.observatory_only is not True:
            raise ValueError("runtime stability observatory must remain observatory-only")
        if int(self.soak_window) < 1:
            raise ValueError("invalid soak window")
        _validate_bounded_score("mean_drift_score", self.mean_drift_score)
        _validate_bounded_score("mean_convergence_score", self.mean_convergence_score)
        if len(self.snapshots) != int(self.soak_window):
            raise ValueError("malformed stability bundle")
        if not _is_sha256_hex(str(self.stable_hash)):
            raise ValueError("malformed stability bundle")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "soak_window": int(self.soak_window),
            "snapshots": [s.to_dict() for s in self.snapshots],
            "mean_drift_score": _round(float(self.mean_drift_score)),
            "mean_convergence_score": _round(float(self.mean_convergence_score)),
            "stable_hash": str(self.stable_hash),
            "replay_identity": str(self.replay_identity),
            "observatory_only": bool(self.observatory_only),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class RuntimeStabilityLedger:
    version: str
    soak_window: int
    convergence_report: StabilityConvergenceReport
    advisory_stability_score: float
    provenance_stability_score: float
    stable_hash: str
    replay_identity: str
    observatory_only: bool = True

    def __post_init__(self) -> None:
        if self.observatory_only is not True:
            raise ValueError("runtime stability observatory must remain observatory-only")
        if int(self.soak_window) < 1:
            raise ValueError("invalid soak window")
        _validate_bounded_score("advisory_stability_score", self.advisory_stability_score)
        _validate_bounded_score("provenance_stability_score", self.provenance_stability_score)
        if int(self.convergence_report.soak_window) != int(self.soak_window):
            raise ValueError("malformed stability bundle")
        if not _is_sha256_hex(str(self.stable_hash)):
            raise ValueError("malformed stability bundle")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": str(self.version),
            "soak_window": int(self.soak_window),
            "convergence_report": self.convergence_report.to_dict(),
            "advisory_stability_score": _round(float(self.advisory_stability_score)),
            "provenance_stability_score": _round(float(self.provenance_stability_score)),
            "stable_hash": str(self.stable_hash),
            "replay_identity": str(self.replay_identity),
            "observatory_only": bool(self.observatory_only),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def analyze_repair_suggestion_stability(repair_ledger: RepairSuggestionLedger) -> float:
    suggestions = tuple(repair_ledger.suggestion_bundle.suggestions)
    if len(suggestions) == 0:
        raise ValueError("invalid advisory ranking state")
    ranking_signature = tuple((s.advisory_priority, s.repair_action, s.source_module) for s in suggestions)
    unique = len(set(ranking_signature))
    return _round(max(0.0, min(1.0, 1.0 - (float(unique - 1) / float(len(ranking_signature))))))


def track_drift_convergence(
    provenance_ledger: DriftProvenanceLedger,
    repair_ledger: RepairSuggestionLedger,
    *,
    soak_window: int,
) -> StabilityConvergenceReport:
    if int(soak_window) < 1:
        raise ValueError("invalid soak window")
    if int(provenance_ledger.replay_cycles) < int(soak_window):
        raise ValueError("invalid soak window")
    advisory_stability = analyze_repair_suggestion_stability(repair_ledger)
    snapshots = []
    for cycle_index, report in enumerate(provenance_ledger.cycle_reports[: int(soak_window)]):
        drift_score = _validate_bounded_score("drift_score", float(report.record.bounded_drift_score))
        convergence_score = _validate_bounded_score("convergence_score", 1.0 - drift_score)
        snapshot_payload = {
            "cycle_index": int(cycle_index),
            "soak_window": int(soak_window),
            "drift_score": drift_score,
            "convergence_score": convergence_score,
            "replay_identity": str(report.record.replay_identity),
            "advisory_rank_drift": _round(1.0 - advisory_stability),
            "provenance_source_drift": str(report.drift_source),
            "first_divergence_anchor": int(provenance_ledger.first_divergence_point),
            "chain_digest_anchor": str(provenance_ledger.chain_digest_anchor),
            "symbolic_trace_stability": bool(report.symbolic_trace_match),
            "observatory_only": True,
        }
        snapshot_hash = _stable_hash_dict(snapshot_payload)
        snapshots.append(
            RuntimeStabilitySnapshot(
                cycle_index=int(cycle_index),
                soak_window=int(soak_window),
                drift_score=drift_score,
                convergence_score=convergence_score,
                replay_identity=str(report.record.replay_identity),
                advisory_rank_drift=_round(1.0 - advisory_stability),
                provenance_source_drift=str(report.drift_source),
                first_divergence_anchor=int(provenance_ledger.first_divergence_point),
                chain_digest_anchor=str(provenance_ledger.chain_digest_anchor),
                symbolic_trace_stability=bool(report.symbolic_trace_match),
                stable_hash=snapshot_hash,
                observatory_only=True,
            )
        )
    mean_drift = _round(sum(float(s.drift_score) for s in snapshots) / float(soak_window))
    mean_convergence = _round(sum(float(s.convergence_score) for s in snapshots) / float(soak_window))
    report_payload = {
        "soak_window": int(soak_window),
        "snapshots": [s.to_dict() for s in snapshots],
        "mean_drift_score": mean_drift,
        "mean_convergence_score": mean_convergence,
        "observatory_only": True,
    }
    report_hash = _stable_hash_dict(report_payload)
    return StabilityConvergenceReport(
        soak_window=int(soak_window),
        snapshots=tuple(snapshots),
        mean_drift_score=mean_drift,
        mean_convergence_score=mean_convergence,
        stable_hash=report_hash,
        replay_identity=report_hash,
        observatory_only=True,
    )


def observe_runtime_stability(
    provenance_ledger: DriftProvenanceLedger,
    repair_ledger: RepairSuggestionLedger,
    *,
    soak_window: int,
) -> RuntimeStabilityLedger:
    convergence_report = track_drift_convergence(
        provenance_ledger,
        repair_ledger,
        soak_window=soak_window,
    )
    advisory_stability = analyze_repair_suggestion_stability(repair_ledger)
    source_set = tuple(sorted({str(r.drift_source) for r in provenance_ledger.cycle_reports[: int(soak_window)]}))
    provenance_stability = _round(max(0.0, min(1.0, 1.0 - (float(len(source_set) - 1) / 5.0))))
    ledger_payload = {
        "version": UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        "soak_window": int(soak_window),
        "convergence_report": convergence_report.to_dict(),
        "advisory_stability_score": advisory_stability,
        "provenance_stability_score": provenance_stability,
        "observatory_only": True,
    }
    ledger_hash = _stable_hash_dict(ledger_payload)
    return RuntimeStabilityLedger(
        version=UNIFIED_PHYSICS_SIMULATION_ORCHESTRATOR_VERSION,
        soak_window=int(soak_window),
        convergence_report=convergence_report,
        advisory_stability_score=advisory_stability,
        provenance_stability_score=provenance_stability,
        stable_hash=ledger_hash,
        replay_identity=ledger_hash,
        observatory_only=True,
    )


def export_runtime_stability_bundle(ledger: RuntimeStabilityLedger) -> Dict[str, Any]:
    payload = ledger.to_dict()
    required_keys = (
        "version",
        "soak_window",
        "convergence_report",
        "advisory_stability_score",
        "provenance_stability_score",
        "stable_hash",
        "replay_identity",
        "observatory_only",
    )
    if tuple(sorted(payload.keys())) != tuple(sorted(required_keys)):
        raise ValueError("malformed stability bundle")
    if bool(payload.get("observatory_only", False)) is not True:
        raise ValueError("malformed stability bundle")
    return payload


def verify_runtime_stability_roundtrip(ledger: RuntimeStabilityLedger) -> bool:
    exported = export_runtime_stability_bundle(ledger)
    content_payload = {k: v for k, v in exported.items() if k not in ("stable_hash", "replay_identity")}
    recomputed_hash = _stable_hash_dict(content_payload)
    if recomputed_hash != str(ledger.stable_hash):
        raise ValueError("malformed stability bundle")
    if int(ledger.soak_window) < 1:
        raise ValueError("invalid soak window")
    if float(ledger.convergence_report.mean_convergence_score) < 0.0 or float(
        ledger.convergence_report.mean_convergence_score
    ) > 1.0:
        raise ValueError("invalid convergence score")
    if not _is_sha256_hex(str(ledger.convergence_report.snapshots[0].chain_digest_anchor)):
        raise ValueError("corrupted replay anchor")
    if float(ledger.advisory_stability_score) < 0.0 or float(ledger.advisory_stability_score) > 1.0:
        raise ValueError("invalid advisory stability score")
    return str(exported.get("replay_identity", "")) == str(ledger.replay_identity)
