"""Deterministic Layer-4 Complex Systems Phase Engine (v137.1.9)."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

COMPLEX_SYSTEMS_PHASE_ENGINE_VERSION: str = "v137.1.9"
ROUND_DIGITS: int = 12
GENESIS_HASH: str = "0" * 64

# Theory invariants (explicitly preserved by this module)
PHASE_STATE_DETECTION_LAW: str = "PHASE_STATE_DETECTION_LAW"
DETERMINISTIC_BIFURCATION_WARNING: str = "DETERMINISTIC_BIFURCATION_WARNING"
ATTRACTOR_TRANSITION_INVARIANT: str = "ATTRACTOR_TRANSITION_INVARIANT"
REPLAY_SAFE_PHASE_CHAIN: str = "REPLAY_SAFE_PHASE_CHAIN"


@dataclass(frozen=True)
class PhaseStateSnapshot:
    state_id: str
    phase_label: str
    stability_score: float
    transition_pressure: float
    bifurcation_score: float
    attractor_cluster: str
    replay_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "phase_label": self.phase_label,
            "stability_score": _round_float(self.stability_score),
            "transition_pressure": _round_float(self.transition_pressure),
            "bifurcation_score": _round_float(self.bifurcation_score),
            "attractor_cluster": self.attractor_cluster,
            "replay_hash": self.replay_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class BifurcationWarningReport:
    warning_score: float
    transition_imminent: bool
    confidence_score: float
    decision_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "warning_score": _round_float(self.warning_score),
            "transition_imminent": self.transition_imminent,
            "confidence_score": _round_float(self.confidence_score),
            "decision_hash": self.decision_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class AttractorTransitionEdge:
    source_state: str
    target_state: str
    transition_weight: float
    stable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_state": self.source_state,
            "target_state": self.target_state,
            "transition_weight": _round_float(self.transition_weight),
            "stable": self.stable,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class AttractorTransitionGraph:
    nodes: tuple[str, ...]
    edges: tuple[AttractorTransitionEdge, ...]
    graph_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": list(self.nodes),
            "edges": [edge.to_dict() for edge in self.edges],
            "graph_hash": self.graph_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PhaseLedgerEntry:
    sequence_id: int
    phase_hash: str
    parent_hash: str
    warning_score: float
    stability_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "phase_hash": self.phase_hash,
            "parent_hash": self.parent_hash,
            "warning_score": _round_float(self.warning_score),
            "stability_score": _round_float(self.stability_score),
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PhaseLedger:
    entries: tuple[PhaseLedgerEntry, ...]
    head_hash: str
    chain_valid: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "head_hash": self.head_hash,
            "chain_valid": self.chain_valid,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _round_float(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


def _clamp01(value: float) -> float:
    return _round_float(max(0.0, min(1.0, float(value))))


def _hash_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _normalize_named_vector(
    vector: Mapping[str, float] | Iterable[tuple[str, float]],
    *,
    vector_name: str,
) -> tuple[tuple[str, float], ...]:
    items = tuple(vector.items()) if isinstance(vector, Mapping) else tuple(vector)
    normalized: list[tuple[str, float]] = []

    for idx, item in enumerate(items):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{vector_name} entry at index {idx} must be a (name, value) pair")
        name, value = item
        if not isinstance(name, str) or not name:
            raise ValueError(f"{vector_name} names must be non-empty strings")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{vector_name} value for {name!r} must be finite")
        normalized.append((name, _round_float(numeric)))

    normalized.sort(key=lambda pair: pair[0])
    for i in range(1, len(normalized)):
        if normalized[i - 1][0] == normalized[i][0]:
            raise ValueError(f"duplicate {vector_name} entry name: {normalized[i][0]!r}")
    return tuple(normalized)


def normalize_phase_inputs(
    stability_metrics: Mapping[str, float] | Iterable[tuple[str, float]],
    perturbation_signals: Mapping[str, float] | Iterable[tuple[str, float]],
    state_vector: Mapping[str, float] | Iterable[tuple[str, float]] | None = None,
) -> dict[str, tuple[tuple[str, float], ...]]:
    """Deterministically sort and validate all phase inputs."""
    normalized_stability = _normalize_named_vector(
        stability_metrics,
        vector_name="stability_metrics",
    )
    normalized_perturbation = _normalize_named_vector(
        perturbation_signals,
        vector_name="perturbation_signals",
    )
    normalized_state = (
        _normalize_named_vector(state_vector, vector_name="state_vector")
        if state_vector is not None
        else ()
    )
    return {
        "stability_metrics": normalized_stability,
        "perturbation_signals": normalized_perturbation,
        "state_vector": normalized_state,
    }


def compute_bifurcation_warning_score(
    current_stability: float,
    prior_stability: float,
    transition_pressure: float,
    attractor_divergence: float,
) -> float:
    """DETERMINISTIC_BIFURCATION_WARNING: bounded [0, 1] phase warning."""
    cs = float(current_stability)
    ps = float(prior_stability)
    tp = float(transition_pressure)
    ad = float(attractor_divergence)
    for name, val in (
        ("current_stability", cs),
        ("prior_stability", ps),
        ("transition_pressure", tp),
        ("attractor_divergence", ad),
    ):
        if not math.isfinite(val):
            raise ValueError(f"{name} must be a finite float, got {val!r}")
    raw = abs(cs - ps) + tp + ad
    return _clamp01(raw)


def detect_phase_state(
    stability_metrics: Mapping[str, float] | Iterable[tuple[str, float]],
    perturbation_signals: Mapping[str, float] | Iterable[tuple[str, float]],
    prior_phase_state: PhaseStateSnapshot | None = None,
) -> PhaseStateSnapshot:
    """PHASE_STATE_DETECTION_LAW: deterministic phase-label assignment."""
    normalized = normalize_phase_inputs(stability_metrics, perturbation_signals)
    stability_values = tuple(v for _, v in normalized["stability_metrics"])
    perturbation_values = tuple(v for _, v in normalized["perturbation_signals"])

    stability_score = _clamp01(
        sum(stability_values) / float(len(stability_values)) if stability_values else 0.0,
    )
    transition_pressure = _clamp01(
        sum(abs(v) for v in perturbation_values) / float(len(perturbation_values))
        if perturbation_values
        else 0.0,
    )

    prior_stability = (
        prior_phase_state.stability_score if prior_phase_state is not None else stability_score
    )
    attractor_divergence = _clamp01(abs(stability_score - transition_pressure))
    bifurcation_score = compute_bifurcation_warning_score(
        current_stability=stability_score,
        prior_stability=prior_stability,
        transition_pressure=transition_pressure,
        attractor_divergence=attractor_divergence,
    )

    cluster_seed = {
        "stability_keys": [k for k, _ in normalized["stability_metrics"]],
        "perturbation_keys": [k for k, _ in normalized["perturbation_signals"]],
    }
    attractor_cluster = "cluster_" + _hash_sha256(cluster_seed)[:12]

    if (
        prior_phase_state is not None
        and prior_phase_state.attractor_cluster == attractor_cluster
        and transition_pressure <= 0.08
        and stability_score >= 0.75
    ):
        phase_label = "attractor_locked"
    elif stability_score >= 0.8 and transition_pressure <= 0.2:
        phase_label = "stable"
    elif stability_score >= 0.6 and transition_pressure <= 0.45:
        phase_label = "metastable"
    elif bifurcation_score >= 0.85:
        phase_label = "bifurcating"
    else:
        phase_label = "transitional"

    state_payload = {
        "normalized": normalized,
        "prior_replay_hash": prior_phase_state.replay_hash if prior_phase_state else "",
        "phase_label": phase_label,
    }
    state_id = _hash_sha256(state_payload)

    snapshot_payload = {
        "state_id": state_id,
        "phase_label": phase_label,
        "stability_score": stability_score,
        "transition_pressure": transition_pressure,
        "bifurcation_score": bifurcation_score,
        "attractor_cluster": attractor_cluster,
    }

    return PhaseStateSnapshot(
        state_id=state_id,
        phase_label=phase_label,
        stability_score=stability_score,
        transition_pressure=transition_pressure,
        bifurcation_score=bifurcation_score,
        attractor_cluster=attractor_cluster,
        replay_hash=_hash_sha256(snapshot_payload),
    )


def build_attractor_transition_graph(
    nodes: Iterable[str],
    edges: Iterable[AttractorTransitionEdge | tuple[str, str, float, bool]],
) -> AttractorTransitionGraph:
    """ATTRACTOR_TRANSITION_INVARIANT: deterministic graph construction."""
    unique_nodes = sorted({str(node) for node in nodes})
    if any(node == "" for node in unique_nodes):
        raise ValueError("nodes must be non-empty strings")
    node_set = set(unique_nodes)

    edge_map: dict[tuple[str, str], AttractorTransitionEdge] = {}
    for edge in edges:
        if isinstance(edge, AttractorTransitionEdge):
            parsed = edge
        else:
            if not isinstance(edge, tuple) or len(edge) != 4:
                raise ValueError(
                    f"edge must be an AttractorTransitionEdge or a 4-tuple "
                    f"(source, target, weight, stable), got {edge!r}"
                )
            source, target, weight, stable = edge
            parsed = AttractorTransitionEdge(
                source_state=str(source),
                target_state=str(target),
                transition_weight=_clamp01(float(weight)),
                stable=bool(stable),
            )

        if parsed.source_state not in node_set or parsed.target_state not in node_set:
            raise ValueError("edge references unknown node")
        key = (parsed.source_state, parsed.target_state)
        if key in edge_map:
            raise ValueError(f"duplicate edge detected for {key!r}")
        edge_map[key] = AttractorTransitionEdge(
            source_state=parsed.source_state,
            target_state=parsed.target_state,
            transition_weight=_clamp01(parsed.transition_weight),
            stable=parsed.stable,
        )

    sorted_edges = tuple(
        edge_map[key]
        for key in sorted(
            edge_map.keys(),
            key=lambda k: (k[0], k[1], edge_map[k].transition_weight, edge_map[k].stable),
        )
    )

    graph_payload = {
        "nodes": unique_nodes,
        "edges": [edge.to_dict() for edge in sorted_edges],
    }
    return AttractorTransitionGraph(
        nodes=tuple(unique_nodes),
        edges=sorted_edges,
        graph_hash=_hash_sha256(graph_payload),
    )


def detect_macro_state_transition(
    phase_state: PhaseStateSnapshot,
    noise_level: float,
    prior_stability: float | None = None,
    attractor_divergence: float | None = None,
) -> BifurcationWarningReport:
    nl = float(noise_level)
    if not math.isfinite(nl):
        raise ValueError(f"noise_level must be a finite float, got {nl!r}")
    current_stability = phase_state.stability_score
    prior = current_stability if prior_stability is None else float(prior_stability)
    divergence = (
        phase_state.bifurcation_score
        if attractor_divergence is None
        else float(attractor_divergence)
    )

    warning_score = compute_bifurcation_warning_score(
        current_stability=current_stability,
        prior_stability=prior,
        transition_pressure=phase_state.transition_pressure,
        attractor_divergence=divergence,
    )
    confidence_score = _clamp01(1.0 - abs(nl - current_stability))
    transition_imminent = warning_score >= 0.7

    payload = {
        "warning_score": warning_score,
        "transition_imminent": transition_imminent,
        "confidence_score": confidence_score,
        "phase_replay_hash": phase_state.replay_hash,
    }
    return BifurcationWarningReport(
        warning_score=warning_score,
        transition_imminent=transition_imminent,
        confidence_score=confidence_score,
        decision_hash=_hash_sha256(payload),
    )


def append_phase_ledger_entry(
    phase_state: PhaseStateSnapshot,
    warning_report: BifurcationWarningReport,
    prior_ledger: PhaseLedger | None = None,
) -> PhaseLedger:
    """REPLAY_SAFE_PHASE_CHAIN: append a deterministic SHA-256 chain entry."""
    if prior_ledger is not None and not validate_phase_ledger(prior_ledger):
        raise ValueError("cannot append to invalid ledger")

    entries = prior_ledger.entries if prior_ledger is not None else ()
    parent_hash = prior_ledger.head_hash if prior_ledger is not None else GENESIS_HASH

    entry = PhaseLedgerEntry(
        sequence_id=len(entries),
        phase_hash=phase_state.replay_hash,
        parent_hash=parent_hash,
        warning_score=warning_report.warning_score,
        stability_score=phase_state.stability_score,
    )

    head_hash = _hash_sha256({"parent_hash": parent_hash, "entry": entry.to_dict()})
    tentative = PhaseLedger(
        entries=entries + (entry,),
        head_hash=head_hash,
        chain_valid=True,
    )
    return PhaseLedger(
        entries=tentative.entries,
        head_hash=tentative.head_hash,
        chain_valid=validate_phase_ledger(tentative),
    )


def validate_phase_ledger(ledger: PhaseLedger) -> bool:
    if len(ledger.head_hash) != 64:
        return False
    parent_hash = GENESIS_HASH
    for index, entry in enumerate(ledger.entries):
        if entry.sequence_id != index:
            return False
        if entry.parent_hash != parent_hash:
            return False
        parent_hash = _hash_sha256({"parent_hash": parent_hash, "entry": entry.to_dict()})
    return parent_hash == ledger.head_hash


def run_complex_systems_phase_engine(
    stability_metrics: Mapping[str, float] | Iterable[tuple[str, float]],
    perturbation_signals: Mapping[str, float] | Iterable[tuple[str, float]],
    attractor_edges: Iterable[AttractorTransitionEdge | tuple[str, str, float, bool]] = (),
    prior_phase_state: PhaseStateSnapshot | None = None,
    prior_ledger: PhaseLedger | None = None,
    noise_level: float = 0.0,
) -> tuple[
    PhaseStateSnapshot,
    BifurcationWarningReport,
    AttractorTransitionGraph,
    PhaseLedger,
]:
    phase_state = detect_phase_state(
        stability_metrics=stability_metrics,
        perturbation_signals=perturbation_signals,
        prior_phase_state=prior_phase_state,
    )

    prior_stability = prior_phase_state.stability_score if prior_phase_state else phase_state.stability_score
    attractor_divergence = _clamp01(abs(phase_state.stability_score - phase_state.transition_pressure))
    warning_report = detect_macro_state_transition(
        phase_state=phase_state,
        noise_level=noise_level,
        prior_stability=prior_stability,
        attractor_divergence=attractor_divergence,
    )

    graph_nodes = {phase_state.state_id}
    for edge in attractor_edges:
        if isinstance(edge, AttractorTransitionEdge):
            graph_nodes.add(edge.source_state)
            graph_nodes.add(edge.target_state)
        else:
            src, dst, _, _ = edge
            graph_nodes.add(str(src))
            graph_nodes.add(str(dst))
    graph = build_attractor_transition_graph(sorted(graph_nodes), attractor_edges)

    ledger = append_phase_ledger_entry(
        phase_state=phase_state,
        warning_report=warning_report,
        prior_ledger=prior_ledger,
    )

    return (phase_state, warning_report, graph, ledger)
