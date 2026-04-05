"""Deterministic Layer-4 quantum route optimization layer (v137.1.12).

ROUTING LAWS preserved by this module:
- DETERMINISTIC_ROUTE_MINIMIZATION_LAW
- WEIGHTED_ROUTE_LATTICE_INVARIANT
- REPLAY_SAFE_PATH_SCHEDULING_CHAIN
- PATH_DIVERGENCE_DETECTION_LAW
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Iterable, Mapping, Tuple

QUANTUM_ROUTE_OPTIMIZATION_LAYER_VERSION: str = "v137.1.12"
ROUND_DIGITS: int = 12
GENESIS_HASH: str = "0" * 64

DETERMINISTIC_ROUTE_MINIMIZATION_LAW: str = "DETERMINISTIC_ROUTE_MINIMIZATION_LAW"
WEIGHTED_ROUTE_LATTICE_INVARIANT: str = "WEIGHTED_ROUTE_LATTICE_INVARIANT"
REPLAY_SAFE_PATH_SCHEDULING_CHAIN: str = "REPLAY_SAFE_PATH_SCHEDULING_CHAIN"
PATH_DIVERGENCE_DETECTION_LAW: str = "PATH_DIVERGENCE_DETECTION_LAW"


@dataclass(frozen=True)
class RouteNode:
    node_id: str
    node_kind: str
    state_label: str
    bounded: bool
    node_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_kind": self.node_kind,
            "state_label": self.state_label,
            "bounded": self.bounded,
            "node_hash": self.node_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class RouteEdge:
    source_node: str
    target_node: str
    transition_weight: float
    allowed: bool
    edge_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_node": self.source_node,
            "target_node": self.target_node,
            "transition_weight": _round_float(self.transition_weight),
            "allowed": self.allowed,
            "edge_hash": self.edge_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class WeightedRouteLattice:
    nodes: Tuple[RouteNode, ...]
    edges: Tuple[RouteEdge, ...]
    adjacency: Tuple[Tuple[str, Tuple[Tuple[str, float], ...]], ...]
    lattice_valid: bool
    lattice_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "adjacency": [
                [source, [[target, _round_float(weight)] for target, weight in targets]]
                for source, targets in self.adjacency
            ],
            "lattice_valid": self.lattice_valid,
            "lattice_hash": self.lattice_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class ShortestPathResult:
    source_node: str
    target_node: str
    path_nodes: Tuple[str, ...]
    total_weight: float
    reachable: bool
    result_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_node": self.source_node,
            "target_node": self.target_node,
            "path_nodes": list(self.path_nodes),
            "total_weight": _round_float(self.total_weight),
            "reachable": self.reachable,
            "result_hash": self.result_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class ScheduledPathStep:
    sequence_id: int
    node_id: str
    step_weight: float
    cumulative_weight: float
    scheduler_bucket: str
    step_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "node_id": self.node_id,
            "step_weight": _round_float(self.step_weight),
            "cumulative_weight": _round_float(self.cumulative_weight),
            "scheduler_bucket": self.scheduler_bucket,
            "step_hash": self.step_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class PathDivergenceReport:
    divergence_detected: bool
    divergence_score: float
    expected_path: Tuple[str, ...]
    candidate_path: Tuple[str, ...]
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "divergence_detected": self.divergence_detected,
            "divergence_score": _round_float(self.divergence_score),
            "expected_path": list(self.expected_path),
            "candidate_path": list(self.candidate_path),
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class RouteLedgerEntry:
    sequence_id: int
    route_hash: str
    parent_hash: str
    divergence_score: float
    total_weight: float
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "route_hash": self.route_hash,
            "parent_hash": self.parent_hash,
            "divergence_score": _round_float(self.divergence_score),
            "total_weight": _round_float(self.total_weight),
            "entry_hash": self.entry_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True)
class RouteLedger:
    entries: Tuple[RouteLedgerEntry, ...]
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


@dataclass(frozen=True)
class RouteTransitionReport:
    route_found: bool
    path_length: int
    total_weight: float
    divergence_detected: bool
    deterministic: bool
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_found": self.route_found,
            "path_length": self.path_length,
            "total_weight": _round_float(self.total_weight),
            "divergence_detected": self.divergence_detected,
            "deterministic": self.deterministic,
            "report_hash": self.report_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _normalize_string(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    out = value.strip()
    if not out:
        raise ValueError(f"{name} must be a non-empty string")
    return out


def _round_float(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("float value must be finite")
    return round(numeric, ROUND_DIGITS)


def _clamp01(value: float) -> float:
    return _round_float(max(0.0, min(1.0, float(value))))


def _node_payload(node_id: str, node_kind: str, state_label: str, bounded: bool) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "node_kind": node_kind,
        "state_label": state_label,
        "bounded": bool(bounded),
    }


def _edge_payload(source_node: str, target_node: str, transition_weight: float, allowed: bool) -> dict[str, Any]:
    return {
        "source_node": source_node,
        "target_node": target_node,
        "transition_weight": _round_float(transition_weight),
        "allowed": bool(allowed),
    }


def normalize_route_inputs(
    nodes: Iterable[Mapping[str, Any] | RouteNode],
    edges: Iterable[Mapping[str, Any] | RouteEdge],
    *,
    allow_self_loops: bool = False,
) -> tuple[Tuple[RouteNode, ...], Tuple[RouteEdge, ...]]:
    """Canonicalize route inputs for deterministic route minimization.

    Preserves DETERMINISTIC_ROUTE_MINIMIZATION_LAW via explicit validation and
    deterministic ordering.
    """
    raw_nodes = tuple(nodes)
    raw_edges = tuple(edges)

    normalized_nodes: list[RouteNode] = []
    seen_node_ids: set[str] = set()
    for raw in raw_nodes:
        node_id = _normalize_string("node_id", raw.node_id if isinstance(raw, RouteNode) else raw.get("node_id"))
        if node_id in seen_node_ids:
            raise ValueError(f"duplicate node_id={node_id!r}")
        seen_node_ids.add(node_id)

        node_kind = _normalize_string("node_kind", raw.node_kind if isinstance(raw, RouteNode) else raw.get("node_kind", "generic"))
        state_label = _normalize_string("state_label", raw.state_label if isinstance(raw, RouteNode) else raw.get("state_label", node_id))
        bounded = bool(raw.bounded if isinstance(raw, RouteNode) else raw.get("bounded", True))

        payload = _node_payload(node_id, node_kind, state_label, bounded)
        normalized_nodes.append(RouteNode(**payload, node_hash=_hash_sha256(payload)))

    normalized_nodes.sort(key=lambda item: item.node_id)
    node_ids = {n.node_id for n in normalized_nodes}

    normalized_edges: list[RouteEdge] = []
    for raw in raw_edges:
        source = _normalize_string("source_node", raw.source_node if isinstance(raw, RouteEdge) else raw.get("source_node"))
        target = _normalize_string("target_node", raw.target_node if isinstance(raw, RouteEdge) else raw.get("target_node"))
        if source not in node_ids or target not in node_ids:
            raise ValueError(f"edge endpoint missing from nodes: {source!r}->{target!r}")
        if source == target and not allow_self_loops:
            raise ValueError(f"self-loop rejected for node_id={source!r}")

        weight = _round_float(raw.transition_weight if isinstance(raw, RouteEdge) else raw.get("transition_weight"))
        if weight < 0.0:
            raise ValueError("transition_weight must be non-negative")
        allowed = bool(raw.allowed if isinstance(raw, RouteEdge) else raw.get("allowed", True))
        payload = _edge_payload(source, target, weight, allowed)
        normalized_edges.append(RouteEdge(**payload, edge_hash=_hash_sha256(payload)))

    normalized_edges.sort(
        key=lambda item: (item.source_node, item.target_node, item.transition_weight, item.allowed),
    )
    return (tuple(normalized_nodes), tuple(normalized_edges))


def build_weighted_route_lattice(
    nodes: Tuple[RouteNode, ...],
    edges: Tuple[RouteEdge, ...],
) -> WeightedRouteLattice:
    """Build deterministic weighted route lattice preserving stable traversal."""
    node_ids = tuple(node.node_id for node in nodes)
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("duplicate node ids are not allowed")

    valid_ids = set(node_ids)
    adjacency_map: dict[str, list[tuple[str, float]]] = {node_id: [] for node_id in node_ids}
    min_allowed_weights: dict[tuple[str, str], float] = {}

    for edge in edges:
        if edge.source_node not in valid_ids or edge.target_node not in valid_ids:
            raise ValueError("edge endpoint missing from node set")
        weight = _round_float(edge.transition_weight)
        if weight < 0.0:
            raise ValueError("transition_weight must be non-negative")
        if not edge.allowed:
            continue
        key = (edge.source_node, edge.target_node)
        previous = min_allowed_weights.get(key)
        if previous is None or weight < previous:
            min_allowed_weights[key] = weight

    for (source, target), weight in sorted(min_allowed_weights.items(), key=lambda item: (item[0][0], item[0][1])):
        adjacency_map[source].append((target, weight))

    adjacency: Tuple[Tuple[str, Tuple[Tuple[str, float], ...]], ...] = tuple(
        (source, tuple(sorted(targets, key=lambda t: (t[0], t[1]))))
        for source, targets in sorted(adjacency_map.items(), key=lambda p: p[0])
    )

    canonical_nodes = sorted(nodes, key=lambda n: n.node_id)
    canonical_edges = sorted(
        edges, key=lambda e: (e.source_node, e.target_node, e.transition_weight, e.allowed)
    )
    payload = {
        "nodes": [node.to_dict() for node in canonical_nodes],
        "edges": [edge.to_dict() for edge in canonical_edges],
        "adjacency": [[s, [[t, _round_float(w)] for t, w in ts]] for s, ts in adjacency],
        "lattice_valid": True,
    }
    lattice_hash = _hash_sha256(payload)
    lattice = WeightedRouteLattice(
        nodes=nodes,
        edges=edges,
        adjacency=adjacency,
        lattice_valid=True,
        lattice_hash=lattice_hash,
    )
    if not validate_weighted_route_lattice(lattice):
        raise ValueError("constructed weighted route lattice failed self-validation")
    return lattice


def validate_weighted_route_lattice(lattice: WeightedRouteLattice) -> bool:
    """Validate WEIGHTED_ROUTE_LATTICE_INVARIANT and hash consistency."""
    node_ids = tuple(node.node_id for node in lattice.nodes)
    if len(set(node_ids)) != len(node_ids):
        return False

    valid_ids = set(node_ids)
    adjacency_expected: dict[str, list[tuple[str, float]]] = {node_id: [] for node_id in node_ids}
    min_allowed_weights: dict[tuple[str, str], float] = {}

    for edge in lattice.edges:
        if edge.source_node not in valid_ids or edge.target_node not in valid_ids:
            return False
        if not math.isfinite(float(edge.transition_weight)) or edge.transition_weight < 0.0:
            return False
        if edge.allowed:
            key = (edge.source_node, edge.target_node)
            previous = min_allowed_weights.get(key)
            if previous is None or edge.transition_weight < previous:
                min_allowed_weights[key] = edge.transition_weight

    for (source, target), weight in sorted(min_allowed_weights.items(), key=lambda item: (item[0][0], item[0][1])):
        adjacency_expected[source].append((target, _round_float(weight)))

    expected_adjacency: Tuple[Tuple[str, Tuple[Tuple[str, float], ...]], ...] = tuple(
        (source, tuple(sorted(targets, key=lambda t: (t[0], t[1]))))
        for source, targets in sorted(adjacency_expected.items(), key=lambda p: p[0])
    )
    actual_adjacency = tuple(
        (source, tuple((target, _round_float(weight)) for target, weight in targets))
        for source, targets in lattice.adjacency
    )
    if actual_adjacency != expected_adjacency:
        return False

    canonical_nodes = sorted(lattice.nodes, key=lambda n: n.node_id)
    canonical_edges = sorted(
        lattice.edges, key=lambda e: (e.source_node, e.target_node, e.transition_weight, e.allowed)
    )
    payload = {
        "nodes": [node.to_dict() for node in canonical_nodes],
        "edges": [edge.to_dict() for edge in canonical_edges],
        "adjacency": [[s, [[t, _round_float(w)] for t, w in ts]] for s, ts in expected_adjacency],
        "lattice_valid": True,
    }
    expected_hash = _hash_sha256(payload)
    if lattice.lattice_hash != expected_hash:
        return False
    if lattice.lattice_valid is not True:
        return False
    return True


def compute_deterministic_shortest_path(
    lattice: WeightedRouteLattice,
    source_node: str,
    target_node: str,
) -> ShortestPathResult:
    """Compute deterministic shortest path with lexicographic tie-break."""
    if not validate_weighted_route_lattice(lattice):
        raise ValueError("lattice must be valid before shortest-path computation")

    source = _normalize_string("source_node", source_node)
    target = _normalize_string("target_node", target_node)
    node_ids = {node.node_id for node in lattice.nodes}
    if source not in node_ids or target not in node_ids:
        raise ValueError("source_node and target_node must exist in lattice")

    adjacency = {source_id: targets for source_id, targets in lattice.adjacency}
    heap: list[tuple[float, tuple[str, ...], str]] = []
    heappush(heap, (0.0, (source,), source))
    best: dict[str, tuple[float, tuple[str, ...]]] = {source: (0.0, (source,))}

    while heap:
        cost, path, node = heappop(heap)
        best_cost, best_path = best[node]
        if _round_float(cost) > _round_float(best_cost):
            continue
        if node == target and path == best_path:
            break

        for neighbor, weight in adjacency.get(node, ()):
            step_weight = _round_float(weight)
            if step_weight < 0.0:
                raise ValueError("negative edge encountered in validated lattice")
            candidate_cost = _round_float(cost + step_weight)
            candidate_path = path + (neighbor,)
            existing = best.get(neighbor)
            if existing is None or candidate_cost < existing[0] or (
                candidate_cost == existing[0] and candidate_path < existing[1]
            ):
                best[neighbor] = (candidate_cost, candidate_path)
                heappush(heap, (candidate_cost, candidate_path, neighbor))

    if target not in best:
        result_payload = {
            "source_node": source,
            "target_node": target,
            "path_nodes": [],
            "total_weight": 0.0,
            "reachable": False,
        }
        return ShortestPathResult(
            source_node=source,
            target_node=target,
            path_nodes=(),
            total_weight=0.0,
            reachable=False,
            result_hash=_hash_sha256(result_payload),
        )

    total_weight, path_nodes = best[target]
    result_payload = {
        "source_node": source,
        "target_node": target,
        "path_nodes": list(path_nodes),
        "total_weight": _round_float(total_weight),
        "reachable": True,
    }
    return ShortestPathResult(
        source_node=source,
        target_node=target,
        path_nodes=path_nodes,
        total_weight=_round_float(total_weight),
        reachable=True,
        result_hash=_hash_sha256(result_payload),
    )


def schedule_shortest_path(
    lattice: WeightedRouteLattice,
    path_result: ShortestPathResult,
) -> Tuple[ScheduledPathStep, ...]:
    """Create replay-safe deterministic schedule from shortest path."""
    if not path_result.reachable or not path_result.path_nodes:
        return ()

    adjacency_lookup = {
        source: {target: weight for target, weight in targets}
        for source, targets in lattice.adjacency
    }

    steps: list[ScheduledPathStep] = []
    cumulative = 0.0
    for idx, node_id in enumerate(path_result.path_nodes):
        if idx == 0:
            step_weight = 0.0
            bucket = "entry" if len(path_result.path_nodes) > 1 else "terminal"
        else:
            prev = path_result.path_nodes[idx - 1]
            if node_id not in adjacency_lookup.get(prev, {}):
                raise ValueError("path_result contains edge not present in lattice adjacency")
            step_weight = _round_float(adjacency_lookup[prev][node_id])
            bucket = "terminal" if idx == len(path_result.path_nodes) - 1 else "transit"

        cumulative = _round_float(cumulative + step_weight)
        payload = {
            "sequence_id": idx,
            "node_id": node_id,
            "step_weight": _round_float(step_weight),
            "cumulative_weight": _round_float(cumulative),
            "scheduler_bucket": bucket,
        }
        steps.append(
            ScheduledPathStep(
                sequence_id=idx,
                node_id=node_id,
                step_weight=_round_float(step_weight),
                cumulative_weight=_round_float(cumulative),
                scheduler_bucket=bucket,
                step_hash=_hash_sha256(payload),
            )
        )
    return tuple(steps)


def detect_path_divergence(
    expected_path: Iterable[str],
    candidate_path: Iterable[str],
) -> PathDivergenceReport:
    """Compute bounded deterministic divergence score in [0, 1]."""
    expected = tuple(_normalize_string("expected_path item", v) for v in expected_path)
    candidate = tuple(_normalize_string("candidate_path item", v) for v in candidate_path)

    if expected == candidate:
        payload = {
            "divergence_detected": False,
            "divergence_score": 0.0,
            "expected_path": list(expected),
            "candidate_path": list(candidate),
        }
        return PathDivergenceReport(
            divergence_detected=False,
            divergence_score=0.0,
            expected_path=expected,
            candidate_path=candidate,
            report_hash=_hash_sha256(payload),
        )

    max_len = max(max(len(expected), len(candidate)), 1)
    length_delta = abs(len(expected) - len(candidate)) / max_len
    if not math.isfinite(length_delta):
        raise ValueError("length_delta must be finite")

    mismatch_index = 0
    for idx in range(min(len(expected), len(candidate))):
        if expected[idx] != candidate[idx]:
            mismatch_index = idx
            break
    else:
        mismatch_index = min(len(expected), len(candidate))

    first_mismatch_penalty = (mismatch_index + 1) / max_len
    if not math.isfinite(first_mismatch_penalty):
        raise ValueError("first_mismatch_penalty must be finite")

    union_size = max(len(set(expected) | set(candidate)), 1)
    shared_node_count = len(set(expected) & set(candidate))
    overlap_penalty = 1.0 - (shared_node_count / union_size)
    if not math.isfinite(overlap_penalty):
        raise ValueError("overlap_penalty must be finite")

    score = (0.4 * length_delta) + (0.3 * first_mismatch_penalty) + (0.3 * overlap_penalty)
    if not math.isfinite(score):
        raise ValueError("divergence_score must be finite")
    score = _clamp01(score)

    payload = {
        "divergence_detected": score > 0.0,
        "divergence_score": score,
        "expected_path": list(expected),
        "candidate_path": list(candidate),
    }
    return PathDivergenceReport(
        divergence_detected=score > 0.0,
        divergence_score=score,
        expected_path=expected,
        candidate_path=candidate,
        report_hash=_hash_sha256(payload),
    )


def empty_route_ledger() -> RouteLedger:
    return RouteLedger(entries=(), head_hash=GENESIS_HASH, chain_valid=True)


def validate_route_ledger(ledger: RouteLedger) -> bool:
    """Validate REPLAY_SAFE_PATH_SCHEDULING_CHAIN hash-link integrity."""
    expected_parent = GENESIS_HASH
    expected_head = GENESIS_HASH

    for idx, entry in enumerate(ledger.entries):
        if entry.sequence_id != idx:
            return False
        if entry.parent_hash != expected_parent:
            return False
        if not math.isfinite(float(entry.divergence_score)) or not math.isfinite(float(entry.total_weight)):
            return False

        payload = {
            "sequence_id": entry.sequence_id,
            "route_hash": entry.route_hash,
            "parent_hash": entry.parent_hash,
            "divergence_score": _round_float(entry.divergence_score),
            "total_weight": _round_float(entry.total_weight),
        }
        expected_entry_hash = _hash_sha256(payload)
        if entry.entry_hash != expected_entry_hash:
            return False

        expected_parent = entry.entry_hash
        expected_head = _hash_sha256({"parent": expected_head, "entry": entry.entry_hash})

    if ledger.head_hash != expected_head:
        return False
    if ledger.chain_valid is not True:
        return False
    return True


def append_route_ledger_entry(
    ledger: RouteLedger | None,
    route_hash: str,
    divergence_score: float,
    total_weight: float,
) -> RouteLedger:
    """Append one deterministic ledger entry preserving parent-linked integrity."""
    current = ledger if ledger is not None else empty_route_ledger()
    if not validate_route_ledger(current):
        raise ValueError("prior ledger failed integrity validation")

    route_hash_norm = _normalize_string("route_hash", route_hash)
    if not math.isfinite(float(divergence_score)):
        raise ValueError("divergence_score must be finite")
    if not math.isfinite(float(total_weight)):
        raise ValueError("total_weight must be finite")
    divergence = _clamp01(divergence_score)
    weight = _round_float(total_weight)
    if weight < 0.0:
        raise ValueError("total_weight must be non-negative")

    sequence_id = len(current.entries)
    parent_hash = current.entries[-1].entry_hash if current.entries else GENESIS_HASH
    payload = {
        "sequence_id": sequence_id,
        "route_hash": route_hash_norm,
        "parent_hash": parent_hash,
        "divergence_score": divergence,
        "total_weight": weight,
    }
    entry_hash = _hash_sha256(payload)
    entry = RouteLedgerEntry(
        sequence_id=sequence_id,
        route_hash=route_hash_norm,
        parent_hash=parent_hash,
        divergence_score=divergence,
        total_weight=weight,
        entry_hash=entry_hash,
    )
    entries = current.entries + (entry,)
    new_head = _hash_sha256({"parent": current.head_hash, "entry": entry_hash})
    updated = RouteLedger(entries=entries, head_hash=new_head, chain_valid=True)
    if not validate_route_ledger(updated):
        raise ValueError("updated ledger failed integrity validation")
    return updated


def run_quantum_route_optimization_layer(
    nodes: Iterable[Mapping[str, Any] | RouteNode],
    edges: Iterable[Mapping[str, Any] | RouteEdge],
    source_node: str,
    target_node: str,
    *,
    expected_path: Iterable[str] | None = None,
    prior_route_ledger: RouteLedger | None = None,
    allow_self_loops: bool = False,
) -> tuple[
    WeightedRouteLattice,
    ShortestPathResult,
    Tuple[ScheduledPathStep, ...],
    PathDivergenceReport,
    RouteTransitionReport,
    RouteLedger,
]:
    """Deterministic Layer-4 route optimization orchestration wrapper."""
    normalized_nodes, normalized_edges = normalize_route_inputs(
        nodes,
        edges,
        allow_self_loops=allow_self_loops,
    )
    lattice = build_weighted_route_lattice(normalized_nodes, normalized_edges)
    path_result = compute_deterministic_shortest_path(lattice, source_node, target_node)
    schedule = schedule_shortest_path(lattice, path_result)

    expected = tuple(expected_path) if expected_path is not None else path_result.path_nodes
    divergence = detect_path_divergence(expected, path_result.path_nodes)

    transition_payload = {
        "route_found": path_result.reachable,
        "path_length": len(path_result.path_nodes),
        "total_weight": _round_float(path_result.total_weight),
        "divergence_detected": divergence.divergence_detected,
        "deterministic": True,
    }
    transition_report = RouteTransitionReport(
        route_found=path_result.reachable,
        path_length=len(path_result.path_nodes),
        total_weight=_round_float(path_result.total_weight),
        divergence_detected=divergence.divergence_detected,
        deterministic=True,
        report_hash=_hash_sha256(transition_payload),
    )

    route_identity = _hash_sha256(
        {
            "lattice_hash": lattice.lattice_hash,
            "path_result_hash": path_result.result_hash,
            "schedule_hashes": [step.step_hash for step in schedule],
            "transition_report_hash": transition_report.report_hash,
        }
    )
    updated_ledger = append_route_ledger_entry(
        prior_route_ledger,
        route_hash=route_identity,
        divergence_score=divergence.divergence_score,
        total_weight=path_result.total_weight,
    )
    return (
        lattice,
        path_result,
        schedule,
        divergence,
        transition_report,
        updated_ledger,
    )
