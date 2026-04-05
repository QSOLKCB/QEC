from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

AUTONOMOUS_PLANNING_PREKERNEL_VERSION: str = "v137.1.19"
_PRECISION = 12
_MISSING_EDGE_PENALTY = 1.0e6


def _round64(value: float) -> float:
    return float(round(float(value), _PRECISION))


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return _canonical_json(payload).encode("utf-8")


def _sha256_hex_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_hex_mapping(payload: Mapping[str, Any]) -> str:
    return _sha256_hex_bytes(_canonical_json_bytes(payload))


def _norm_token(value: str, *, name: str) -> str:
    token = str(value).strip()
    if not token:
        raise ValueError(f"{name} must be a non-empty token")
    return token


def _normalize_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("world_state values must be finite")
        return _round64(numeric)
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        normalized_items: list[tuple[str, Any]] = []
        for key, item in value.items():
            norm_key = _norm_token(str(key), name="world_state key")
            normalized_items.append((norm_key, _normalize_json_value(item)))
        normalized_items.sort(key=lambda item: item[0])
        normalized: dict[str, Any] = {}
        for key, item in normalized_items:
            if key in normalized:
                raise ValueError(f"duplicate world_state key after normalization: {key}")
            normalized[key] = item
        return normalized
    if isinstance(value, (tuple, list)):
        return [_normalize_json_value(item) for item in value]
    raise ValueError(f"unsupported world_state value type: {type(value).__name__}")


def _normalize_edge_costs(edge_costs: Mapping[tuple[str, str], float] | None) -> dict[tuple[str, str], float]:
    if edge_costs is None:
        return {}
    normalized: dict[tuple[str, str], float] = {}
    for key, cost in edge_costs.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("edge_cost keys must be (source, target) tuples")
        src = _norm_token(key[0], name="edge source")
        dst = _norm_token(key[1], name="edge target")
        numeric = float(cost)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("edge costs must be finite and >= 0")
        normalized[(src, dst)] = _round64(numeric)
    return normalized


def _normalize_routes(candidate_routes: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    if not candidate_routes:
        raise ValueError("candidate_routes must not be empty")
    normalized: list[tuple[str, ...]] = []
    for index, route in enumerate(candidate_routes):
        if not route:
            raise ValueError(f"candidate route at index {index} must not be empty")
        normalized_route = tuple(_norm_token(step, name="route step") for step in route)
        normalized.append(normalized_route)
    return tuple(normalized)


@dataclass(frozen=True)
class WorldStateSnapshot:
    normalized_state: Mapping[str, Any]
    canonical_json: str
    replay_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_state": self.normalized_state,
            "canonical_json": self.canonical_json,
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class PlanningNode:
    node_id: str
    outgoing_degree: int
    incoming_degree: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "outgoing_degree": int(self.outgoing_degree),
            "incoming_degree": int(self.incoming_degree),
        }


@dataclass(frozen=True)
class PlanningEdge:
    source: str
    target: str
    cost: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "cost": _round64(self.cost),
        }


@dataclass(frozen=True)
class PlanningGraph:
    nodes: tuple[PlanningNode, ...]
    edges: tuple[PlanningEdge, ...]
    graph_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "graph_identity": self.graph_identity,
        }


@dataclass(frozen=True)
class PolicySearchResult:
    selected_route: tuple[str, ...]
    selected_objective: float
    ranked_routes: tuple[tuple[tuple[str, ...], float], ...]
    replay_identity: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_route": list(self.selected_route),
            "selected_objective": _round64(self.selected_objective),
            "ranked_routes": [
                {"route": list(route), "objective": _round64(score)} for route, score in self.ranked_routes
            ],
            "replay_identity": self.replay_identity,
        }


@dataclass(frozen=True)
class AutonomousPlanningPrekernelReport:
    version: str
    world_state_snapshot: WorldStateSnapshot
    planning_graph: PlanningGraph
    policy_search: PolicySearchResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "world_state_snapshot": self.world_state_snapshot.to_dict(),
            "planning_graph": self.planning_graph.to_dict(),
            "policy_search": self.policy_search.to_dict(),
            "analysis_only": True,
            "deterministic": True,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def snapshot_world_state(world_state: Mapping[str, Any]) -> WorldStateSnapshot:
    normalized = _normalize_json_value(world_state)
    if not isinstance(normalized, dict) or not normalized:
        raise ValueError("world_state must normalize to a non-empty mapping")
    canonical_json = _canonical_json(normalized)
    replay_identity = _sha256_hex_bytes(canonical_json.encode("utf-8"))
    return WorldStateSnapshot(normalized_state=normalized, canonical_json=canonical_json, replay_identity=replay_identity)


def synthesize_planning_graph(
    candidate_routes: Sequence[Sequence[str]],
    *,
    edge_costs: Mapping[tuple[str, str], float] | None = None,
) -> PlanningGraph:
    normalized_routes = _normalize_routes(candidate_routes)
    normalized_costs = _normalize_edge_costs(edge_costs)

    edges_map: dict[tuple[str, str], float] = {}
    nodes: set[str] = set()

    for route in normalized_routes:
        for node in route:
            nodes.add(node)
        for idx in range(len(route) - 1):
            src = route[idx]
            dst = route[idx + 1]
            pair = (src, dst)
            base_cost = normalized_costs.get(pair, 1.0)
            prior = edges_map.get(pair)
            if prior is None or base_cost < prior:
                edges_map[pair] = base_cost

    if not nodes:
        raise ValueError("planning graph must contain at least one node")

    out_degree: dict[str, int] = {node: 0 for node in nodes}
    in_degree: dict[str, int] = {node: 0 for node in nodes}
    for src, dst in edges_map:
        out_degree[src] += 1
        in_degree[dst] += 1

    ordered_nodes = tuple(
        PlanningNode(node_id=node, outgoing_degree=out_degree[node], incoming_degree=in_degree[node])
        for node in sorted(nodes)
    )
    ordered_edges = tuple(
        PlanningEdge(source=src, target=dst, cost=_round64(cost))
        for (src, dst), cost in sorted(edges_map.items(), key=lambda item: (item[0][0], item[0][1], item[1]))
    )

    graph_identity = _sha256_hex_mapping(
        {
            "version": AUTONOMOUS_PLANNING_PREKERNEL_VERSION,
            "nodes": [node.to_dict() for node in ordered_nodes],
            "edges": [edge.to_dict() for edge in ordered_edges],
        }
    )
    return PlanningGraph(nodes=ordered_nodes, edges=ordered_edges, graph_identity=graph_identity)


def bounded_route_objective(route: Sequence[str], planning_graph: PlanningGraph) -> float:
    normalized_route = tuple(_norm_token(step, name="route step") for step in route)
    if not normalized_route:
        raise ValueError("route must not be empty")

    node_set = {node.node_id for node in planning_graph.nodes}
    for node in normalized_route:
        if node not in node_set:
            raise ValueError(f"route node not present in planning graph: {node}")

    edge_cost_map = {(edge.source, edge.target): float(edge.cost) for edge in planning_graph.edges}
    cumulative_cost = 0.0
    for idx in range(len(normalized_route) - 1):
        pair = (normalized_route[idx], normalized_route[idx + 1])
        cumulative_cost += edge_cost_map.get(pair, _MISSING_EDGE_PENALTY)

    normalized_length = float(max(len(normalized_route) - 1, 0))
    bounded = 1.0 / (1.0 + cumulative_cost + normalized_length)
    return _round64(max(0.0, min(1.0, bounded)))


def deterministic_policy_search(
    planning_graph: PlanningGraph,
    candidate_routes: Sequence[Sequence[str]],
) -> PolicySearchResult:
    normalized_routes = _normalize_routes(candidate_routes)
    scored: list[tuple[tuple[str, ...], float]] = []
    for route in normalized_routes:
        score = bounded_route_objective(route, planning_graph)
        scored.append((route, score))

    ranked = tuple(sorted(scored, key=lambda item: (-item[1], len(item[0]), item[0])))
    best_route, best_score = ranked[0]

    replay_identity = _sha256_hex_mapping(
        {
            "graph_identity": planning_graph.graph_identity,
            "selected_route": list(best_route),
            "selected_objective": _round64(best_score),
            "ranked_routes": [{"route": list(route), "objective": _round64(score)} for route, score in ranked],
        }
    )
    return PolicySearchResult(
        selected_route=best_route,
        selected_objective=_round64(best_score),
        ranked_routes=ranked,
        replay_identity=replay_identity,
    )


def run_autonomous_planning_prekernel(
    world_state: Mapping[str, Any],
    candidate_routes: Sequence[Sequence[str]],
    *,
    edge_costs: Mapping[tuple[str, str], float] | None = None,
) -> AutonomousPlanningPrekernelReport:
    snapshot = snapshot_world_state(world_state)
    graph = synthesize_planning_graph(candidate_routes, edge_costs=edge_costs)
    policy = deterministic_policy_search(graph, candidate_routes)
    return AutonomousPlanningPrekernelReport(
        version=AUTONOMOUS_PLANNING_PREKERNEL_VERSION,
        world_state_snapshot=snapshot,
        planning_graph=graph,
        policy_search=policy,
    )


__all__ = [
    "AUTONOMOUS_PLANNING_PREKERNEL_VERSION",
    "AutonomousPlanningPrekernelReport",
    "PlanningEdge",
    "PlanningGraph",
    "PlanningNode",
    "PolicySearchResult",
    "WorldStateSnapshot",
    "bounded_route_objective",
    "deterministic_policy_search",
    "run_autonomous_planning_prekernel",
    "snapshot_world_state",
    "synthesize_planning_graph",
]
