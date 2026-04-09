from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

AUTONOMOUS_PLANNING_KERNEL_VERSION: str = "v137.2.0"
GENESIS_HASH: str = "0" * 64
_PRECISION: int = 12
_MISSING_EDGE_PENALTY: float = 1.0e6


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


def _normalize_json_value(value: Any, *, label: str) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{label} values must be finite")
        return _round64(numeric)
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        normalized_items: list[tuple[str, Any]] = []
        for key, item in value.items():
            norm_key = _norm_token(str(key), name=f"{label} key")
            normalized_items.append((norm_key, _normalize_json_value(item, label=label)))
        normalized_items.sort(key=lambda item: item[0])
        normalized: dict[str, Any] = {}
        for key, item in normalized_items:
            if key in normalized:
                raise ValueError(f"duplicate {label} key after normalization: {key}")
            normalized[key] = item
        return normalized
    if isinstance(value, (tuple, list)):
        return [_normalize_json_value(item, label=label) for item in value]
    raise ValueError(f"unsupported {label} value type: {type(value).__name__}")


def _normalize_routes(candidate_routes: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    if not candidate_routes:
        raise ValueError("candidate_routes must not be empty")
    normalized: list[tuple[str, ...]] = []
    for idx, route in enumerate(candidate_routes):
        if not route:
            raise ValueError(f"candidate route at index {idx} must not be empty")
        normalized_route = tuple(_norm_token(step, name="route step") for step in route)
        normalized.append(normalized_route)
    return tuple(normalized)


def _normalize_edge_costs(edge_costs: Mapping[tuple[str, str], float] | None) -> dict[tuple[str, str], float]:
    if edge_costs is None:
        return {}
    normalized: dict[tuple[str, str], float] = {}
    for key, cost in edge_costs.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("edge_costs keys must be (source, target) tuples")
        src = _norm_token(key[0], name="edge source")
        dst = _norm_token(key[1], name="edge target")
        norm_key = (src, dst)
        numeric = float(cost)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("edge costs must be finite and >= 0")
        if norm_key in normalized:
            raise ValueError(f"duplicate edge_costs key after normalization: {norm_key}")
        normalized[norm_key] = _round64(numeric)
    return normalized


def _normalize_world_state_bounds(
    bounds: Mapping[str, tuple[float, float]] | None,
) -> dict[str, tuple[float, float]]:
    if bounds is None:
        return {}
    normalized: dict[str, tuple[float, float]] = {}
    for key, interval in bounds.items():
        name = _norm_token(key, name="world_state_bound key")
        if not isinstance(interval, tuple) or len(interval) != 2:
            raise ValueError("world_state_bounds values must be (min, max) tuples")
        lo = float(interval[0])
        hi = float(interval[1])
        if not math.isfinite(lo) or not math.isfinite(hi):
            raise ValueError("world_state_bounds values must be finite")
        if lo > hi:
            raise ValueError("world_state_bounds min must be <= max")
        if name in normalized:
            raise ValueError(f"duplicate world_state_bound key after normalization: {name}")
        normalized[name] = (_round64(lo), _round64(hi))
    return normalized


def _extract_numeric_world_state(world_state: Mapping[str, Any]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, value in world_state.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            numeric[key] = float(value)
            continue
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("world_state numeric values must be finite")
            numeric[key] = float(value)
    return numeric


@dataclass(frozen=True)
class WorldStateBound:
    key: str
    lower: float
    upper: float
    bounded_value: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "lower": _round64(self.lower),
            "upper": _round64(self.upper),
            "bounded_value": _round64(self.bounded_value),
            "status": self.status,
        }


@dataclass(frozen=True)
class BoundedWorldState:
    normalized_state_canonical_json: str
    bounds: tuple[WorldStateBound, ...]
    world_state_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_state": json.loads(self.normalized_state_canonical_json),
            "bounds": [item.to_dict() for item in self.bounds],
            "world_state_hash": self.world_state_hash,
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
    graph_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "graph_hash": self.graph_hash,
        }


@dataclass(frozen=True)
class PolicyRouteScore:
    route: tuple[str, ...]
    objective: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": list(self.route),
            "objective": _round64(self.objective),
        }


@dataclass(frozen=True)
class ControlSynthesis:
    selected_route: tuple[str, ...]
    ranked_routes: tuple[PolicyRouteScore, ...]
    synthesis_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_route": list(self.selected_route),
            "ranked_routes": [item.to_dict() for item in self.ranked_routes],
            "synthesis_hash": self.synthesis_hash,
        }


@dataclass(frozen=True)
class ExecutionStep:
    step_index: int
    source: str
    target: str
    action: str
    edge_cost: float
    prior_hash: str
    step_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "source": self.source,
            "target": self.target,
            "action": self.action,
            "edge_cost": _round64(self.edge_cost),
            "prior_hash": self.prior_hash,
            "step_hash": self.step_hash,
        }


@dataclass(frozen=True)
class ExecutionArtifact:
    route: tuple[str, ...]
    steps: tuple[ExecutionStep, ...]
    execution_head_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": list(self.route),
            "steps": [step.to_dict() for step in self.steps],
            "execution_head_hash": self.execution_head_hash,
        }


@dataclass(frozen=True)
class PolicyLedgerEntry:
    sequence: int
    node_id: str
    policy_action: str
    bounded_status: str
    prior_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "node_id": self.node_id,
            "policy_action": self.policy_action,
            "bounded_status": self.bounded_status,
            "prior_hash": self.prior_hash,
            "entry_hash": self.entry_hash,
        }


@dataclass(frozen=True)
class PolicyLedger:
    entries: tuple[PolicyLedgerEntry, ...]
    head_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "head_hash": self.head_hash,
        }


@dataclass(frozen=True)
class AutonomousPlanningKernelReport:
    version: str
    bounded_world_state: BoundedWorldState
    planning_graph: PlanningGraph
    control_synthesis: ControlSynthesis
    execution_artifact: ExecutionArtifact
    policy_ledger: PolicyLedger
    plan_identity_chain: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "bounded_world_state": self.bounded_world_state.to_dict(),
            "planning_graph": self.planning_graph.to_dict(),
            "control_synthesis": self.control_synthesis.to_dict(),
            "execution_artifact": self.execution_artifact.to_dict(),
            "policy_ledger": self.policy_ledger.to_dict(),
            "plan_identity_chain": list(self.plan_identity_chain),
            "runtime_mode": "deterministic",
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def bound_world_state(
    world_state: Mapping[str, Any],
    *,
    world_state_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> BoundedWorldState:
    normalized = _normalize_json_value(world_state, label="world_state")
    if not isinstance(normalized, dict) or not normalized:
        raise ValueError("world_state must normalize to a non-empty mapping")

    numeric = _extract_numeric_world_state(normalized)
    bounds = _normalize_world_state_bounds(world_state_bounds)

    bound_records: list[WorldStateBound] = []
    for key, interval in sorted(bounds.items(), key=lambda item: item[0]):
        if key not in numeric:
            raise ValueError(f"world_state bound key missing numeric value: {key}")
        lo, hi = interval
        value = float(numeric[key])
        if value < lo:
            bounded = lo
            status = "clamped_low"
        elif value > hi:
            bounded = hi
            status = "clamped_high"
        else:
            bounded = value
            status = "within"
        bound_records.append(
            WorldStateBound(
                key=key,
                lower=lo,
                upper=hi,
                bounded_value=_round64(bounded),
                status=status,
            )
        )

    payload = {
        "normalized_state": normalized,
        "bounds": [record.to_dict() for record in bound_records],
        "version": AUTONOMOUS_PLANNING_KERNEL_VERSION,
    }
    canonical_json = _canonical_json(normalized)
    return BoundedWorldState(
        normalized_state_canonical_json=canonical_json,
        bounds=tuple(bound_records),
        world_state_hash=_sha256_hex_mapping(payload),
    )


def synthesize_planning_graph(
    candidate_routes: Sequence[Sequence[str]],
    *,
    edge_costs: Mapping[tuple[str, str], float] | None = None,
) -> PlanningGraph:
    routes = _normalize_routes(candidate_routes)
    costs = _normalize_edge_costs(edge_costs)

    nodes: set[str] = set()
    edges_map: dict[tuple[str, str], float] = {}

    for route in routes:
        for node in route:
            nodes.add(node)
        for idx in range(len(route) - 1):
            pair = (route[idx], route[idx + 1])
            base_cost = costs.get(pair, 1.0)
            prior = edges_map.get(pair)
            if prior is None or base_cost < prior:
                edges_map[pair] = base_cost

    if not nodes:
        raise ValueError("planning graph must contain at least one node")

    incoming: dict[str, int] = {node: 0 for node in nodes}
    outgoing: dict[str, int] = {node: 0 for node in nodes}
    for src, dst in edges_map:
        outgoing[src] += 1
        incoming[dst] += 1

    ordered_nodes = tuple(
        PlanningNode(node_id=node, outgoing_degree=outgoing[node], incoming_degree=incoming[node])
        for node in sorted(nodes)
    )
    ordered_edges = tuple(
        PlanningEdge(source=src, target=dst, cost=cost)
        for (src, dst), cost in sorted(edges_map.items(), key=lambda item: (item[0][0], item[0][1], item[1]))
    )

    graph_hash = _sha256_hex_mapping(
        {
            "version": AUTONOMOUS_PLANNING_KERNEL_VERSION,
            "nodes": [node.to_dict() for node in ordered_nodes],
            "edges": [edge.to_dict() for edge in ordered_edges],
        }
    )
    return PlanningGraph(nodes=ordered_nodes, edges=ordered_edges, graph_hash=graph_hash)


def _bounded_route_objective(route: Sequence[str], planning_graph: PlanningGraph) -> float:
    normalized_route = tuple(_norm_token(step, name="route step") for step in route)
    if not normalized_route:
        raise ValueError("route must not be empty")
    node_set = {node.node_id for node in planning_graph.nodes}
    for step in normalized_route:
        if step not in node_set:
            raise ValueError(f"route node not present in planning graph: {step}")

    cost_map = {(edge.source, edge.target): float(edge.cost) for edge in planning_graph.edges}
    cumulative_cost = 0.0
    for idx in range(len(normalized_route) - 1):
        cumulative_cost += cost_map.get((normalized_route[idx], normalized_route[idx + 1]), _MISSING_EDGE_PENALTY)
    length_penalty = float(max(len(normalized_route) - 1, 0))
    bounded = 1.0 / (1.0 + cumulative_cost + length_penalty)
    return _round64(max(0.0, min(1.0, bounded)))


def synthesize_control_policy(
    planning_graph: PlanningGraph,
    candidate_routes: Sequence[Sequence[str]],
) -> ControlSynthesis:
    routes = _normalize_routes(candidate_routes)
    ranked_raw: list[PolicyRouteScore] = []
    for route in routes:
        ranked_raw.append(PolicyRouteScore(route=route, objective=_bounded_route_objective(route, planning_graph)))

    ranked = tuple(sorted(ranked_raw, key=lambda item: (-item.objective, len(item.route), item.route)))
    selected = ranked[0]
    synthesis_hash = _sha256_hex_mapping(
        {
            "graph_hash": planning_graph.graph_hash,
            "selected_route": list(selected.route),
            "ranked_routes": [entry.to_dict() for entry in ranked],
        }
    )
    return ControlSynthesis(
        selected_route=selected.route,
        ranked_routes=ranked,
        synthesis_hash=synthesis_hash,
    )


def execute_deterministic_route(
    planning_graph: PlanningGraph,
    selected_route: Sequence[str],
) -> ExecutionArtifact:
    route = tuple(_norm_token(step, name="route step") for step in selected_route)
    if not route:
        raise ValueError("selected_route must not be empty")

    node_set = {node.node_id for node in planning_graph.nodes}
    for step in route:
        if step not in node_set:
            raise ValueError(f"route node not present in planning graph: {step}")

    edge_cost_map = {(edge.source, edge.target): float(edge.cost) for edge in planning_graph.edges}
    for idx in range(len(route) - 1):
        pair = (route[idx], route[idx + 1])
        if pair not in edge_cost_map:
            raise ValueError(f"route edge not present in planning graph: {pair}")
    steps: list[ExecutionStep] = []
    prior_hash = GENESIS_HASH
    for idx in range(len(route) - 1):
        source = route[idx]
        target = route[idx + 1]
        edge_cost = edge_cost_map.get((source, target), _MISSING_EDGE_PENALTY)
        action = f"TRANSITION_{source}_TO_{target}"
        step_payload = {
            "step_index": idx,
            "source": source,
            "target": target,
            "action": action,
            "edge_cost": _round64(edge_cost),
            "prior_hash": prior_hash,
        }
        step_hash = _sha256_hex_mapping(step_payload)
        steps.append(
            ExecutionStep(
                step_index=idx,
                source=source,
                target=target,
                action=action,
                edge_cost=edge_cost,
                prior_hash=prior_hash,
                step_hash=step_hash,
            )
        )
        prior_hash = step_hash

    return ExecutionArtifact(route=route, steps=tuple(steps), execution_head_hash=prior_hash)


def build_policy_ledger(
    planning_graph: PlanningGraph,
    bounded_world_state: BoundedWorldState,
) -> PolicyLedger:
    status_by_key = {bound.key: bound.status for bound in bounded_world_state.bounds}
    nodes = tuple(node.node_id for node in planning_graph.nodes)

    entries: list[PolicyLedgerEntry] = []
    prior_hash = GENESIS_HASH
    for seq, node_id in enumerate(nodes):
        status = status_by_key.get(node_id, "unbounded")
        action = "HOLD" if status == "within" else "STABILIZE"
        payload = {
            "sequence": seq,
            "node_id": node_id,
            "policy_action": action,
            "bounded_status": status,
            "prior_hash": prior_hash,
        }
        entry_hash = _sha256_hex_mapping(payload)
        entries.append(
            PolicyLedgerEntry(
                sequence=seq,
                node_id=node_id,
                policy_action=action,
                bounded_status=status,
                prior_hash=prior_hash,
                entry_hash=entry_hash,
            )
        )
        prior_hash = entry_hash

    return PolicyLedger(entries=tuple(entries), head_hash=prior_hash)


def _validate_plan_identity_chain(chain: tuple[str, ...]) -> None:
    if len(chain) != 6:
        raise ValueError("plan_identity_chain must include exactly 6 identities")
    for idx, digest in enumerate(chain):
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"plan_identity_chain entry {idx} must be a 64-char SHA-256 hex string")
        int(digest, 16)


def run_autonomous_planning_kernel(
    world_state: Mapping[str, Any],
    candidate_routes: Sequence[Sequence[str]],
    *,
    edge_costs: Mapping[tuple[str, str], float] | None = None,
    world_state_bounds: Mapping[str, tuple[float, float]] | None = None,
    enable_v137_2_runtime: bool = False,
) -> AutonomousPlanningKernelReport:
    if not enable_v137_2_runtime:
        raise ValueError("enable_v137_2_runtime must be True for v137.2.0 runtime execution")

    bounded_state = bound_world_state(world_state, world_state_bounds=world_state_bounds)
    graph = synthesize_planning_graph(candidate_routes, edge_costs=edge_costs)
    synthesis = synthesize_control_policy(graph, candidate_routes)
    execution = execute_deterministic_route(graph, synthesis.selected_route)
    ledger = build_policy_ledger(graph, bounded_state)

    chain = (
        bounded_state.world_state_hash,
        graph.graph_hash,
        synthesis.synthesis_hash,
        execution.execution_head_hash,
        ledger.head_hash,
        _sha256_hex_mapping(
            {
                "version": AUTONOMOUS_PLANNING_KERNEL_VERSION,
                "world_state_hash": bounded_state.world_state_hash,
                "graph_hash": graph.graph_hash,
                "synthesis_hash": synthesis.synthesis_hash,
                "execution_head_hash": execution.execution_head_hash,
                "ledger_head_hash": ledger.head_hash,
            }
        ),
    )
    _validate_plan_identity_chain(chain)

    return AutonomousPlanningKernelReport(
        version=AUTONOMOUS_PLANNING_KERNEL_VERSION,
        bounded_world_state=bounded_state,
        planning_graph=graph,
        control_synthesis=synthesis,
        execution_artifact=execution,
        policy_ledger=ledger,
        plan_identity_chain=chain,
    )


def export_policy_ledger_canonical_bytes(ledger: PolicyLedger) -> bytes:
    if not isinstance(ledger, PolicyLedger):
        raise TypeError("ledger must be a PolicyLedger")
    return _canonical_json_bytes(ledger.to_dict())


__all__ = [
    "AUTONOMOUS_PLANNING_KERNEL_VERSION",
    "AutonomousPlanningKernelReport",
    "BoundedWorldState",
    "ControlSynthesis",
    "ExecutionArtifact",
    "ExecutionStep",
    "GENESIS_HASH",
    "PlanningEdge",
    "PlanningGraph",
    "PlanningNode",
    "PolicyLedger",
    "PolicyLedgerEntry",
    "PolicyRouteScore",
    "WorldStateBound",
    "bound_world_state",
    "build_policy_ledger",
    "execute_deterministic_route",
    "export_policy_ledger_canonical_bytes",
    "run_autonomous_planning_kernel",
    "synthesize_control_policy",
    "synthesize_planning_graph",
]
