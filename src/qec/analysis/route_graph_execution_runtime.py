from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

ROUTE_GRAPH_EXECUTION_RUNTIME_VERSION: str = "v137.6.1"
GENESIS_HASH: str = "0" * 64
_PRECISION: int = 12


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


def _validate_hash(token: str, *, name: str) -> str:
    value = _norm_token(token, name=name)
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


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


def _validate_max_path_length(max_path_length: int) -> int:
    value = int(max_path_length)
    if value < 1:
        raise ValueError("max_path_length must be >= 1")
    return value


def _normalize_route_graph(route_graph: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    if not route_graph:
        raise ValueError("route_graph must not be empty")

    normalized: dict[str, tuple[str, ...]] = {}
    all_nodes: set[str] = set()
    for raw_node, raw_neighbors in route_graph.items():
        node = _norm_token(str(raw_node), name="route node")
        if node in normalized:
            raise ValueError(f"route node key collision after normalization: {node!r}")
        if isinstance(raw_neighbors, (str, bytes, bytearray)):
            raise ValueError(
                f"route_graph[{node}] neighbors must not be a string or bytes sequence"
            )
        if not isinstance(raw_neighbors, Sequence):
            raise ValueError(f"route_graph[{node}] must be a sequence of next nodes")

        next_nodes: list[str] = []
        for raw_next in raw_neighbors:
            next_nodes.append(_norm_token(str(raw_next), name="next route node"))

        next_nodes_sorted = tuple(sorted(next_nodes))
        if len(next_nodes_sorted) != len(set(next_nodes_sorted)):
            raise ValueError(f"route_graph[{node}] contains duplicate branch targets")

        normalized[node] = next_nodes_sorted
        all_nodes.add(node)
        for branch in next_nodes_sorted:
            all_nodes.add(branch)

    if not all_nodes:
        raise ValueError("route_graph must define at least one node")

    for node in sorted(all_nodes):
        normalized.setdefault(node, tuple())

    return normalized


def _ensure_known_initial_node(initial_node: str, route_graph: Mapping[str, tuple[str, ...]]) -> str:
    node = _norm_token(initial_node, name="initial_node")
    if node not in route_graph:
        raise ValueError("initial_node must exist in route_graph node universe")
    return node


def _compute_final_world_state_hash(world_state: Mapping[str, Any], path: tuple[str, ...]) -> str:
    return _sha256_hex_mapping(
        {
            "world_state": world_state,
            "terminal_node": path[-1],
            "executed_route": list(path),
        }
    )


@dataclass(frozen=True)
class RouteExecution:
    schema_version: str
    source_plan_hash: str
    executed_route: tuple[str, ...]
    executed_route_hash: str
    path_length: int
    execution_stability_score: float
    final_world_state_hash: str
    stable_execution_hash: str
    execution_identity_chain: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_plan_hash": self.source_plan_hash,
            "executed_route": list(self.executed_route),
            "executed_route_hash": self.executed_route_hash,
            "path_length": int(self.path_length),
            "execution_stability_score": _round64(self.execution_stability_score),
            "final_world_state_hash": self.final_world_state_hash,
            "stable_execution_hash": self.stable_execution_hash,
            "execution_identity_chain": list(self.execution_identity_chain),
            "deterministic": True,
            "replay_safe": True,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class ExecutionReceipt:
    schema_version: str
    source_plan_hash: str
    executed_route_hash: str
    path_length: int
    execution_stability_score: float
    final_world_state_hash: str
    stable_execution_hash: str
    receipt_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_plan_hash": self.source_plan_hash,
            "executed_route_hash": self.executed_route_hash,
            "path_length": int(self.path_length),
            "execution_stability_score": _round64(self.execution_stability_score),
            "final_world_state_hash": self.final_world_state_hash,
            "stable_execution_hash": self.stable_execution_hash,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def _advance_path_normalized(
    current_path: tuple[str, ...],
    normalized_graph: dict[str, tuple[str, ...]],
    bounded_max_path_length: int,
) -> tuple[str, ...]:
    """Advance path by one step using a pre-normalized graph (no re-normalization).

    Preconditions:
    - ``current_path`` must contain normalized (stripped, non-empty) node tokens.
    - ``normalized_graph`` must be a fully normalized graph (keys and neighbor tuples
      already produced by ``_normalize_route_graph``).
    - ``bounded_max_path_length`` must be >= 1 (already validated).
    """
    if len(current_path) >= bounded_max_path_length:
        return current_path
    current_node = current_path[-1]
    candidates = normalized_graph.get(current_node, ())
    if not candidates:
        return current_path
    return current_path + (candidates[0],)


def advance_path_state(
    current_path: Sequence[str], route_graph: Mapping[str, Sequence[str]], *, max_path_length: int
) -> tuple[str, ...]:
    if not current_path:
        raise ValueError("current_path must not be empty")

    bounded_max_path_length = _validate_max_path_length(max_path_length)
    normalized_path = tuple(_norm_token(node, name="path node") for node in current_path)
    normalized_graph = _normalize_route_graph(route_graph)

    if normalized_path[-1] not in normalized_graph:
        raise ValueError("current_path terminal node must exist in route_graph")

    return _advance_path_normalized(normalized_path, normalized_graph, bounded_max_path_length)


def compute_execution_stability_score(branch_factors: Sequence[int]) -> float:
    if not branch_factors:
        return 1.0

    reciprocal_sum = 0.0
    for factor in branch_factors:
        value = int(factor)
        if value < 1:
            raise ValueError("branch_factors must be >= 1")
        reciprocal_sum += 1.0 / float(value)

    return _round64(reciprocal_sum / float(len(branch_factors)))


def execute_route_graph(
    source_plan_hash: str,
    route_graph: Mapping[str, Sequence[str]],
    *,
    initial_node: str,
    world_state: Mapping[str, Any],
    max_path_length: int,
    enable_v137_6_route_runtime: bool = False,
) -> RouteExecution:
    if not enable_v137_6_route_runtime:
        raise ValueError("enable_v137_6_route_runtime must be True to enable v137.6.1 route runtime")

    normalized_source_plan_hash = _validate_hash(source_plan_hash, name="source_plan_hash")
    normalized_route_graph = _normalize_route_graph(route_graph)
    normalized_initial_node = _ensure_known_initial_node(initial_node, normalized_route_graph)

    normalized_world_state = _normalize_json_value(world_state, label="world_state")
    if not isinstance(normalized_world_state, dict) or not normalized_world_state:
        raise ValueError("world_state must normalize to a non-empty mapping")

    bounded_max_path_length = _validate_max_path_length(max_path_length)

    path: tuple[str, ...] = (normalized_initial_node,)
    branch_factors: list[int] = []
    identity_chain: list[str] = [GENESIS_HASH]

    while True:
        if len(path) >= bounded_max_path_length:
            break

        terminal_node = path[-1]
        candidates = normalized_route_graph[terminal_node]
        if not candidates:
            break

        branch_factors.append(len(candidates))
        next_path = _advance_path_normalized(path, normalized_route_graph, bounded_max_path_length)
        if next_path == path:
            break

        selected_node = next_path[-1]
        step_hash = _sha256_hex_mapping(
            {
                "prior_hash": identity_chain[-1],
                "step_index": len(next_path) - 1,
                "from_node": terminal_node,
                "candidate_nodes": list(candidates),
                "selected_node": selected_node,
                "source_plan_hash": normalized_source_plan_hash,
            }
        )
        identity_chain.append(step_hash)
        path = next_path

    execution_stability_score = compute_execution_stability_score(branch_factors)
    executed_route_hash = _sha256_hex_mapping({"executed_route": list(path)})
    final_world_state_hash = _compute_final_world_state_hash(normalized_world_state, path)

    execution_payload = {
        "schema_version": ROUTE_GRAPH_EXECUTION_RUNTIME_VERSION,
        "source_plan_hash": normalized_source_plan_hash,
        "executed_route": list(path),
        "executed_route_hash": executed_route_hash,
        "path_length": len(path),
        "execution_stability_score": execution_stability_score,
        "final_world_state_hash": final_world_state_hash,
        "execution_identity_chain": list(identity_chain),
    }
    stable_execution_hash = _sha256_hex_mapping(execution_payload)

    return RouteExecution(
        schema_version=ROUTE_GRAPH_EXECUTION_RUNTIME_VERSION,
        source_plan_hash=normalized_source_plan_hash,
        executed_route=path,
        executed_route_hash=executed_route_hash,
        path_length=len(path),
        execution_stability_score=execution_stability_score,
        final_world_state_hash=final_world_state_hash,
        stable_execution_hash=stable_execution_hash,
        execution_identity_chain=tuple(identity_chain),
    )


def export_execution_bytes(execution: RouteExecution) -> bytes:
    if not isinstance(execution, RouteExecution):
        raise TypeError("execution must be a RouteExecution instance")
    return execution.to_canonical_bytes()


def generate_execution_receipt(execution: RouteExecution) -> ExecutionReceipt:
    if not isinstance(execution, RouteExecution):
        raise TypeError("execution must be a RouteExecution instance")

    receipt_payload = {
        "schema_version": execution.schema_version,
        "source_plan_hash": execution.source_plan_hash,
        "executed_route_hash": execution.executed_route_hash,
        "path_length": int(execution.path_length),
        "execution_stability_score": _round64(execution.execution_stability_score),
        "final_world_state_hash": execution.final_world_state_hash,
        "stable_execution_hash": execution.stable_execution_hash,
    }
    receipt_hash = _sha256_hex_mapping(receipt_payload)

    return ExecutionReceipt(
        schema_version=execution.schema_version,
        source_plan_hash=execution.source_plan_hash,
        executed_route_hash=execution.executed_route_hash,
        path_length=int(execution.path_length),
        execution_stability_score=execution.execution_stability_score,
        final_world_state_hash=execution.final_world_state_hash,
        stable_execution_hash=execution.stable_execution_hash,
        receipt_hash=receipt_hash,
    )


__all__ = [
    "ROUTE_GRAPH_EXECUTION_RUNTIME_VERSION",
    "GENESIS_HASH",
    "RouteExecution",
    "ExecutionReceipt",
    "advance_path_state",
    "compute_execution_stability_score",
    "execute_route_graph",
    "export_execution_bytes",
    "generate_execution_receipt",
]
