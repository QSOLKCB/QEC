from __future__ import annotations

"""v137.6.2 deterministic dead-end pruning for route runtime.

Release laws:
- DEAD_END_PRUNING_LAW:
  A candidate is a dead end iff it cannot reach any valid terminal node (a node
  with zero outgoing continuations) within the bounded runtime horizon.
- DETERMINISTIC_ROUTE_ELIMINATION_RULE:
  Candidate examination, elimination, and survivor ordering are stable under
  identical input (lexicographic ordering + canonical hashing).
- BOUNDED_PRUNING_PRESSURE_INVARIANT:
  dead_end_pressure_score is always in [0.0, 1.0].
- REPLAY_SAFE_PRUNING_CHAIN:
  Pruning artifacts export canonical JSON/bytes and include stable SHA-256
  identity chain fields for replay-safe verification.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

ROUTE_DEAD_END_PRUNING_VERSION: str = "v137.6.2"
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
            raise ValueError(f"route_graph[{node}] neighbors must not be a string or bytes sequence")
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

    for node in sorted(all_nodes):
        normalized.setdefault(node, tuple())

    return normalized


def _normalize_path(current_path: Sequence[str]) -> tuple[str, ...]:
    if not current_path:
        raise ValueError("current_path must not be empty")
    return tuple(_norm_token(node, name="path node") for node in current_path)


def _normalize_frontier(frontier_candidates: Sequence[str]) -> tuple[str, ...]:
    if isinstance(frontier_candidates, (str, bytes, bytearray)):
        raise ValueError("frontier_candidates must be a sequence of node tokens")
    canonical = tuple(sorted(_norm_token(node, name="frontier candidate") for node in frontier_candidates))
    if len(canonical) != len(set(canonical)):
        raise ValueError("frontier_candidates must not contain duplicates")
    return canonical


def _can_reach_terminal(
    start_node: str,
    normalized_graph: Mapping[str, tuple[str, ...]],
    *,
    max_remaining_steps: int,
) -> bool:
    if max_remaining_steps < 0:
        return False

    stack: list[tuple[str, int]] = [(start_node, max_remaining_steps)]
    visited: set[tuple[str, int]] = set()

    while stack:
        node, steps_left = stack.pop()
        state = (node, steps_left)
        if state in visited:
            continue
        visited.add(state)

        branches = normalized_graph.get(node, tuple())
        if not branches:
            return True
        if steps_left == 0:
            continue
        for nxt in reversed(branches):
            stack.append((nxt, steps_left - 1))

    return False


def compute_dead_end_pressure(total_candidates_examined: int, dead_end_count: int) -> float:
    examined = int(total_candidates_examined)
    dead_ends = int(dead_end_count)
    if examined < 0:
        raise ValueError("total_candidates_examined must be >= 0")
    if dead_ends < 0:
        raise ValueError("dead_end_count must be >= 0")
    if dead_ends > examined:
        raise ValueError("dead_end_count must be <= total_candidates_examined")
    if examined == 0:
        return 0.0
    return _round64(float(dead_ends) / float(examined))


@dataclass(frozen=True)
class DeadEndPruningArtifact:
    schema_version: str
    source_plan_hash: str
    current_path: tuple[str, ...]
    route_graph_hash: str
    examined_candidates: tuple[str, ...]
    dead_end_candidates: tuple[str, ...]
    survivor_candidates: tuple[str, ...]
    reachable_terminal_nodes: tuple[str, ...]
    dead_end_pressure_score: float
    pruning_identity_chain: tuple[str, ...]
    stable_pruning_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_plan_hash": self.source_plan_hash,
            "current_path": list(self.current_path),
            "route_graph_hash": self.route_graph_hash,
            "examined_candidates": list(self.examined_candidates),
            "dead_end_candidates": list(self.dead_end_candidates),
            "survivor_candidates": list(self.survivor_candidates),
            "reachable_terminal_nodes": list(self.reachable_terminal_nodes),
            "dead_end_pressure_score": _round64(self.dead_end_pressure_score),
            "pruning_identity_chain": list(self.pruning_identity_chain),
            "stable_pruning_hash": self.stable_pruning_hash,
            "deterministic": True,
            "replay_safe": True,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class DeadEndPruningReceipt:
    schema_version: str
    source_plan_hash: str
    route_graph_hash: str
    total_candidates_examined: int
    total_dead_ends_found: int
    total_candidates_pruned: int
    survivors_count: int
    dead_end_pressure_score: float
    stable_pruning_hash: str
    receipt_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_plan_hash": self.source_plan_hash,
            "route_graph_hash": self.route_graph_hash,
            "total_candidates_examined": int(self.total_candidates_examined),
            "total_dead_ends_found": int(self.total_dead_ends_found),
            "total_candidates_pruned": int(self.total_candidates_pruned),
            "survivors_count": int(self.survivors_count),
            "dead_end_pressure_score": _round64(self.dead_end_pressure_score),
            "stable_pruning_hash": self.stable_pruning_hash,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def analyze_dead_end_pruning(
    source_plan_hash: str,
    route_graph: Mapping[str, Sequence[str]],
    *,
    current_path: Sequence[str],
    max_path_length: int,
    enable_v137_6_2_dead_end_pruning: bool = False,
) -> DeadEndPruningArtifact:
    if not enable_v137_6_2_dead_end_pruning:
        raise ValueError("enable_v137_6_2_dead_end_pruning must be True to enable v137.6.2 dead-end pruning")

    normalized_source_plan_hash = _validate_hash(source_plan_hash, name="source_plan_hash")
    normalized_graph = _normalize_route_graph(route_graph)
    normalized_path = _normalize_path(current_path)
    if normalized_path[-1] not in normalized_graph:
        raise ValueError("current_path terminal node must exist in route_graph")

    bounded_max_path_length = _validate_max_path_length(max_path_length)
    if len(normalized_path) > bounded_max_path_length:
        raise ValueError("current_path length must be <= max_path_length")

    frontier = _normalize_frontier(normalized_graph[normalized_path[-1]])
    route_graph_hash = _sha256_hex_mapping({"route_graph": {k: list(v) for k, v in sorted(normalized_graph.items())}})

    remaining_after_step = bounded_max_path_length - (len(normalized_path) + 1)
    identity_chain: list[str] = [GENESIS_HASH]
    dead_ends: list[str] = []
    survivors: list[str] = []
    reachable_terminals: set[str] = set()

    for idx, candidate in enumerate(frontier):
        can_reach_terminal = _can_reach_terminal(candidate, normalized_graph, max_remaining_steps=remaining_after_step)
        if can_reach_terminal:
            survivors.append(candidate)
        else:
            dead_ends.append(candidate)

        terminal_marker = candidate if not normalized_graph[candidate] else ""
        if terminal_marker:
            reachable_terminals.add(terminal_marker)

        step_hash = _sha256_hex_mapping(
            {
                "prior_hash": identity_chain[-1],
                "candidate_index": idx,
                "candidate_node": candidate,
                "is_dead_end": not can_reach_terminal,
                "remaining_steps_after_candidate": int(remaining_after_step),
                "source_plan_hash": normalized_source_plan_hash,
                "route_graph_hash": route_graph_hash,
            }
        )
        identity_chain.append(step_hash)

    pressure = compute_dead_end_pressure(len(frontier), len(dead_ends))
    reachable_terminal_nodes = tuple(sorted(reachable_terminals))

    pruning_payload = {
        "schema_version": ROUTE_DEAD_END_PRUNING_VERSION,
        "source_plan_hash": normalized_source_plan_hash,
        "current_path": list(normalized_path),
        "route_graph_hash": route_graph_hash,
        "examined_candidates": list(frontier),
        "dead_end_candidates": list(dead_ends),
        "survivor_candidates": list(survivors),
        "reachable_terminal_nodes": list(reachable_terminal_nodes),
        "dead_end_pressure_score": pressure,
        "pruning_identity_chain": list(identity_chain),
    }
    stable_pruning_hash = _sha256_hex_mapping(pruning_payload)

    return DeadEndPruningArtifact(
        schema_version=ROUTE_DEAD_END_PRUNING_VERSION,
        source_plan_hash=normalized_source_plan_hash,
        current_path=normalized_path,
        route_graph_hash=route_graph_hash,
        examined_candidates=frontier,
        dead_end_candidates=tuple(dead_ends),
        survivor_candidates=tuple(survivors),
        reachable_terminal_nodes=reachable_terminal_nodes,
        dead_end_pressure_score=pressure,
        pruning_identity_chain=tuple(identity_chain),
        stable_pruning_hash=stable_pruning_hash,
    )


def prune_route_frontier(
    source_plan_hash: str,
    route_graph: Mapping[str, Sequence[str]],
    *,
    current_path: Sequence[str],
    max_path_length: int,
    enable_v137_6_2_dead_end_pruning: bool = False,
) -> tuple[tuple[str, ...], DeadEndPruningArtifact]:
    artifact = analyze_dead_end_pruning(
        source_plan_hash,
        route_graph,
        current_path=current_path,
        max_path_length=max_path_length,
        enable_v137_6_2_dead_end_pruning=enable_v137_6_2_dead_end_pruning,
    )
    return artifact.survivor_candidates, artifact


def generate_dead_end_pruning_receipt(artifact: DeadEndPruningArtifact) -> DeadEndPruningReceipt:
    if not isinstance(artifact, DeadEndPruningArtifact):
        raise TypeError("artifact must be a DeadEndPruningArtifact instance")

    receipt_payload = {
        "schema_version": artifact.schema_version,
        "source_plan_hash": artifact.source_plan_hash,
        "route_graph_hash": artifact.route_graph_hash,
        "total_candidates_examined": len(artifact.examined_candidates),
        "total_dead_ends_found": len(artifact.dead_end_candidates),
        "total_candidates_pruned": len(artifact.dead_end_candidates),
        "survivors_count": len(artifact.survivor_candidates),
        "dead_end_pressure_score": _round64(artifact.dead_end_pressure_score),
        "stable_pruning_hash": artifact.stable_pruning_hash,
    }
    receipt_hash = _sha256_hex_mapping(receipt_payload)

    return DeadEndPruningReceipt(
        schema_version=artifact.schema_version,
        source_plan_hash=artifact.source_plan_hash,
        route_graph_hash=artifact.route_graph_hash,
        total_candidates_examined=len(artifact.examined_candidates),
        total_dead_ends_found=len(artifact.dead_end_candidates),
        total_candidates_pruned=len(artifact.dead_end_candidates),
        survivors_count=len(artifact.survivor_candidates),
        dead_end_pressure_score=artifact.dead_end_pressure_score,
        stable_pruning_hash=artifact.stable_pruning_hash,
        receipt_hash=receipt_hash,
    )


def export_dead_end_pruning_bytes(artifact: DeadEndPruningArtifact) -> bytes:
    if not isinstance(artifact, DeadEndPruningArtifact):
        raise TypeError("artifact must be a DeadEndPruningArtifact instance")
    return artifact.to_canonical_bytes()


__all__ = [
    "ROUTE_DEAD_END_PRUNING_VERSION",
    "GENESIS_HASH",
    "DeadEndPruningArtifact",
    "DeadEndPruningReceipt",
    "analyze_dead_end_pruning",
    "prune_route_frontier",
    "compute_dead_end_pressure",
    "generate_dead_end_pruning_receipt",
    "export_dead_end_pruning_bytes",
]
