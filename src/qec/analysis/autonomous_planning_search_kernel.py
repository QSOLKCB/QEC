from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

AUTONOMOUS_PLANNING_SEARCH_KERNEL_VERSION: str = "v137.6.0"
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
    for index, route in enumerate(candidate_routes):
        if not route:
            raise ValueError(f"candidate route at index {index} must not be empty")
        normalized.append(tuple(_norm_token(step, name="route step") for step in route))
    return tuple(normalized)


def _validate_search_depth(search_depth: int) -> int:
    depth = int(search_depth)
    if depth < 1:
        raise ValueError("search_depth must be >= 1")
    return depth


def _route_depth(route: tuple[str, ...]) -> int:
    return max(0, len(route) - 1)


def _route_score(route: tuple[str, ...], preferred_nodes: tuple[str, ...]) -> float:
    if not preferred_nodes:
        return _round64(1.0 / float(len(route)))
    route_node_set = set(route)
    coverage = 0
    for node in preferred_nodes:
        if node in route_node_set:
            coverage += 1
    return _round64(float(coverage) / float(len(preferred_nodes)))


def _compute_stable_plan_hash(payload: Mapping[str, Any]) -> str:
    return _sha256_hex_mapping(payload)


@dataclass(frozen=True)
class PlanIR:
    schema_version: str
    world_state: Mapping[str, Any]
    objective: Mapping[str, Any]
    candidate_routes: tuple[tuple[str, ...], ...]
    world_state_hash: str
    objective_hash: str
    search_depth: int
    search_stability_score: float
    selected_route: tuple[str, ...]
    selected_route_hash: str
    stable_plan_hash: str
    search_identity_chain: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "world_state": self.world_state,
            "objective": self.objective,
            "candidate_routes": [list(route) for route in self.candidate_routes],
            "world_state_hash": self.world_state_hash,
            "objective_hash": self.objective_hash,
            "search_depth": int(self.search_depth),
            "search_stability_score": _round64(self.search_stability_score),
            "selected_route": list(self.selected_route),
            "selected_route_hash": self.selected_route_hash,
            "stable_plan_hash": self.stable_plan_hash,
            "search_identity_chain": list(self.search_identity_chain),
            "deterministic": True,
            "replay_safe": True,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


@dataclass(frozen=True)
class PlanReceipt:
    schema_version: str
    world_state_hash: str
    objective_hash: str
    search_depth: int
    search_stability_score: float
    selected_route_hash: str
    stable_plan_hash: str
    receipt_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "world_state_hash": self.world_state_hash,
            "objective_hash": self.objective_hash,
            "search_depth": int(self.search_depth),
            "search_stability_score": _round64(self.search_stability_score),
            "selected_route_hash": self.selected_route_hash,
            "stable_plan_hash": self.stable_plan_hash,
            "receipt_hash": self.receipt_hash,
        }

    def to_canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def to_canonical_bytes(self) -> bytes:
        return self.to_canonical_json().encode("utf-8")


def run_deterministic_search(
    candidate_routes: Sequence[Sequence[str]],
    *,
    search_depth: int,
    objective: Mapping[str, Any],
    world_state_hash: str,
    objective_hash: str,
) -> tuple[tuple[str, ...], float, tuple[str, ...]]:
    depth_limit = _validate_search_depth(search_depth)
    normalized_routes = _normalize_routes(candidate_routes)

    preferred_raw = objective.get("preferred_nodes", [])
    if not isinstance(preferred_raw, list):
        raise ValueError("objective.preferred_nodes must be a list when provided")
    preferred_nodes: tuple[str, ...] = tuple(_norm_token(item, name="preferred node") for item in preferred_raw)

    ranked: list[tuple[tuple[str, ...], float, int, str]] = []
    identity_chain: list[str] = [GENESIS_HASH]
    for route in normalized_routes:
        depth = _route_depth(route)
        if depth > depth_limit:
            continue
        score = _route_score(route, preferred_nodes)
        route_json = _canonical_json({"route": list(route)})
        route_hash = _sha256_hex_bytes(route_json.encode("utf-8"))
        ranked.append((route, score, depth, route_hash))

    if not ranked:
        raise ValueError("no candidate routes satisfy bounded search_depth")

    ranked.sort(key=lambda item: (-item[1], item[2], item[0]))

    for rank_index, (route, score, depth, route_hash) in enumerate(ranked):
        step_hash = _sha256_hex_mapping(
            {
                "prior_hash": identity_chain[-1],
                "rank_index": rank_index,
                "route": list(route),
                "route_depth": depth,
                "route_hash": route_hash,
                "route_score": _round64(score),
                "world_state_hash": world_state_hash,
                "objective_hash": objective_hash,
            }
        )
        identity_chain.append(step_hash)

    best_route, best_score, _, _ = ranked[0]
    stability_score = compute_search_stability_score(ranked)
    return best_route, stability_score, tuple(identity_chain)


def compute_search_stability_score(ranked_routes: Sequence[tuple[tuple[str, ...], float, int, str]]) -> float:
    if not ranked_routes:
        raise ValueError("ranked_routes must not be empty")
    best_score = max(score for _, score, _, _ in ranked_routes)
    best_count = 0
    for _, score, _, _ in ranked_routes:
        if score == best_score:
            best_count += 1
    return _round64(1.0 / float(best_count))


def synthesize_plan_ir(
    world_state: Mapping[str, Any],
    objective: Mapping[str, Any],
    candidate_routes: Sequence[Sequence[str]],
    *,
    search_depth: int,
    enable_v137_6_search: bool = False,
) -> PlanIR:
    if not enable_v137_6_search:
        raise ValueError("enable_v137_6_search must be True to enable v137.6.0 planning kernel")

    normalized_world_state = _normalize_json_value(world_state, label="world_state")
    if not isinstance(normalized_world_state, dict) or not normalized_world_state:
        raise ValueError("world_state must normalize to a non-empty mapping")

    normalized_objective = _normalize_json_value(objective, label="objective")
    if not isinstance(normalized_objective, dict) or not normalized_objective:
        raise ValueError("objective must normalize to a non-empty mapping")

    normalized_routes = _normalize_routes(candidate_routes)
    bounded_search_depth = _validate_search_depth(search_depth)

    world_state_hash = _sha256_hex_mapping(normalized_world_state)
    objective_hash = _sha256_hex_mapping(normalized_objective)

    selected_route, stability_score, identity_chain = run_deterministic_search(
        normalized_routes,
        search_depth=bounded_search_depth,
        objective=normalized_objective,
        world_state_hash=world_state_hash,
        objective_hash=objective_hash,
    )
    selected_route_hash = _sha256_hex_mapping({"route": list(selected_route)})

    plan_payload = {
        "schema_version": AUTONOMOUS_PLANNING_SEARCH_KERNEL_VERSION,
        "world_state": normalized_world_state,
        "objective": normalized_objective,
        "candidate_routes": [list(route) for route in normalized_routes],
        "world_state_hash": world_state_hash,
        "objective_hash": objective_hash,
        "search_depth": bounded_search_depth,
        "search_stability_score": _round64(stability_score),
        "selected_route": list(selected_route),
        "selected_route_hash": selected_route_hash,
        "search_identity_chain": list(identity_chain),
    }
    stable_plan_hash = _compute_stable_plan_hash(plan_payload)

    return PlanIR(
        schema_version=AUTONOMOUS_PLANNING_SEARCH_KERNEL_VERSION,
        world_state=normalized_world_state,
        objective=normalized_objective,
        candidate_routes=normalized_routes,
        world_state_hash=world_state_hash,
        objective_hash=objective_hash,
        search_depth=bounded_search_depth,
        search_stability_score=stability_score,
        selected_route=selected_route,
        selected_route_hash=selected_route_hash,
        stable_plan_hash=stable_plan_hash,
        search_identity_chain=identity_chain,
    )


def export_plan_bytes(plan_ir: PlanIR) -> bytes:
    if not isinstance(plan_ir, PlanIR):
        raise TypeError("plan_ir must be a PlanIR instance")
    return plan_ir.to_canonical_bytes()


def generate_plan_receipt(plan_ir: PlanIR) -> PlanReceipt:
    if not isinstance(plan_ir, PlanIR):
        raise TypeError("plan_ir must be a PlanIR instance")

    receipt_payload = {
        "schema_version": plan_ir.schema_version,
        "world_state_hash": plan_ir.world_state_hash,
        "objective_hash": plan_ir.objective_hash,
        "search_depth": int(plan_ir.search_depth),
        "search_stability_score": _round64(plan_ir.search_stability_score),
        "selected_route_hash": plan_ir.selected_route_hash,
        "stable_plan_hash": plan_ir.stable_plan_hash,
    }
    receipt_hash = _sha256_hex_mapping(receipt_payload)

    return PlanReceipt(
        schema_version=plan_ir.schema_version,
        world_state_hash=plan_ir.world_state_hash,
        objective_hash=plan_ir.objective_hash,
        search_depth=int(plan_ir.search_depth),
        search_stability_score=plan_ir.search_stability_score,
        selected_route_hash=plan_ir.selected_route_hash,
        stable_plan_hash=plan_ir.stable_plan_hash,
        receipt_hash=receipt_hash,
    )


__all__ = [
    "AUTONOMOUS_PLANNING_SEARCH_KERNEL_VERSION",
    "GENESIS_HASH",
    "PlanIR",
    "PlanReceipt",
    "compute_search_stability_score",
    "export_plan_bytes",
    "generate_plan_receipt",
    "run_deterministic_search",
    "synthesize_plan_ir",
]
