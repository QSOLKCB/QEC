"""v138.8.2 — Honest Noise Approximation Pack.

Constraint + projection layer that maps arbitrary noise outputs into physically
admissible, bounded, comparable, and deterministic representations.

Pipeline (per weight):
1) clamp
2) contract
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

_ROUND = 12
_REQUIRED_CONFIG_KEYS = (
    "max_node_weight",
    "max_edge_weight",
    "min_weight",
    "contraction_factor",
)
Edge = tuple[str, str]


def _is_valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_nonnegative_number(value: Any, name: str) -> float:
    if not _is_valid_number(value):
        raise ValueError(f"Invalid config: '{name}' must be a finite numeric value (not bool)")
    parsed = float(value)
    if parsed < 0.0:
        raise ValueError(f"Invalid config: '{name}' must be >= 0")
    return parsed


def _validate_config(config: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(config, Mapping):
        raise ValueError("Invalid config: config must be a mapping")

    parsed: dict[str, float] = {}
    for key in _REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise ValueError(f"Invalid config: missing required key '{key}'")

    parsed["max_node_weight"] = _require_nonnegative_number(config["max_node_weight"], "max_node_weight")
    parsed["max_edge_weight"] = _require_nonnegative_number(config["max_edge_weight"], "max_edge_weight")
    parsed["min_weight"] = _require_nonnegative_number(config["min_weight"], "min_weight")

    contraction = config["contraction_factor"]
    if not _is_valid_number(contraction):
        raise ValueError("Invalid config: 'contraction_factor' must be a finite numeric value (not bool)")
    parsed["contraction_factor"] = float(contraction)
    if parsed["contraction_factor"] <= 0.0 or parsed["contraction_factor"] > 1.0:
        raise ValueError("Invalid config: 'contraction_factor' must satisfy 0 < contraction_factor <= 1")

    if parsed["min_weight"] > parsed["max_node_weight"]:
        raise ValueError("Invalid config: 'min_weight' must be <= 'max_node_weight'")
    if parsed["min_weight"] > parsed["max_edge_weight"]:
        raise ValueError("Invalid config: 'min_weight' must be <= 'max_edge_weight'")

    return parsed


def _validate_noisy_result(noisy_result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(noisy_result, Mapping):
        raise ValueError("Invalid noisy_result: noisy_result must be a mapping")

    scenario_id = noisy_result.get("id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("Invalid noisy_result: 'id' must be a non-empty string")

    regime = noisy_result.get("regime")
    if regime not in {"S1", "S2", "S3"}:
        raise ValueError("Invalid noisy_result: 'regime' must be one of S1, S2, S3")

    nodes_raw = noisy_result.get("nodes")
    if not isinstance(nodes_raw, Sequence) or isinstance(nodes_raw, (str, bytes)):
        raise ValueError("Invalid noisy_result: 'nodes' must be a sequence of strings")
    nodes: list[str] = []
    for node in nodes_raw:
        if not isinstance(node, str):
            raise ValueError("Invalid noisy_result: 'nodes' must contain only strings")
        nodes.append(node)
    if len(set(nodes)) != len(nodes):
        raise ValueError("Invalid noisy_result: 'nodes' must contain unique node identifiers")

    node_set = set(nodes)

    edges_raw = noisy_result.get("edges")
    if not isinstance(edges_raw, Sequence) or isinstance(edges_raw, (str, bytes)):
        raise ValueError("Invalid noisy_result: 'edges' must be a sequence of (src, dst) tuples")
    edges: list[Edge] = []
    for edge in edges_raw:
        if (
            not isinstance(edge, tuple)
            or len(edge) != 2
            or not isinstance(edge[0], str)
            or not isinstance(edge[1], str)
        ):
            raise ValueError("Invalid noisy_result: 'edges' must contain only (str, str) tuples")
        if edge[0] not in node_set or edge[1] not in node_set:
            raise ValueError("Invalid noisy_result: all edge endpoints must be present in 'nodes'")
        edges.append((edge[0], edge[1]))
    if len(set(edges)) != len(edges):
        raise ValueError("Invalid noisy_result: 'edges' must not contain duplicates")

    node_weights_raw = noisy_result.get("node_weights")
    if not isinstance(node_weights_raw, Mapping):
        raise ValueError("Invalid noisy_result: 'node_weights' must be a mapping")

    edge_weights_raw = noisy_result.get("edge_weights")
    if not isinstance(edge_weights_raw, Mapping):
        raise ValueError("Invalid noisy_result: 'edge_weights' must be a mapping")

    edge_key_to_tuple: dict[str, Edge] = {_edge_key(edge): edge for edge in edges}
    if len(edge_key_to_tuple) != len(edges):
        raise ValueError("Invalid noisy_result: 'edges' produce ambiguous edge_weights keys")

    node_weights: dict[str, float] = {}
    for key, value in node_weights_raw.items():
        if not isinstance(key, str) or key not in node_set:
            raise ValueError("Invalid noisy_result: 'node_weights' keys must match declared nodes")
        if not _is_valid_number(value):
            raise ValueError("Invalid noisy_result: 'node_weights' values must be finite numeric values")
        node_weights[key] = float(value)

    edge_weights: dict[Edge, float] = {}
    for key, value in edge_weights_raw.items():
        if not isinstance(key, str):
            raise ValueError("Invalid noisy_result: 'edge_weights' keys must be strings")
        if key not in edge_key_to_tuple:
            raise ValueError("Invalid noisy_result: 'edge_weights' keys must match declared edges")
        if not _is_valid_number(value):
            raise ValueError("Invalid noisy_result: 'edge_weights' values must be finite numeric values")
        edge_weights[edge_key_to_tuple[key]] = float(value)

    return {
        "id": scenario_id,
        "regime": regime,
        "nodes": nodes,
        "edges": edges,
        "node_weights": node_weights,
        "edge_weights": edge_weights,
    }


def _project_weight(raw: float, min_weight: float, max_weight: float, contraction_factor: float) -> float:
    clamped = min(max(raw, min_weight), max_weight)
    contracted = clamped * contraction_factor
    return contracted


def _edge_key(edge: Edge) -> str:
    return f"{edge[0]}->{edge[1]}"


def _ordered_edges(edges: Sequence[Edge]) -> list[Edge]:
    return sorted(edges, key=lambda pair: (pair[0], pair[1]))


def _finalize_weights(weights: Mapping[str, float], ordered_keys: Sequence[str], min_weight: float, max_weight: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ordered_keys:
        bounded = min(max(weights[key], min_weight), max_weight)
        if not math.isfinite(bounded):
            raise ValueError("Invalid computed weight: non-finite projected weight encountered")
        out[key] = round(bounded, _ROUND)
    return out


def _finalize_edge_weights(
    weights: Mapping[Edge, float],
    ordered_edges: Sequence[Edge],
    min_weight: float,
    max_weight: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for edge in ordered_edges:
        bounded = min(max(weights[edge], min_weight), max_weight)
        if not math.isfinite(bounded):
            raise ValueError("Invalid computed weight: non-finite projected weight encountered")
        out[_edge_key(edge)] = round(bounded, _ROUND)
    return out


def _project_to_honest_noise_validated(
    normalized: Mapping[str, Any],
    parsed_config: Mapping[str, float],
    *,
    ordered_nodes: Sequence[str] | None = None,
    ordered_edges: Sequence[Edge] | None = None,
) -> dict[str, Any]:
    if ordered_nodes is None:
        ordered_nodes = sorted(normalized["nodes"])
    if ordered_edges is None:
        ordered_edges = _ordered_edges(normalized["edges"])
    node_projected: dict[str, float] = {}
    for node in ordered_nodes:
        base = normalized["node_weights"].get(node, 1.0)
        node_projected[node] = _project_weight(
            base,
            parsed_config["min_weight"],
            parsed_config["max_node_weight"],
            parsed_config["contraction_factor"],
        )

    edge_projected: dict[Edge, float] = {}
    for edge in ordered_edges:
        base = normalized["edge_weights"].get(edge, 1.0)
        edge_projected[edge] = _project_weight(
            base,
            parsed_config["min_weight"],
            parsed_config["max_edge_weight"],
            parsed_config["contraction_factor"],
        )

    node_out = _finalize_weights(
        node_projected,
        ordered_nodes,
        parsed_config["min_weight"],
        parsed_config["max_node_weight"],
    )
    edge_out = _finalize_edge_weights(
        edge_projected,
        ordered_edges,
        parsed_config["min_weight"],
        parsed_config["max_edge_weight"],
    )

    return {
        "id": normalized["id"],
        "regime": normalized["regime"],
        "nodes": list(normalized["nodes"]),
        "edges": list(normalized["edges"]),
        "node_weights": node_out,
        "edge_weights": edge_out,
    }


def project_to_honest_noise(noisy_result: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Project one noisy result into bounded and deterministic honest-noise space."""
    parsed_config = _validate_config(config)
    normalized = _validate_noisy_result(noisy_result)
    return _project_to_honest_noise_validated(normalized, parsed_config)


def run_honest_noise_projection(
    scenarios: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Run deterministic aggregate projection across scenarios."""
    parsed_config = _validate_config(config)
    normalized_scenarios = list(scenarios)
    if not normalized_scenarios:
        raise ValueError("Invalid scenarios: scenarios must be a non-empty sequence")

    validated = [_validate_noisy_result(scenario) for scenario in normalized_scenarios]
    ids = [scenario["id"] for scenario in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("Invalid scenarios: duplicate scenario ids are not allowed")

    ordered = sorted(validated, key=lambda scenario: scenario["id"])

    node_weights_all: list[float] = []
    edge_weights_all: list[float] = []
    node_adjustments_all: list[float] = []
    edge_adjustments_all: list[float] = []

    for scenario in ordered:
        ordered_nodes = sorted(scenario["nodes"])
        ordered_edges = _ordered_edges(scenario["edges"])
        projected = _project_to_honest_noise_validated(
            scenario,
            parsed_config,
            ordered_nodes=ordered_nodes,
            ordered_edges=ordered_edges,
        )

        for node in ordered_nodes:
            key = str(node)
            base = float(scenario["node_weights"].get(key, 1.0))
            out = float(projected["node_weights"][key])
            node_weights_all.append(out)
            node_adjustments_all.append(out - base)

        for edge in ordered_edges:
            key = _edge_key(edge)
            base = float(scenario["edge_weights"].get(edge, 1.0))
            out = float(projected["edge_weights"][key])
            edge_weights_all.append(out)
            edge_adjustments_all.append(out - base)

    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    return {
        "mean_node_weight": round(_mean(node_weights_all), _ROUND),
        "mean_edge_weight": round(_mean(edge_weights_all), _ROUND),
        "mean_node_adjustment": round(_mean(node_adjustments_all), _ROUND),
        "mean_edge_adjustment": round(_mean(edge_adjustments_all), _ROUND),
    }
