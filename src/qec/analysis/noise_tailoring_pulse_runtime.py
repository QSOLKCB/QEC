"""v138.8.3 — Noise-Tailoring Pulse Runtime.

Deterministic control-policy layer that transforms admissible noise projections
into mitigation-oriented control signals. This module does not model hardware;
it deterministically rescales node/edge noise weights by regime-aware factors.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

_ROUND = 12
_REQUIRED_CONFIG_KEYS = (
    "node_mitigation_strength",
    "edge_mitigation_strength",
    "regime_bias_strength",
)
_REGIME_FACTORS = {
    "S1": 0.5,
    "S2": 1.0,
    "S3": 1.5,
}
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

    parsed["node_mitigation_strength"] = _require_nonnegative_number(
        config["node_mitigation_strength"],
        "node_mitigation_strength",
    )
    parsed["edge_mitigation_strength"] = _require_nonnegative_number(
        config["edge_mitigation_strength"],
        "edge_mitigation_strength",
    )
    parsed["regime_bias_strength"] = _require_nonnegative_number(
        config["regime_bias_strength"],
        "regime_bias_strength",
    )

    if parsed["node_mitigation_strength"] > 1.0:
        raise ValueError("Invalid config: 'node_mitigation_strength' must be <= 1.0")
    if parsed["edge_mitigation_strength"] > 1.0:
        raise ValueError("Invalid config: 'edge_mitigation_strength' must be <= 1.0")

    return parsed


def _edge_key(edge: Edge) -> str:
    return f"{edge[0]}->{edge[1]}"


def _ordered_edges(edges: Sequence[Edge]) -> list[Edge]:
    return sorted(edges, key=lambda pair: (pair[0], pair[1]))


def _validate_projected_result(projected_result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(projected_result, Mapping):
        raise ValueError("Invalid projected_result: projected_result must be a mapping")

    scenario_id = projected_result.get("id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("Invalid projected_result: 'id' must be a non-empty string")

    regime = projected_result.get("regime")
    if regime not in _REGIME_FACTORS:
        raise ValueError("Invalid projected_result: 'regime' must be one of S1, S2, S3")

    nodes_raw = projected_result.get("nodes")
    if not isinstance(nodes_raw, Sequence) or isinstance(nodes_raw, (str, bytes)):
        raise ValueError("Invalid projected_result: 'nodes' must be a sequence of strings")
    nodes: list[str] = []
    for node in nodes_raw:
        if not isinstance(node, str):
            raise ValueError("Invalid projected_result: 'nodes' must contain only strings")
        nodes.append(node)
    if len(set(nodes)) != len(nodes):
        raise ValueError("Invalid projected_result: 'nodes' must contain unique node identifiers")

    node_set = set(nodes)

    edges_raw = projected_result.get("edges")
    if not isinstance(edges_raw, Sequence) or isinstance(edges_raw, (str, bytes)):
        raise ValueError("Invalid projected_result: 'edges' must be a sequence of (src, dst) tuples")
    edges: list[Edge] = []
    for edge in edges_raw:
        if (
            not isinstance(edge, tuple)
            or len(edge) != 2
            or not isinstance(edge[0], str)
            or not isinstance(edge[1], str)
        ):
            raise ValueError("Invalid projected_result: 'edges' must contain only (str, str) tuples")
        if edge[0] not in node_set or edge[1] not in node_set:
            raise ValueError("Invalid projected_result: all edge endpoints must be present in 'nodes'")
        edges.append((edge[0], edge[1]))

    node_weights_raw = projected_result.get("node_weights")
    if not isinstance(node_weights_raw, Mapping):
        raise ValueError("Invalid projected_result: 'node_weights' must be a mapping")

    edge_weights_raw = projected_result.get("edge_weights")
    if not isinstance(edge_weights_raw, Mapping):
        raise ValueError("Invalid projected_result: 'edge_weights' must be a mapping")

    edge_key_to_tuple: dict[str, Edge] = {_edge_key(edge): edge for edge in edges}
    if len(edge_key_to_tuple) != len(edges):
        raise ValueError("Invalid projected_result: 'edges' produce ambiguous edge_weights keys")

    node_weights: dict[str, float] = {}
    for key, value in node_weights_raw.items():
        if not isinstance(key, str) or key not in node_set:
            raise ValueError("Invalid projected_result: 'node_weights' keys must match declared nodes")
        if not _is_valid_number(value):
            raise ValueError("Invalid projected_result: 'node_weights' values must be finite numeric values")
        node_weights[key] = float(value)

    edge_weights: dict[Edge, float] = {}
    for key, value in edge_weights_raw.items():
        if not isinstance(key, str) or key not in edge_key_to_tuple:
            raise ValueError("Invalid projected_result: 'edge_weights' keys must match declared edges")
        if not _is_valid_number(value):
            raise ValueError("Invalid projected_result: 'edge_weights' values must be finite numeric values")
        edge_weights[edge_key_to_tuple[key]] = float(value)

    return {
        "id": scenario_id,
        "regime": regime,
        "nodes": nodes,
        "edges": edges,
        "node_weights": node_weights,
        "edge_weights": edge_weights,
    }


def derive_noise_control_signal(
    projected_result: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive deterministic regime-aware control signal for one scenario."""
    parsed = _validate_config(config)
    validated = _validate_projected_result(projected_result)
    return _derive_noise_control_signal_validated(validated, parsed)


def _derive_noise_control_signal_validated(
    validated: Mapping[str, Any],
    parsed: Mapping[str, float],
) -> dict[str, Any]:
    regime_factor = _REGIME_FACTORS[validated["regime"]] * parsed["regime_bias_strength"]
    ordered_nodes = sorted(validated["nodes"])
    ordered_edges = _ordered_edges(validated["edges"])

    node_weights_out: dict[str, float] = {}
    for node in ordered_nodes:
        base = validated["node_weights"].get(node, 0.0)
        control_factor = 1.0 - (parsed["node_mitigation_strength"] * regime_factor * base)
        adjusted = base * control_factor
        if adjusted < 0.0:
            adjusted = 0.0
        if not math.isfinite(adjusted):
            raise ValueError("Invalid computed weight: non-finite mitigated node weight encountered")
        node_weights_out[node] = round(adjusted, _ROUND)

    edge_weights_out: dict[str, float] = {}
    for edge in ordered_edges:
        base = validated["edge_weights"].get(edge, 0.0)
        control_factor = 1.0 - (parsed["edge_mitigation_strength"] * regime_factor * base)
        adjusted = base * control_factor
        if adjusted < 0.0:
            adjusted = 0.0
        if not math.isfinite(adjusted):
            raise ValueError("Invalid computed weight: non-finite mitigated edge weight encountered")
        edge_weights_out[_edge_key(edge)] = round(adjusted, _ROUND)

    return {
        "id": validated["id"],
        "regime": validated["regime"],
        "nodes": ordered_nodes,
        "edges": ordered_edges,
        "node_weights": node_weights_out,
        "edge_weights": edge_weights_out,
        "control_metadata": {
            "regime_factor": round(regime_factor, _ROUND),
        },
    }


def run_noise_tailoring_runtime(
    scenarios: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Run deterministic aggregate noise-tailoring statistics over scenarios."""
    parsed = _validate_config(config)
    normalized_scenarios = list(scenarios)
    if not normalized_scenarios:
        raise ValueError("Invalid scenarios: scenarios must be a non-empty sequence")

    validated = [_validate_projected_result(scenario) for scenario in normalized_scenarios]
    ids = [scenario["id"] for scenario in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("Invalid scenarios: duplicate scenario ids are not allowed")

    ordered = sorted(validated, key=lambda scenario: scenario["id"])

    node_after_all: list[float] = []
    edge_after_all: list[float] = []
    node_reduction_all: list[float] = []
    edge_reduction_all: list[float] = []

    for scenario in ordered:
        control = _derive_noise_control_signal_validated(scenario, parsed)
        for node in control["nodes"]:
            base = float(scenario["node_weights"].get(node, 0.0))
            after = float(control["node_weights"][node])
            node_after_all.append(after)
            node_reduction_all.append(base - after)

        for edge in control["edges"]:
            key = _edge_key(edge)
            base = float(scenario["edge_weights"].get(edge, 0.0))
            after = float(control["edge_weights"][key])
            edge_after_all.append(after)
            edge_reduction_all.append(base - after)

    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    return {
        "mean_node_weight_after": round(_mean(node_after_all), _ROUND),
        "mean_edge_weight_after": round(_mean(edge_after_all), _ROUND),
        "mean_node_reduction": round(_mean(node_reduction_all), _ROUND),
        "mean_edge_reduction": round(_mean(edge_reduction_all), _ROUND),
    }
