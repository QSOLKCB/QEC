"""v138.8.1 — Cluster Expansion Noise Kernel (deterministic, SU(3)-aligned).

Deterministic analysis-layer extension of the v138.8.0 correlated noise model:
- base factors: local contrast (λ3) + global drift (λ8)
- pairwise cluster expansion term:
  - node correction by relative degree
  - uniform edge correction

No randomness, no stochastic sampling, no decoder-core coupling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

_ROUND = 12
_REQUIRED_CONFIG_KEYS = (
    "local_contrast_noise_scale",
    "global_drift_noise_scale",
    "pairwise_correlation_scale",
    "stable_threshold",
    "relaxation_threshold",
)


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
        parsed[key] = _require_nonnegative_number(config[key], key)

    if parsed["stable_threshold"] > parsed["relaxation_threshold"]:
        raise ValueError("Invalid config: stable_threshold must be <= relaxation_threshold")

    return parsed


def _edge_key(edge: tuple[str, str]) -> str:
    return f"{edge[0]}->{edge[1]}"


def _validate_scenario(scenario: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(scenario, Mapping):
        raise ValueError("Invalid scenario: scenario must be a mapping")

    scenario_id = scenario.get("id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("Invalid scenario: 'id' must be a non-empty string")

    nodes_raw = scenario.get("nodes")
    if not isinstance(nodes_raw, Sequence) or isinstance(nodes_raw, (str, bytes)):
        raise ValueError("Invalid scenario: 'nodes' must be a sequence of strings")
    nodes: list[str] = []
    for node in nodes_raw:
        if not isinstance(node, str):
            raise ValueError("Invalid scenario: 'nodes' must contain only strings")
        nodes.append(node)
    if len(set(nodes)) != len(nodes):
        raise ValueError("Invalid scenario: 'nodes' must contain unique node identifiers")

    node_set = set(nodes)
    edges_raw = scenario.get("edges")
    if not isinstance(edges_raw, Sequence) or isinstance(edges_raw, (str, bytes)):
        raise ValueError("Invalid scenario: 'edges' must be a sequence of (src, dst) tuples")
    edges: list[tuple[str, str]] = []
    for edge in edges_raw:
        if (
            not isinstance(edge, tuple)
            or len(edge) != 2
            or not isinstance(edge[0], str)
            or not isinstance(edge[1], str)
        ):
            raise ValueError("Invalid scenario: 'edges' must contain only (str, str) tuples")
        if edge[0] not in node_set or edge[1] not in node_set:
            raise ValueError("Invalid scenario: all edge endpoints must be present in 'nodes'")
        edges.append((edge[0], edge[1]))
    if len(set(edges)) != len(edges):
        raise ValueError("Invalid scenario: 'edges' must not contain duplicates")

    known_edge_keys = {_edge_key(edge) for edge in edges}

    node_weights_raw = scenario.get("node_weights", {})
    if node_weights_raw is None:
        node_weights_raw = {}
    if not isinstance(node_weights_raw, Mapping):
        raise ValueError("Invalid scenario: 'node_weights' must be a mapping")

    edge_weights_raw = scenario.get("edge_weights", {})
    if edge_weights_raw is None:
        edge_weights_raw = {}
    if not isinstance(edge_weights_raw, Mapping):
        raise ValueError("Invalid scenario: 'edge_weights' must be a mapping")

    node_weights: dict[str, float] = {}
    for key, value in node_weights_raw.items():
        if not isinstance(key, str):
            raise ValueError("Invalid scenario: 'node_weights' keys must be strings")
        if key not in node_set:
            raise ValueError("Invalid scenario: 'node_weights' keys must match declared nodes")
        if not _is_valid_number(value):
            raise ValueError("Invalid scenario: 'node_weights' values must be finite numeric values")
        node_weights[key] = float(value)

    edge_weights: dict[str, float] = {}
    for key in known_edge_keys:
        if key not in edge_weights_raw:
            continue
        value = edge_weights_raw[key]
        if not _is_valid_number(value):
            raise ValueError("Invalid scenario: 'edge_weights' values must be finite numeric values")
        edge_weights[key] = float(value)

    nodes_sorted = sorted(nodes)
    edges_sorted = sorted(edges, key=lambda edge: (edge[0], edge[1]))

    return {
        "id": scenario_id,
        "nodes": nodes_sorted,
        "edges": edges_sorted,
        "node_weights": node_weights,
        "edge_weights": edge_weights,
    }


def _classify_noise_regime(normalized: Mapping[str, Any], parsed_config: Mapping[str, float]) -> str:
    total_load = len(normalized["nodes"]) + len(normalized["edges"])
    if total_load <= parsed_config["stable_threshold"]:
        return "S1"
    if total_load >= parsed_config["relaxation_threshold"]:
        return "S3"
    return "S2"


def _su3_factors(regime: str, parsed_config: Mapping[str, float]) -> tuple[float, float]:
    if regime == "S1":
        return (
            1.0 - (parsed_config["local_contrast_noise_scale"] * 0.5),
            1.0 - (parsed_config["global_drift_noise_scale"] * 0.25),
        )
    if regime == "S2":
        return (
            1.0 - (parsed_config["local_contrast_noise_scale"] * 1.0),
            1.0 - (parsed_config["global_drift_noise_scale"] * 1.0),
        )
    if regime == "S3":
        return (
            1.0 - (parsed_config["local_contrast_noise_scale"] * 0.25),
            1.0 - (parsed_config["global_drift_noise_scale"] * 0.5),
        )
    raise ValueError("Invalid regime output: expected one of S1, S2, S3")


def apply_cluster_expansion_noise(
    scenario: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply deterministic SU(3) local+global+pairwise cluster expansion noise."""
    parsed_config = _validate_config(config)
    normalized = _validate_scenario(scenario)

    regime = _classify_noise_regime(normalized, parsed_config)
    local_factor, drift_factor = _su3_factors(regime, parsed_config)

    degree: dict[str, int] = {node: 0 for node in normalized["nodes"]}
    for source, target in normalized["edges"]:
        degree[source] += 1
        degree[target] += 1
    max_degree = max(degree.values(), default=0)

    pairwise_scale = parsed_config["pairwise_correlation_scale"]

    node_weights_out: dict[str, float] = {}
    for node in normalized["nodes"]:
        base = normalized["node_weights"].get(node, 1.0)
        node_pairwise = 1.0
        if max_degree > 0:
            node_pairwise = 1.0 - (pairwise_scale * float(degree[node]) / float(max_degree))
        computed = base * local_factor * drift_factor * node_pairwise
        if not math.isfinite(computed):
            raise ValueError("Invalid computed weight: non-finite node weight encountered")
        node_weights_out[node] = round(max(0.0, computed), _ROUND)

    edge_pairwise = 1.0 - pairwise_scale
    edge_weights_out: dict[str, float] = {}
    for edge in normalized["edges"]:
        key = _edge_key(edge)
        base = normalized["edge_weights"].get(key, 1.0)
        computed = base * local_factor * drift_factor * edge_pairwise
        if not math.isfinite(computed):
            raise ValueError("Invalid computed weight: non-finite edge weight encountered")
        edge_weights_out[key] = round(max(0.0, computed), _ROUND)

    return {
        "id": normalized["id"],
        "regime": regime,
        "nodes": normalized["nodes"],
        "edges": normalized["edges"],
        "node_weights": node_weights_out,
        "edge_weights": edge_weights_out,
    }


def run_cluster_expansion_simulation(
    scenarios: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Run deterministic aggregate simulation over sorted scenarios."""
    parsed_config = _validate_config(config)
    normalized_scenarios = list(scenarios)
    if not normalized_scenarios:
        raise ValueError("Invalid scenarios: scenarios must be a non-empty sequence")

    validated = [_validate_scenario(scenario) for scenario in normalized_scenarios]
    ids = [scenario["id"] for scenario in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("Invalid scenarios: duplicate scenario ids are not allowed")

    ordered = sorted(validated, key=lambda scenario: scenario["id"])

    node_weights_all: list[float] = []
    edge_weights_all: list[float] = []
    node_deltas_all: list[float] = []
    edge_deltas_all: list[float] = []

    for scenario_norm in ordered:
        perturbed = _apply_cluster_expansion_noise_validated(
            scenario_norm,
            parsed_config,
        )

        for node in scenario_norm["nodes"]:
            base = scenario_norm["node_weights"].get(node, 1.0)
            perturbed_weight = perturbed["node_weights"][node]
            node_weights_all.append(float(perturbed_weight))
            node_deltas_all.append(float(base) - float(perturbed_weight))

        for edge in scenario_norm["edges"]:
            key = _edge_key(edge)
            base = scenario_norm["edge_weights"].get(key, 1.0)
            perturbed_weight = perturbed["edge_weights"][key]
            edge_weights_all.append(float(perturbed_weight))
            edge_deltas_all.append(float(base) - float(perturbed_weight))

    def _mean(values: list[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    return {
        "mean_node_weight": round(_mean(node_weights_all), _ROUND),
        "mean_edge_weight": round(_mean(edge_weights_all), _ROUND),
        "mean_node_noise_delta": round(_mean(node_deltas_all), _ROUND),
        "mean_edge_noise_delta": round(_mean(edge_deltas_all), _ROUND),
    }
