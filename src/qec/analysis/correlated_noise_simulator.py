"""v138.8.0 — Correlated Noise Simulator (SU(3-native bootstrap)).

Deterministic analysis-layer simulation with a triadic SU(3)/qutrit regime
interpretation:
- S1: stable
- S2: perturbation
- S3: relaxation

No randomness, no time dependence, no decoder core coupling.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

_ROUND = 12
_REQUIRED_CONFIG_KEYS = (
    "local_contrast_noise_scale",
    "global_drift_noise_scale",
    "stable_threshold",
    "relaxation_threshold",
)


def _is_valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_nonnegative_number(value: Any, name: str) -> float:
    if not _is_valid_number(value):
        raise ValueError(f"Invalid config: '{name}' must be a finite numeric value (not bool)")
    value_f = float(value)
    if value_f < 0.0:
        raise ValueError(f"Invalid config: '{name}' must be >= 0")
    return value_f


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

    nodes = scenario.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        raise ValueError("Invalid scenario: 'nodes' must be a sequence of strings")
    for node in nodes:
        if not isinstance(node, str):
            raise ValueError("Invalid scenario: 'nodes' must contain only strings")

    edges = scenario.get("edges")
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        raise ValueError("Invalid scenario: 'edges' must be a sequence of (src, dst) tuples")
    normalized_edges: list[tuple[str, str]] = []
    for edge in edges:
        if (
            not isinstance(edge, tuple)
            or len(edge) != 2
            or not isinstance(edge[0], str)
            or not isinstance(edge[1], str)
        ):
            raise ValueError("Invalid scenario: 'edges' must contain only (str, str) tuples")
        normalized_edges.append((edge[0], edge[1]))

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
    for k, v in node_weights_raw.items():
        if not isinstance(k, str):
            raise ValueError("Invalid scenario: 'node_weights' keys must be strings")
        if not _is_valid_number(v):
            raise ValueError("Invalid scenario: 'node_weights' values must be finite numeric values")
        node_weights[k] = float(v)

    edge_weights: dict[str, float] = {}
    for k, v in edge_weights_raw.items():
        if not isinstance(k, str):
            raise ValueError("Invalid scenario: 'edge_weights' keys must be strings")
        if not _is_valid_number(v):
            raise ValueError("Invalid scenario: 'edge_weights' values must be finite numeric values")
        edge_weights[k] = float(v)

    return {
        "id": scenario_id,
        "nodes": list(nodes),
        "edges": normalized_edges,
        "node_weights": node_weights,
        "edge_weights": edge_weights,
    }


def classify_noise_regime(scenario: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    """Classify a scenario into S1/S2/S3 by total load vs thresholds."""
    parsed_config = _validate_config(config)
    normalized = _validate_scenario(scenario)

    total_load = len(normalized["nodes"]) + len(normalized["edges"])
    if total_load <= parsed_config["stable_threshold"]:
        return "S1"
    if total_load >= parsed_config["relaxation_threshold"]:
        return "S3"
    return "S2"


def apply_su3_correlated_noise(
    scenario: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply deterministic SU(3)-interpreted local+drift noise to one scenario."""
    parsed_config = _validate_config(config)
    normalized = _validate_scenario(scenario)

    regime = classify_noise_regime(normalized, parsed_config)
    if regime not in {"S1", "S2", "S3"}:
        raise ValueError("Invalid regime output: expected one of S1, S2, S3")

    if regime == "S1":
        local_factor = 1.0 - (parsed_config["local_contrast_noise_scale"] * 0.5)
        drift_factor = 1.0 - (parsed_config["global_drift_noise_scale"] * 0.25)
    elif regime == "S2":
        local_factor = 1.0 - (parsed_config["local_contrast_noise_scale"] * 1.0)
        drift_factor = 1.0 - (parsed_config["global_drift_noise_scale"] * 1.0)
    else:
        local_factor = 1.0 - (parsed_config["local_contrast_noise_scale"] * 0.25)
        drift_factor = 1.0 - (parsed_config["global_drift_noise_scale"] * 0.5)

    nodes_sorted = sorted(normalized["nodes"])
    edges_sorted = sorted(normalized["edges"])

    node_weights_out: dict[str, float] = {}
    for node in nodes_sorted:
        base = normalized["node_weights"].get(node, 1.0)
        computed = base * local_factor * drift_factor
        if not math.isfinite(computed):
            raise ValueError("Invalid computed weight: non-finite node weight encountered")
        node_weights_out[node] = round(max(0.0, computed), _ROUND)

    edge_weights_out: dict[str, float] = {}
    for edge in edges_sorted:
        key = _edge_key(edge)
        base = normalized["edge_weights"].get(key, 1.0)
        computed = base * local_factor * drift_factor
        if not math.isfinite(computed):
            raise ValueError("Invalid computed weight: non-finite edge weight encountered")
        edge_weights_out[key] = round(max(0.0, computed), _ROUND)

    return {
        "id": normalized["id"],
        "regime": regime,
        "nodes": nodes_sorted,
        "edges": edges_sorted,
        "node_weights": node_weights_out,
        "edge_weights": edge_weights_out,
    }


def run_correlated_noise_simulation(
    scenarios: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Run deterministic aggregate correlated-noise simulation across scenarios."""
    parsed_config = _validate_config(config)
    normalized_scenarios = list(scenarios)
    if not normalized_scenarios:
        raise ValueError("Invalid scenarios: scenarios must be a non-empty sequence")

    validated: list[dict[str, Any]] = [_validate_scenario(s) for s in normalized_scenarios]
    ids = [s["id"] for s in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("Invalid scenarios: duplicate scenario ids are not allowed")

    ordered = sorted(validated, key=lambda s: s["id"])

    node_weights_all: list[float] = []
    edge_weights_all: list[float] = []
    node_deltas_all: list[float] = []
    edge_deltas_all: list[float] = []

    for scenario in ordered:
        perturbed = apply_su3_correlated_noise(scenario, parsed_config)

        for node in sorted(scenario["nodes"]):
            base = scenario["node_weights"].get(node, 1.0)
            p = perturbed["node_weights"][node]
            node_weights_all.append(float(p))
            node_deltas_all.append(float(base) - float(p))

        for edge in sorted(scenario["edges"]):
            key = _edge_key(edge)
            base = scenario["edge_weights"].get(key, 1.0)
            p = perturbed["edge_weights"][key]
            edge_weights_all.append(float(p))
            edge_deltas_all.append(float(base) - float(p))

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
