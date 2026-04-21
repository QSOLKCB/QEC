"""v138.8.4 — Noise-Tailoring Hardware Bridge.

Deterministic analysis-layer bridge comparing tailored control outputs from
v138.8.3 against hardware-like post-mitigation measurements.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .noise_tailoring_pulse_runtime import derive_noise_control_signal

_ROUND = 12
_REQUIRED_CONFIG_KEYS = ("max_mean_node_error", "max_mean_edge_error")
_REQUIRED_HARDWARE_KEYS = ("mean_node_weight_after", "mean_edge_weight_after")


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
    return parsed


def _validate_projected_results(projected_results: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not isinstance(projected_results, Sequence) or isinstance(projected_results, (str, bytes)):
        raise ValueError("Invalid projected_results: projected_results must be a non-empty sequence")

    normalized = list(projected_results)
    if not normalized:
        raise ValueError("Invalid projected_results: projected_results must be a non-empty sequence")

    scenario_ids: list[str] = []
    for projected in normalized:
        if not isinstance(projected, Mapping):
            raise ValueError("Invalid projected_result: projected_result must be a mapping")
        scenario_id = projected.get("id")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("Invalid projected_result: 'id' must be a non-empty string")

        regime = projected.get("regime")
        if regime not in {"S1", "S2", "S3"}:
            raise ValueError("Invalid projected_result: 'regime' must be one of S1, S2, S3")

        for key in ("nodes", "edges", "node_weights", "edge_weights"):
            if key not in projected:
                raise ValueError(f"Invalid projected_result: missing required key '{key}'")

        scenario_ids.append(scenario_id)

    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Invalid projected_results: duplicate scenario ids are not allowed")

    return sorted(normalized, key=lambda item: item["id"])


def _validate_hardware_entry(hardware_entry: Mapping[str, float | int], scenario_id: str) -> tuple[float, float]:
    if not isinstance(hardware_entry, Mapping):
        raise ValueError(
            f"Invalid hardware_measurements: entry must be a mapping for scenario '{scenario_id}'"
        )

    values: dict[str, float] = {}
    for key in _REQUIRED_HARDWARE_KEYS:
        if key not in hardware_entry:
            raise ValueError(
                f"Invalid hardware_measurements: missing '{key}' for scenario '{scenario_id}'"
            )
        raw = hardware_entry[key]
        if not _is_valid_number(raw):
            raise ValueError(
                f"Invalid hardware_measurements: '{key}' must be finite numeric for scenario '{scenario_id}'"
            )
        parsed = float(raw)
        if parsed < 0.0:
            raise ValueError(
                f"Invalid hardware_measurements: '{key}' must be >= 0 for scenario '{scenario_id}'"
            )
        values[key] = parsed

    return values["mean_node_weight_after"], values["mean_edge_weight_after"]


def _mean_from_values(values: Mapping[Any, Any]) -> float:
    numeric_values = [float(value) for value in values.values()]
    if not numeric_values:
        return 0.0
    return float(sum(numeric_values) / len(numeric_values))


def compare_tailored_to_hardware(
    tailored_result: Mapping[str, Any],
    hardware_entry: Mapping[str, float | int],
) -> dict[str, float]:
    """Compute deterministic error and agreement metrics for one scenario."""
    if not isinstance(tailored_result, Mapping):
        raise ValueError("Invalid tailored_result: tailored_result must be a mapping")

    scenario_id = tailored_result.get("id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("Invalid tailored_result: 'id' must be a non-empty string")

    node_weights = tailored_result.get("node_weights")
    edge_weights = tailored_result.get("edge_weights")
    if not isinstance(node_weights, Mapping):
        raise ValueError("Invalid tailored_result: 'node_weights' must be a mapping")
    if not isinstance(edge_weights, Mapping):
        raise ValueError("Invalid tailored_result: 'edge_weights' must be a mapping")

    tailored_mean_node_weight_after = _mean_from_values(node_weights)
    tailored_mean_edge_weight_after = _mean_from_values(edge_weights)

    hardware_mean_node_weight_after, hardware_mean_edge_weight_after = _validate_hardware_entry(
        hardware_entry,
        scenario_id,
    )

    node_error = abs(tailored_mean_node_weight_after - hardware_mean_node_weight_after)
    edge_error = abs(tailored_mean_edge_weight_after - hardware_mean_edge_weight_after)

    node_agreement = 1.0 / (1.0 + node_error)
    edge_agreement = 1.0 / (1.0 + edge_error)

    return {
        "node_error": round(node_error, _ROUND),
        "edge_error": round(edge_error, _ROUND),
        "node_agreement": round(node_agreement, _ROUND),
        "edge_agreement": round(edge_agreement, _ROUND),
    }


def run_noise_tailoring_hardware_bridge(
    projected_results: Sequence[Mapping[str, Any]],
    hardware_measurements: Mapping[str, Mapping[str, float | int]],
    config: Mapping[str, float | int],
) -> dict[str, float]:
    """Run deterministic bridge validation from projected scenarios to hardware-like results."""
    validated_config = _validate_config(config)

    if not isinstance(hardware_measurements, Mapping):
        raise ValueError("Invalid hardware_measurements: hardware_measurements must be a mapping")

    ordered_projected_results = _validate_projected_results(projected_results)

    node_errors: list[float] = []
    edge_errors: list[float] = []
    node_agreements: list[float] = []
    edge_agreements: list[float] = []

    for projected in ordered_projected_results:
        scenario_id = projected["id"]
        if scenario_id not in hardware_measurements:
            raise ValueError(
                f"Invalid hardware_measurements: missing scenario id '{scenario_id}'"
            )

        tailored = derive_noise_control_signal(projected, config)
        per_scenario = compare_tailored_to_hardware(tailored, hardware_measurements[scenario_id])

        node_errors.append(per_scenario["node_error"])
        edge_errors.append(per_scenario["edge_error"])
        node_agreements.append(per_scenario["node_agreement"])
        edge_agreements.append(per_scenario["edge_agreement"])

    count = float(len(ordered_projected_results))
    metrics = {
        "mean_node_error": round(sum(node_errors) / count, _ROUND),
        "mean_edge_error": round(sum(edge_errors) / count, _ROUND),
        "mean_node_agreement": round(sum(node_agreements) / count, _ROUND),
        "mean_edge_agreement": round(sum(edge_agreements) / count, _ROUND),
    }

    if metrics["mean_node_error"] > validated_config["max_mean_node_error"]:
        raise ValueError(
            "Validation failed: mean_node_error exceeds configured max_mean_node_error"
        )
    if metrics["mean_edge_error"] > validated_config["max_mean_edge_error"]:
        raise ValueError(
            "Validation failed: mean_edge_error exceeds configured max_mean_edge_error"
        )

    return metrics
