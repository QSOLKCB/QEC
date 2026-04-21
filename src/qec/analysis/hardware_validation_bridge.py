"""v138.7.4 — deterministic ML hardware validation bridge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .neural_acceleration_sim import run_neural_acceleration_simulation


def _validate_scenarios(scenarios: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
        raise ValueError("scenarios must be a non-empty sequence")

    scenario_list = list(scenarios)
    if not scenario_list:
        raise ValueError("scenarios must be a non-empty sequence")

    scenario_ids: list[str] = []
    for scenario in scenario_list:
        if not isinstance(scenario, Mapping):
            raise ValueError("scenario must be mapping-like")
        scenario_id = scenario.get("id")
        if not isinstance(scenario_id, str):
            raise ValueError("scenario id must be str")
        scenario_ids.append(scenario_id)

    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario ids must be unique")

    return sorted(scenario_list, key=lambda scenario: scenario["id"])


def _validate_hardware_latency(value: Any, *, scenario_id: str) -> float:
    if not isinstance(value, float) or isinstance(value, bool) or value <= 0.0:
        raise ValueError(
            f"hardware latency must be positive float for scenario '{scenario_id}'"
        )
    return float(value)


def run_hardware_validation_bridge(
    scenarios: Sequence[Mapping[str, Any]],
    hardware_measurements: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    sorted_scenarios = _validate_scenarios(scenarios)

    if not isinstance(hardware_measurements, Mapping):
        raise ValueError("hardware_measurements must be mapping-like")

    for scenario in sorted_scenarios:
        scenario_id = scenario["id"]
        if scenario_id not in hardware_measurements:
            raise ValueError(
                f"hardware_measurements missing scenario id '{scenario_id}'"
            )

    total_simulated_latency = 0.0
    total_hardware_latency = 0.0
    total_absolute_error = 0.0
    total_relative_error = 0.0
    total_agreement_score = 0.0

    for scenario in sorted_scenarios:
        scenario_id = scenario["id"]

        simulation_result = run_neural_acceleration_simulation([scenario])
        simulated_latency = float(simulation_result["mean_latency_accelerated"])

        hardware_entry = hardware_measurements[scenario_id]
        if not isinstance(hardware_entry, Mapping):
            raise ValueError(
                f"hardware_measurements entry must be mapping-like for scenario '{scenario_id}'"
            )
        if "latency" not in hardware_entry:
            raise ValueError(
                f"hardware_measurements missing latency for scenario '{scenario_id}'"
            )
        hardware_latency = _validate_hardware_latency(
            hardware_entry["latency"], scenario_id=scenario_id
        )

        absolute_error = abs(simulated_latency - hardware_latency)
        relative_error = absolute_error / max(1.0, simulated_latency)
        agreement_score = 1.0 / (1.0 + relative_error)

        total_simulated_latency += simulated_latency
        total_hardware_latency += hardware_latency
        total_absolute_error += absolute_error
        total_relative_error += relative_error
        total_agreement_score += agreement_score

    count = float(len(sorted_scenarios))
    return {
        "mean_simulated_latency": total_simulated_latency / count,
        "mean_hardware_latency": total_hardware_latency / count,
        "mean_absolute_error": total_absolute_error / count,
        "mean_relative_error": total_relative_error / count,
        "mean_agreement_score": total_agreement_score / count,
    }
