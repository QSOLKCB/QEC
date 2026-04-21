"""v138.7.4.2 — deterministic hardware validation governance layer."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .hardware_validation_bridge import run_hardware_validation_bridge


def _validate_scenarios(scenarios: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
        raise ValueError("scenarios must be a non-empty sequence")

    scenarios_list = list(scenarios)
    if not scenarios_list:
        raise ValueError("scenarios must be a non-empty sequence")

    scenario_ids: list[str] = []
    for scenario in scenarios_list:
        if not isinstance(scenario, Mapping):
            raise ValueError("scenario must be mapping-like")
        scenario_id = scenario.get("id")
        if not isinstance(scenario_id, str):
            raise ValueError("scenario id must be str")
        scenario_ids.append(scenario_id)

    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario ids must be unique")

    return scenarios_list


def _validate_hardware_measurements(
    hardware_measurements: Mapping[str, Mapping[str, float]],
    scenario_ids: Sequence[str],
) -> None:
    if not isinstance(hardware_measurements, Mapping):
        raise ValueError("hardware_measurements must be mapping-like")

    for scenario_id in scenario_ids:
        if scenario_id not in hardware_measurements:
            raise ValueError(f"hardware_measurements missing scenario id '{scenario_id}'")


def _validate_config(config: Mapping[str, float]) -> tuple[float, float, float]:
    if not isinstance(config, Mapping):
        raise ValueError("config must be mapping-like")

    required_keys = (
        "max_mean_relative_error",
        "max_mean_absolute_error",
        "min_mean_agreement_score",
    )
    for key in required_keys:
        if key not in config:
            raise ValueError(f"config missing required key '{key}'")

    values: dict[str, float] = {}
    for key in required_keys:
        value = config[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"config value for '{key}' must be numeric")
        value_float = float(value)
        if not math.isfinite(value_float):
            raise ValueError(f"config value for '{key}' must be finite")
        values[key] = value_float

    max_mean_relative_error = values["max_mean_relative_error"]
    max_mean_absolute_error = values["max_mean_absolute_error"]
    min_mean_agreement_score = values["min_mean_agreement_score"]

    if max_mean_relative_error < 0.0:
        raise ValueError("max_mean_relative_error must be >= 0")
    if max_mean_absolute_error < 0.0:
        raise ValueError("max_mean_absolute_error must be >= 0")
    if not 0.0 <= min_mean_agreement_score <= 1.0:
        raise ValueError("min_mean_agreement_score must be between 0 and 1")

    return max_mean_relative_error, max_mean_absolute_error, min_mean_agreement_score


def run_hardware_validation_governance(
    scenarios: Sequence[Mapping[str, Any]],
    hardware_measurements: Mapping[str, Mapping[str, float]],
    config: Mapping[str, float],
) -> dict[str, Any]:
    scenarios_list = _validate_scenarios(scenarios)
    scenario_ids = [str(scenario["id"]) for scenario in scenarios_list]
    _validate_hardware_measurements(hardware_measurements, scenario_ids)
    max_mean_relative_error, max_mean_absolute_error, min_mean_agreement_score = (
        _validate_config(config)
    )

    bridge_result = run_hardware_validation_bridge(scenarios_list, hardware_measurements)

    mean_relative_error = float(bridge_result["mean_relative_error"])
    mean_absolute_error = float(bridge_result["mean_absolute_error"])
    mean_agreement_score = float(bridge_result["mean_agreement_score"])

    is_pass = (
        mean_relative_error <= max_mean_relative_error
        and mean_absolute_error <= max_mean_absolute_error
        and mean_agreement_score >= min_mean_agreement_score
    )

    exceeds_double_threshold = (
        mean_relative_error > (2.0 * max_mean_relative_error)
        or mean_absolute_error > (2.0 * max_mean_absolute_error)
        or mean_agreement_score < (0.5 * min_mean_agreement_score)
    )

    if is_pass:
        status = "PASS"
    elif exceeds_double_threshold:
        status = "FAIL"
    else:
        status = "WARN"

    return {
        "status": status,
        "certified": status == "PASS",
        "mean_relative_error": mean_relative_error,
        "mean_absolute_error": mean_absolute_error,
        "mean_agreement_score": mean_agreement_score,
    }
