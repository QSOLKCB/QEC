"""v138.7.4.3 — deterministic governance feedback for validation thresholds."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


_REQUIRED_CONFIG_KEYS = (
    "max_mean_relative_error",
    "max_mean_absolute_error",
    "min_mean_agreement_score",
)
_VALID_STATUSES = {"PASS", "WARN", "FAIL"}


def _validated_config(config: Mapping[str, float | int]) -> tuple[float, float, float]:
    if not isinstance(config, Mapping):
        raise ValueError("config must be mapping-like")

    for key in _REQUIRED_CONFIG_KEYS:
        if key not in config:
            raise ValueError(f"config missing required key '{key}'")

    values: dict[str, float] = {}
    for key in _REQUIRED_CONFIG_KEYS:
        raw_value = config[key]
        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            raise ValueError(f"config value for '{key}' must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"config value for '{key}' must be finite")
        values[key] = value

    r = values["max_mean_relative_error"]
    a = values["max_mean_absolute_error"]
    g = values["min_mean_agreement_score"]

    if r < 0.0:
        raise ValueError("max_mean_relative_error must be >= 0")
    if a < 0.0:
        raise ValueError("max_mean_absolute_error must be >= 0")
    if not 0.0 <= g <= 1.0:
        raise ValueError("min_mean_agreement_score must be between 0 and 1")

    return r, a, g


def _validated_status(governance_result: Mapping[str, Any]) -> str:
    if not isinstance(governance_result, Mapping):
        raise ValueError("governance_result must be mapping-like")

    if "status" not in governance_result:
        raise ValueError("governance_result must include a 'status' field")

    status = governance_result["status"]
    if status is None:
        raise ValueError("governance_result status must not be None")

    if status not in _VALID_STATUSES:
        raise ValueError("governance_result status must be one of: PASS, WARN, FAIL")

    return status


def update_hardware_validation_config(
    config: Mapping[str, float | int],
    governance_result: Mapping[str, Any],
) -> dict[str, float]:
    r, a, g = _validated_config(config)
    status = _validated_status(governance_result)

    if status == "FAIL":
        r_new = r * 0.8
        a_new = a * 0.8
        g_new = min(1.0, g * 1.1)
    elif status == "WARN":
        r_new = r * 0.9
        a_new = a * 0.9
        g_new = min(1.0, g * 1.05)
    else:
        r_new = r * 1.05
        a_new = a * 1.05
        g_new = max(0.0, g * 0.98)

    if not math.isfinite(r_new) or not math.isfinite(a_new) or not math.isfinite(g_new):
        raise ValueError("updated thresholds must remain finite")

    if r_new < 0.0:
        raise ValueError("max_mean_relative_error post-condition violated")
    if a_new < 0.0:
        raise ValueError("max_mean_absolute_error post-condition violated")
    if not 0.0 <= g_new <= 1.0:
        raise ValueError("min_mean_agreement_score post-condition violated")

    return {
        "max_mean_relative_error": round(float(r_new), 12),
        "max_mean_absolute_error": round(float(a_new), 12),
        "min_mean_agreement_score": round(float(g_new), 12),
    }
