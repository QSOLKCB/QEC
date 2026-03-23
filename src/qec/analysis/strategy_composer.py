"""Deterministic Strategy Composer (v98.0.0).

Combines multiple strategies into a single composite strategy.
Moves from "select ONE strategy" to "compose MANY strategies".

Composition rules:
- Numeric parameters: deterministic weighted average by confidence
- Discrete parameters: resolve via confidence -> agreement -> lexicographic
- Partial agreement: some actions merge, others resolve conflicts

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

_NORM_DIGITS = 12

# Parameters known to be numeric (eligible for blending)
NUMERIC_PARAMS = {"alpha", "weight", "threshold"}

# Parameters known to be discrete (mode selection)
DISCRETE_PARAMS = {"mode"}


def _norm(x: float) -> float:
    """Normalize a float to fixed precision to avoid drift."""
    return round(float(x), _NORM_DIGITS)


# ---------------------------------------------------------------------------
# STEP 1 — GROUP STRATEGIES BY ACTION TYPE
# ---------------------------------------------------------------------------


def group_by_action_type(
    strategies: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group strategy dicts by their action_type.

    Each strategy dict must have: action_type, params, confidence.
    Returns {action_type: [strategies]} with deterministic ordering.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in strategies:
        at = s["action_type"]
        if at not in groups:
            groups[at] = []
        groups[at].append(s)
    return groups


# ---------------------------------------------------------------------------
# STEP 2 — BLEND NUMERIC PARAMETERS
# ---------------------------------------------------------------------------


def blend_numeric(
    values: List[float],
    weights: List[float],
) -> float:
    """Deterministic weighted average of numeric values.

    If all weights are zero, returns simple arithmetic mean.
    """
    if not values:
        return 0.0
    w = np.array(weights, dtype=np.float64)
    v = np.array(values, dtype=np.float64)
    total_w = float(np.sum(w))
    if total_w <= 0.0:
        return _norm(float(np.mean(v)))
    return _norm(float(np.dot(w, v) / total_w))


# ---------------------------------------------------------------------------
# STEP 3 — RESOLVE DISCRETE PARAMETERS
# ---------------------------------------------------------------------------


def resolve_discrete(
    values: List[str],
    confidences: List[float],
) -> str:
    """Resolve discrete parameter via confidence -> agreement -> lexicographic.

    1. Group by value, sum confidences per group
    2. Pick group with highest total confidence
    3. Ties broken by count (agreement)
    4. Further ties broken lexicographically
    """
    if not values:
        return ""

    # Aggregate confidence and count per value
    agg: Dict[str, Tuple[float, int]] = {}
    for val, conf in zip(values, confidences):
        if val not in agg:
            agg[val] = (0.0, 0)
        prev_conf, prev_count = agg[val]
        agg[val] = (_norm(prev_conf + conf), prev_count + 1)

    # Sort by (-confidence, -count, value) for deterministic resolution
    ranked = sorted(
        agg.items(),
        key=lambda item: (-item[1][0], -item[1][1], item[0]),
    )
    return ranked[0][0]


# ---------------------------------------------------------------------------
# STEP 4 — COMPOSE PARAMETERS FOR ONE ACTION TYPE
# ---------------------------------------------------------------------------


def _compose_params(
    strategies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compose parameters from multiple strategies of the same action type.

    For each parameter:
    - If numeric: blend via weighted average
    - If discrete: resolve via confidence -> agreement -> lexicographic
    """
    if not strategies:
        return {}

    # Collect all parameter keys (deterministic order)
    all_keys: Dict[str, bool] = {}
    for s in strategies:
        for k in s.get("params", {}):
            all_keys[k] = True
    param_keys = sorted(all_keys.keys())

    result: Dict[str, Any] = {}
    for key in param_keys:
        values = []
        confidences = []
        for s in strategies:
            if key in s.get("params", {}):
                values.append(s["params"][key])
                confidences.append(float(s.get("confidence", 0.0)))

        if not values:
            continue

        # Determine if numeric or discrete
        if key in NUMERIC_PARAMS and all(
            isinstance(v, (int, float)) for v in values
        ):
            result[key] = blend_numeric(
                [float(v) for v in values], confidences
            )
        elif key in DISCRETE_PARAMS or any(
            isinstance(v, str) for v in values
        ):
            result[key] = resolve_discrete(
                [str(v) for v in values], confidences
            )
        else:
            # Fallback: treat as numeric if all are numbers
            if all(isinstance(v, (int, float)) for v in values):
                result[key] = blend_numeric(
                    [float(v) for v in values], confidences
                )
            else:
                result[key] = resolve_discrete(
                    [str(v) for v in values], confidences
                )
    return result


# ---------------------------------------------------------------------------
# STEP 5 — BUILD COMPOSITE STRATEGY
# ---------------------------------------------------------------------------


def _build_composite(
    action_type: str,
    strategies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a single composite strategy for one action type."""
    params = _compose_params(strategies)
    source_ids = sorted(set(s.get("law_id", "") for s in strategies))
    total_confidence = _norm(
        sum(float(s.get("confidence", 0.0)) for s in strategies)
        / len(strategies)
    )
    return {
        "action_type": action_type,
        "params": params,
        "source_ids": source_ids,
        "confidence": total_confidence,
        "component_count": len(strategies),
    }


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------


def compose_strategies(
    strategies: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compose multiple strategies into composite strategies.

    Input: list of strategy dicts, each with:
        - action_type: str
        - params: dict
        - confidence: float
        - law_id: str (optional)

    Output: list of composite strategy dicts, one per action_type,
    sorted by action_type for deterministic ordering.

    Steps:
    1. Group strategies by action_type
    2. For each group, compose parameters
    3. Return composite strategies
    """
    if not strategies:
        return []

    groups = group_by_action_type(strategies)
    composites: List[Dict[str, Any]] = []

    for action_type in sorted(groups.keys()):
        group = groups[action_type]
        composite = _build_composite(action_type, group)
        composites.append(composite)

    return composites
