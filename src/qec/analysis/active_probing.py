"""v105.2.0 — Active probe engine with influence mapping and strategic probing.

Implements Go-inspired influence mapping over system state:
    state → influence map → region classification → probe selection

Influence mapping treats the system as a topological landscape:
    nodes = states / strategies
    edges = transitions
    weights = stability / flow metrics

Region classification:
    stable_territory       — strong attractors, high stability
    contested_regions      — high sensitivity, competing influences
    unstable_regions       — drift / oscillation zones
    neutral_regions        — low signal, minimal activity

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- opt-in only

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

ROUND_PRECISION = 12

# ---------------------------------------------------------------------------
# Influence thresholds (deterministic, non-tunable)
# ---------------------------------------------------------------------------

_STABILITY_HIGH = 0.7
_STABILITY_LOW = 0.3
_SENSITIVITY_HIGH = 0.6
_PRESSURE_HIGH = 0.5
_NEUTRAL_THRESHOLD = 0.2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* into [lo, hi]."""
    return max(lo, min(hi, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Extract float from value, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 1. Influence map computation (Go-inspired)
# ---------------------------------------------------------------------------


def compute_influence_map(state: dict) -> dict:
    """Compute influence map from system state.

    Interprets the system as a topological landscape where each node
    (basin, strategy, or region) carries influence, instability pressure,
    and control sensitivity scores.

    Parameters
    ----------
    state : dict
        System state dict.  May contain keys from system_diagnostics,
        trajectory_geometry, basin_diagnostics, etc.  Gracefully defaults
        to 0.0 for missing keys.

    Returns
    -------
    dict
        Influence map with keys:
        - ``nodes``: dict mapping node_id -> node metrics
        - ``summary``: aggregate influence statistics
        - ``influence_entropy``: float, entropy of influence distribution
    """
    nodes: Dict[str, Dict[str, float]] = {}

    # Extract basin information if available
    basins = state.get("basins", {})
    if isinstance(basins, dict):
        for basin_id in sorted(basins.keys()):
            basin = basins[basin_id]
            if not isinstance(basin, dict):
                continue
            stability = _safe_float(basin.get("stability", 0.5))
            depth = _safe_float(basin.get("depth", 0.0))
            escape_rate = _safe_float(basin.get("escape_rate", 0.0))
            nodes[str(basin_id)] = {
                "influence_score": _round(_clamp(stability * (1.0 - escape_rate))),
                "instability_pressure": _round(_clamp(escape_rate + (1.0 - stability) * 0.5)),
                "control_sensitivity": _round(_clamp(1.0 - depth) if depth <= 1.0 else 0.0),
                "stability": _round(_clamp(stability)),
                "source": "basin",
            }

    # Extract strategy information if available
    strategies = state.get("strategies", {})
    if isinstance(strategies, dict):
        for strat_id in sorted(strategies.keys()):
            strat = strategies[strat_id]
            if not isinstance(strat, dict):
                continue
            score = _safe_float(strat.get("score", 0.5))
            volatility = _safe_float(strat.get("volatility", 0.0))
            node_key = f"strategy_{strat_id}"
            nodes[node_key] = {
                "influence_score": _round(_clamp(score)),
                "instability_pressure": _round(_clamp(volatility)),
                "control_sensitivity": _round(_clamp(1.0 - score)),
                "stability": _round(_clamp(score * (1.0 - volatility))),
                "source": "strategy",
            }

    # Extract global metrics as a fallback node
    gm = state.get("global_metrics", {})
    if isinstance(gm, dict) and gm:
        convergence = _safe_float(gm.get("convergence_rate", 0.5))
        overall_stability = _safe_float(gm.get("stability", 0.5))
        nodes["global"] = {
            "influence_score": _round(_clamp(overall_stability)),
            "instability_pressure": _round(_clamp(1.0 - convergence)),
            "control_sensitivity": _round(_clamp(1.0 - overall_stability)),
            "stability": _round(_clamp(overall_stability)),
            "source": "global",
        }

    # Compute influence entropy (deterministic)
    influence_entropy = _compute_influence_entropy(nodes)

    # Compute summary statistics
    summary = _compute_influence_summary(nodes)

    return {
        "nodes": nodes,
        "summary": summary,
        "influence_entropy": influence_entropy,
    }


def _compute_influence_entropy(nodes: Dict[str, Dict[str, float]]) -> float:
    """Compute Shannon entropy of influence score distribution.

    Returns 0.0 for empty or uniform-zero distributions.
    """
    if not nodes:
        return 0.0

    scores = []
    for key in sorted(nodes.keys()):
        scores.append(max(nodes[key].get("influence_score", 0.0), 0.0))

    total = sum(scores)
    if total < 1e-15:
        return 0.0

    import math

    entropy = 0.0
    for s in scores:
        p = s / total
        if p > 1e-15:
            entropy -= p * math.log(p)
    return _round(entropy)


def _compute_influence_summary(nodes: Dict[str, Dict[str, float]]) -> dict:
    """Compute aggregate statistics over influence map nodes."""
    if not nodes:
        return {
            "node_count": 0,
            "avg_influence": 0.0,
            "avg_pressure": 0.0,
            "avg_sensitivity": 0.0,
            "max_pressure_node": "",
            "max_pressure": 0.0,
        }

    total_influence = 0.0
    total_pressure = 0.0
    total_sensitivity = 0.0
    max_pressure = 0.0
    max_pressure_node = ""

    for key in sorted(nodes.keys()):
        n = nodes[key]
        total_influence += n.get("influence_score", 0.0)
        total_pressure += n.get("instability_pressure", 0.0)
        total_sensitivity += n.get("control_sensitivity", 0.0)
        pressure = n.get("instability_pressure", 0.0)
        if pressure > max_pressure or (pressure == max_pressure and key < max_pressure_node):
            max_pressure = pressure
            max_pressure_node = key

    count = len(nodes)
    return {
        "node_count": count,
        "avg_influence": _round(total_influence / count),
        "avg_pressure": _round(total_pressure / count),
        "avg_sensitivity": _round(total_sensitivity / count),
        "max_pressure_node": max_pressure_node,
        "max_pressure": _round(max_pressure),
    }


# ---------------------------------------------------------------------------
# 2. Region classification (Go-inspired territory mapping)
# ---------------------------------------------------------------------------


def classify_regions(influence_map: dict) -> dict:
    """Classify influence map nodes into territory regions.

    Parameters
    ----------
    influence_map : dict
        Output of ``compute_influence_map``.

    Returns
    -------
    dict
        Keys: ``stable_territory``, ``contested_regions``,
        ``unstable_regions``, ``neutral_regions``.
        Each maps to a list of node_id strings.
        Also includes ``contested_region_count``: int.
    """
    nodes = influence_map.get("nodes", {})

    stable: List[str] = []
    contested: List[str] = []
    unstable: List[str] = []
    neutral: List[str] = []

    for node_id in sorted(nodes.keys()):
        n = nodes[node_id]
        stability = n.get("stability", 0.0)
        pressure = n.get("instability_pressure", 0.0)
        sensitivity = n.get("control_sensitivity", 0.0)
        influence = n.get("influence_score", 0.0)

        # Classification logic (deterministic, priority-ordered)
        if stability >= _STABILITY_HIGH and pressure < _PRESSURE_HIGH:
            stable.append(node_id)
        elif sensitivity >= _SENSITIVITY_HIGH and pressure >= _PRESSURE_HIGH:
            contested.append(node_id)
        elif stability < _STABILITY_LOW or pressure >= _PRESSURE_HIGH:
            unstable.append(node_id)
        elif influence < _NEUTRAL_THRESHOLD and pressure < _NEUTRAL_THRESHOLD:
            neutral.append(node_id)
        else:
            # Default: classify based on net stability
            if stability >= 0.5:
                stable.append(node_id)
            else:
                unstable.append(node_id)

    return {
        "stable_territory": stable,
        "contested_regions": contested,
        "unstable_regions": unstable,
        "neutral_regions": neutral,
        "contested_region_count": len(contested),
    }


# ---------------------------------------------------------------------------
# 3. Strategic probe selection
# ---------------------------------------------------------------------------


def select_probes(
    state: dict,
    registry: dict,
    k: int = 3,
) -> List[dict]:
    """Select strategic probes targeting high-information regions.

    Prioritizes contested and unstable regions.  Avoids redundant probing
    of already-well-characterized regions (stable territory with high
    registry counts).

    Parameters
    ----------
    state : dict
        System state dict.
    registry : dict
        Invariant registry for assessing what is already known.
    k : int
        Maximum number of probes to select.

    Returns
    -------
    list of dict
        Each probe dict contains:
        - ``target_node``: str — node to probe
        - ``probe_type``: str — type of probe
        - ``priority``: float — probe priority score
        - ``reason``: str — why this probe was selected
    """
    if k <= 0:
        return []

    influence_map = compute_influence_map(state)
    regions = classify_regions(influence_map)
    nodes = influence_map.get("nodes", {})

    candidates: List[Tuple[float, str, str, str]] = []

    # Contested regions get highest priority
    for node_id in regions.get("contested_regions", []):
        node = nodes.get(node_id, {})
        priority = _round(0.9 + node.get("control_sensitivity", 0.0) * 0.1)
        candidates.append((priority, node_id, "sensitivity_probe", "contested_region"))

    # Unstable regions get second priority
    for node_id in regions.get("unstable_regions", []):
        node = nodes.get(node_id, {})
        priority = _round(0.6 + node.get("instability_pressure", 0.0) * 0.3)
        candidates.append((priority, node_id, "instability_probe", "unstable_region"))

    # Neutral regions get lowest priority (information discovery)
    for node_id in regions.get("neutral_regions", []):
        candidates.append((0.3, node_id, "discovery_probe", "neutral_region"))

    # Filter out already-well-known nodes (registry has high count)
    filtered: List[Tuple[float, str, str, str]] = []
    for priority, node_id, probe_type, reason in candidates:
        reg_key = f"probe:{node_id}"
        entry = registry.get(reg_key, {})
        count = int(entry.get("count", 0)) if isinstance(entry, dict) else 0
        if count < 5:
            # Boost priority for less-known nodes
            adjusted = _round(priority + (5 - count) * 0.02)
            filtered.append((adjusted, node_id, probe_type, reason))
        else:
            # Reduce priority for well-known nodes
            adjusted = _round(priority * 0.5)
            filtered.append((adjusted, node_id, probe_type, reason))

    # Sort by priority descending, then node_id for determinism
    filtered.sort(key=lambda x: (-x[0], x[1]))

    # Select top k probes
    probes: List[dict] = []
    seen_nodes: set = set()
    for priority, node_id, probe_type, reason in filtered:
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        probes.append({
            "target_node": node_id,
            "probe_type": probe_type,
            "priority": _round(priority),
            "reason": reason,
        })
        if len(probes) >= k:
            break

    return probes
