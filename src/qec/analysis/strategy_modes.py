"""v105.2.0 — Art of War strategy layer: context-aware strategy classification.

Implements strategy mode classification and selection:
    diagnosis + influence map → strategy mode → allowed actions

Strategy modes:
    defensive   — stabilize under threat
    offensive   — force state transition out of traps
    positional  — reshape landscape geometry
    deceptive   — indirect parameter shifts to bypass resistance

Strategy-conditioned action mapping:
    defensive  → boost_stability
    offensive  → force_transition
    positional → reshape_coupling
    deceptive  → indirect_shift

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- opt-in only

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

ROUND_PRECISION = 12

# ---------------------------------------------------------------------------
# Strategy mode definitions
# ---------------------------------------------------------------------------

STRATEGY_MODES = [
    "defensive",
    "offensive",
    "positional",
    "deceptive",
]

# Strategy → allowed actions mapping
STRATEGY_ACTIONS: Dict[str, List[str]] = {
    "defensive": ["boost_stability"],
    "offensive": ["force_transition"],
    "positional": ["reshape_coupling", "reshape_geometry"],
    "deceptive": ["indirect_shift", "parameter_nudge"],
}

# ---------------------------------------------------------------------------
# Thresholds for mode classification
# ---------------------------------------------------------------------------

_INSTABILITY_HIGH = 0.6
_RISK_HIGH = 0.5
_PLATEAU_THRESHOLD = 0.3
_FRAGMENTATION_THRESHOLD = 0.5
_FAILURE_PERSISTENCE_THRESHOLD = 3


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


def _safe_int(value: Any, default: int = 0) -> int:
    """Extract int from value, returning *default* on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 1. Strategy mode classification from state
# ---------------------------------------------------------------------------


def classify_strategy_mode(state: dict, registry: dict) -> str:
    """Classify current strategy mode from system state and registry.

    Parameters
    ----------
    state : dict
        System state dict with global_metrics, trajectory_geometry, etc.
    registry : dict
        Invariant registry for historical context.

    Returns
    -------
    str
        One of ``STRATEGY_MODES``.
    """
    gm = state.get("global_metrics", {}) if isinstance(state, dict) else {}
    if not isinstance(gm, dict):
        gm = {}

    stability = _safe_float(gm.get("stability", 0.5))
    convergence = _safe_float(gm.get("convergence_rate", 0.5))
    escape_rate = _safe_float(gm.get("escape_rate", 0.0))

    # Compute risk score
    risk = _round(_clamp(escape_rate + (1.0 - stability) * 0.5))

    # Count persistent failures in registry
    failure_count = 0
    if isinstance(registry, dict):
        for key in sorted(registry.keys()):
            entry = registry[key]
            if isinstance(entry, dict):
                breaks = _safe_int(entry.get("break_count", 0))
                if breaks >= _FAILURE_PERSISTENCE_THRESHOLD:
                    failure_count += 1

    # Classification rules (priority-ordered, deterministic)
    if stability < _INSTABILITY_HIGH and risk >= _RISK_HIGH:
        return "defensive"

    if convergence < _PLATEAU_THRESHOLD and stability >= _INSTABILITY_HIGH:
        return "offensive"

    if failure_count >= _FAILURE_PERSISTENCE_THRESHOLD:
        return "deceptive"

    # Check for fragmented landscape
    tg = state.get("trajectory_geometry", {}) if isinstance(state, dict) else {}
    if not isinstance(tg, dict):
        tg = {}
    coupling = _safe_float(tg.get("coupling_metrics", {}).get("coupling_strength", 0.5)
                           if isinstance(tg.get("coupling_metrics"), dict) else 0.5)
    if coupling < _FRAGMENTATION_THRESHOLD:
        return "positional"

    # Default to defensive (safest)
    return "defensive"


# ---------------------------------------------------------------------------
# 2. Strategy mode selection from diagnosis + influence map
# ---------------------------------------------------------------------------


def select_strategy_mode(
    diagnosis: dict,
    influence_map: dict,
    laws: List[dict],
) -> str:
    """Select strategy mode based on diagnosis, influence map, and laws.

    Decision rules:
    - unstable + high risk → defensive
    - trapped plateau → offensive
    - fragmented landscape → positional
    - persistent failure modes → deceptive

    Parameters
    ----------
    diagnosis : dict
        Output of differential diagnosis (ranked failure modes).
    influence_map : dict
        Output of ``compute_influence_map``.
    laws : list of dict
        Current law set for context.

    Returns
    -------
    str
        Selected strategy mode from ``STRATEGY_MODES``.
    """
    # Extract top diagnosis
    diagnoses = diagnosis.get("ranked_diagnoses", [])
    if isinstance(diagnoses, list) and diagnoses:
        top = diagnoses[0] if isinstance(diagnoses[0], dict) else {}
    else:
        top = {}

    top_mode = str(top.get("failure_mode", "unknown"))
    top_score = _safe_float(top.get("score", 0.0))

    # Extract influence summary
    summary = influence_map.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    avg_pressure = _safe_float(summary.get("avg_pressure", 0.0))

    # Extract law fragility
    fragile_law_count = 0
    if isinstance(laws, list):
        for law in laws:
            if isinstance(law, dict):
                stability_score = _safe_float(law.get("stability_score", 1.0))
                if stability_score < 0.5:
                    fragile_law_count += 1

    # Decision logic (deterministic, priority-ordered)
    if top_mode in ("oscillatory_trap", "basin_switch_instability") and avg_pressure >= _RISK_HIGH:
        return "defensive"

    if top_mode == "metastable_plateau" and top_score >= 0.5:
        return "offensive"

    if fragile_law_count >= 2:
        return "deceptive"

    if avg_pressure >= _FRAGMENTATION_THRESHOLD:
        return "positional"

    # Fallback based on pressure
    if avg_pressure >= _RISK_HIGH:
        return "defensive"

    return "defensive"


# ---------------------------------------------------------------------------
# 3. Strategy-conditioned intervention mapping
# ---------------------------------------------------------------------------


def get_allowed_actions(strategy_mode: str) -> List[str]:
    """Get allowed intervention actions for a strategy mode.

    Parameters
    ----------
    strategy_mode : str
        One of ``STRATEGY_MODES``.

    Returns
    -------
    list of str
        Allowed actions for the mode.  Returns defensive actions for
        unknown modes.
    """
    return list(STRATEGY_ACTIONS.get(strategy_mode, STRATEGY_ACTIONS["defensive"]))


def build_strategy_intervention(
    strategy_mode: str,
    strength: float = 0.2,
) -> List[dict]:
    """Build intervention candidates conditioned on strategy mode.

    Parameters
    ----------
    strategy_mode : str
        One of ``STRATEGY_MODES``.
    strength : float
        Intervention strength (default: 0.2, minimal per Wu Wei).

    Returns
    -------
    list of dict
        Candidate interventions with ``action`` and ``strength``.
    """
    actions = get_allowed_actions(strategy_mode)
    strength = _clamp(_safe_float(strength))

    interventions: List[dict] = []
    for action in sorted(actions):
        interventions.append({
            "action": action,
            "strength": _round(strength),
            "strategy_mode": strategy_mode,
        })
    return interventions


def compute_strategy_score(
    strategy_mode: str,
    influence_map: dict,
    diagnosis: dict,
) -> float:
    """Score how well a strategy mode fits the current situation.

    Parameters
    ----------
    strategy_mode : str
        Candidate strategy mode.
    influence_map : dict
        Output of ``compute_influence_map``.
    diagnosis : dict
        Output of differential diagnosis.

    Returns
    -------
    float
        Fitness score in [0, 1].
    """
    summary = influence_map.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    avg_pressure = _safe_float(summary.get("avg_pressure", 0.0))
    avg_sensitivity = _safe_float(summary.get("avg_sensitivity", 0.0))

    diagnoses = diagnosis.get("ranked_diagnoses", [])
    top_mode = ""
    if isinstance(diagnoses, list) and diagnoses:
        top = diagnoses[0] if isinstance(diagnoses[0], dict) else {}
        top_mode = str(top.get("failure_mode", ""))

    score = 0.0

    if strategy_mode == "defensive":
        score = avg_pressure * 0.6 + (1.0 if top_mode in (
            "oscillatory_trap", "basin_switch_instability"
        ) else 0.0) * 0.4

    elif strategy_mode == "offensive":
        score = (1.0 if top_mode == "metastable_plateau" else 0.0) * 0.5 + (
            1.0 - avg_pressure
        ) * 0.3 + avg_sensitivity * 0.2

    elif strategy_mode == "positional":
        score = avg_sensitivity * 0.5 + (1.0 - avg_pressure) * 0.3 + 0.2

    elif strategy_mode == "deceptive":
        score = (1.0 if top_mode in (
            "slow_convergence", "underconstrained_dynamics"
        ) else 0.0) * 0.4 + avg_sensitivity * 0.3 + avg_pressure * 0.3

    return _round(_clamp(score))
