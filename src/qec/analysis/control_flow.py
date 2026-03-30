"""v105.4.1 — Deterministic predictive control flow layer.

Consumes outputs from run_collapse_analysis() and produces control
signals for higher-level strategy modulation.

Pure analysis + signal generation only (no decoder integration).

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- stdlib only

Dependencies: none (stdlib only).
"""

from __future__ import annotations

from typing import TypedDict


class CollapseResult(TypedDict, total=False):
    collapse_score: float
    spike_density: float
    basin_switch_prediction: bool

ROUND_PRECISION = 12


def compute_damping_factor(collapse_score: float) -> float:
    """Return damping factor based on collapse score.

    stable (<=0.3) -> 1.0, moderate (<=0.6) -> 0.8, high -> 0.6.
    """
    if collapse_score <= 0.3:
        return 1.0
    elif collapse_score <= 0.6:
        return 0.8
    else:
        return 0.6


def compute_step_aggressiveness(spike_density: float) -> float:
    """Return step aggressiveness clamped to [0.5, 1.0]."""
    return min(1.0, max(0.5, 1.0 - spike_density))


def compute_strategy_escalation(basin_switch_prediction: bool) -> str:
    """Return 'escalate' if basin switch predicted, else 'hold'."""
    return "escalate" if basin_switch_prediction else "hold"


def run_control_flow(collapse_result: CollapseResult) -> dict:
    """Produce control signal from collapse analysis output.

    Parameters
    ----------
    collapse_result : CollapseResult
        May contain keys: collapse_score, spike_density,
        basin_switch_prediction.  Missing keys default safely.

    Returns
    -------
    dict with keys: damping_factor, step_aggressiveness,
        strategy_action, control_stability_score.
    """
    collapse_score = collapse_result.get("collapse_score", 0.0)
    spike_density = collapse_result.get("spike_density", 0.0)
    basin_switch = collapse_result.get("basin_switch_prediction", False)

    damping = compute_damping_factor(collapse_score)
    aggressiveness = compute_step_aggressiveness(spike_density)
    action = compute_strategy_escalation(basin_switch)
    stability = round(0.5 * damping + 0.5 * aggressiveness, ROUND_PRECISION)

    return {
        "damping_factor": damping,
        "step_aggressiveness": aggressiveness,
        "strategy_action": action,
        "control_stability_score": stability,
    }
