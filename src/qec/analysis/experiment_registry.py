"""v103.4.0 — Experiment registry for policy comparison.

Provides:
- run_policy_experiment: run hierarchical control for each policy, collect results
- rank_policies: sort policies by score (DESC) then steps (ASC)
- format_policy_experiment_summary: human-readable experiment output

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- rule-based only (no stochastic routing, no learning)

Dependencies: stdlib only (plus internal qec.analysis modules).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from qec.analysis.policy import Policy


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_policy_experiment(
    runs: List[Dict[str, Any]],
    policies: List[Policy],
    objective: Dict[str, Any],
    *,
    max_steps: int = 5,
    multistate_result: Any = None,
    coupled_result: Any = None,
) -> Dict[str, Dict[str, Any]]:
    """Run hierarchical control for each policy and collect results.

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key.
    policies : list of Policy
        Policies to evaluate.
    objective : dict
        Global objective weights.
    max_steps : int
        Maximum hierarchical feedback iterations per policy.
    multistate_result : dict, optional
        Precomputed multistate result (shared across policies).
    coupled_result : dict, optional
        Precomputed coupled dynamics result (shared across policies).

    Returns
    -------
    dict
        Mapping from policy name to result dict with keys:
        ``"score"``, ``"steps"``, ``"convergence"``.
    """
    from qec.analysis.hierarchical_control import run_hierarchical_control
    from qec.analysis.strategy_adapter import (
        run_coupled_dynamics_analysis,
        run_multistate_analysis,
    )

    # Precompute shared state once.
    if multistate_result is None:
        multistate_result = run_multistate_analysis(runs)
    if coupled_result is None:
        coupled_result = run_coupled_dynamics_analysis(
            runs,
            multistate_result=multistate_result,
        )

    results: Dict[str, Dict[str, Any]] = {}

    for policy in policies:
        policy_dict = policy.to_dict()

        hc_result = run_hierarchical_control(
            runs,
            objective,
            policy_dict,
            max_steps=max_steps,
            multistate_result=multistate_result,
            coupled_result=coupled_result,
        )

        scores = hc_result.get("scores", [])
        final_score = scores[-1] if scores else 0.0
        convergence = hc_result.get("convergence", {})
        convergence_type = convergence.get("type", "max_steps")
        steps_taken = hc_result.get("steps_taken", 0)

        results[policy.name] = {
            "score": round(float(final_score), 12),
            "steps": int(steps_taken),
            "convergence": str(convergence_type),
        }

    return results


# ---------------------------------------------------------------------------
# Policy ranking
# ---------------------------------------------------------------------------


def rank_policies(results: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Rank policies by score (descending) then steps (ascending).

    Parameters
    ----------
    results : dict
        Output of ``run_policy_experiment``.

    Returns
    -------
    list of (name, result_dict)
        Sorted by score DESC, then steps ASC.
    """
    items = list(results.items())
    items.sort(key=lambda x: (-x[1].get("score", 0.0), x[1].get("steps", 0)))
    return items


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_policy_experiment_summary(
    results: Dict[str, Dict[str, Any]],
) -> str:
    """Format policy experiment results as a human-readable summary.

    Parameters
    ----------
    results : dict
        Output of ``run_policy_experiment``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    lines.append("=== Policy Experiment ===")
    lines.append("")

    ranked = rank_policies(results)

    for name, data in ranked:
        score = data.get("score", 0.0)
        steps = data.get("steps", 0)
        convergence = data.get("convergence", "unknown")
        lines.append(f"Policy: {name}")
        lines.append(
            f"  Score: {score:.4f} | Steps: {steps} | "
            f"Convergence: {convergence}"
        )
        lines.append("")

    if ranked:
        best_name = ranked[0][0]
        lines.append(f"Best Policy: {best_name}")
    else:
        lines.append("No policies evaluated.")

    return "\n".join(lines)
