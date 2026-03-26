"""v103.6.0 — Policy refinement and deterministic threshold optimization.

Provides:
- deterministic policy variant generation (threshold neighborhoods)
- variant evaluation via hierarchical control
- best variant selection (score DESC, steps ASC, name ASC)
- iterative refinement loop with evolution tracking

Enables meta-optimization of control rules: instead of fixed thresholds,
the system explores small deterministic neighborhoods and selects
improved variants.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs (thresholds clamped to [0.0, 1.0])
- rule-based only (no stochastic search, no learning)

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from qec.analysis.policy import Policy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REFINEMENT_DELTA = 0.1
REFINEMENT_MAX_ITERS = 3
THRESHOLD_MIN = 0.0
THRESHOLD_MAX = 1.0
ROUND_PRECISION = 12


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round(value: float) -> float:
    """Round to ``ROUND_PRECISION`` decimal places."""
    return round(float(value), ROUND_PRECISION)


def _clamp(value: float) -> float:
    """Clamp *value* to [THRESHOLD_MIN, THRESHOLD_MAX] and round."""
    return _round(max(THRESHOLD_MIN, min(THRESHOLD_MAX, value)))


# ---------------------------------------------------------------------------
# Public API — Variant Generation
# ---------------------------------------------------------------------------


def generate_policy_variants(
    policy: Policy,
    delta: float = REFINEMENT_DELTA,
) -> List[Policy]:
    """Generate a deterministic neighborhood of policy variants.

    For each threshold key, produces three values:
    ``{value - delta, value, value + delta}``, clamped to [0.0, 1.0].

    The total number of variants is ``3^n`` where *n* is the number of
    threshold keys. Each variant receives a deterministic name derived
    from the base policy name and the threshold perturbations applied.

    Parameters
    ----------
    policy : Policy
        Base policy to generate variants from. Not mutated.
    delta : float
        Perturbation step size (default: 0.1).

    Returns
    -------
    list of Policy
        Sorted deterministically by name. Always includes the base
        policy (with all offsets = 0) as one of the variants.
    """
    threshold_keys = sorted(policy.thresholds.keys())

    if not threshold_keys:
        return [
            Policy(
                name=policy.name,
                mode=policy.mode,
                priority=policy.priority,
                thresholds=dict(policy.thresholds),
            )
        ]

    # Build all combinations of offsets deterministically.
    offsets = [-delta, 0.0, delta]
    combinations: List[List[float]] = [[]]
    for _ in threshold_keys:
        new_combinations: List[List[float]] = []
        for combo in combinations:
            for offset in offsets:
                new_combinations.append(combo + [offset])
        combinations = new_combinations

    variants: List[Policy] = []
    for combo in combinations:
        new_thresholds: Dict[str, float] = {}
        suffix_parts: List[str] = []
        for i, key in enumerate(threshold_keys):
            base_val = float(policy.thresholds[key])
            new_val = _clamp(base_val + combo[i])
            new_thresholds[key] = new_val

            # Build a readable suffix.
            if combo[i] < 0:
                suffix_parts.append(f"{key}-")
            elif combo[i] > 0:
                suffix_parts.append(f"{key}+")

        if suffix_parts:
            suffix = "_" + "_".join(suffix_parts)
        else:
            suffix = ""

        variant_name = f"{policy.name}{suffix}"

        variants.append(
            Policy(
                name=variant_name,
                mode=policy.mode,
                priority=policy.priority,
                thresholds=new_thresholds,
            )
        )

    # Sort deterministically by name.
    variants.sort(key=lambda p: p.name)
    return variants


# ---------------------------------------------------------------------------
# Public API — Variant Evaluation
# ---------------------------------------------------------------------------


def evaluate_policy_variants(
    runs: List[Dict[str, Any]],
    variants: List[Policy],
    objective: Dict[str, Any],
    *,
    multistate_result: Optional[Dict[str, Any]] = None,
    coupled_result: Optional[Dict[str, Any]] = None,
    max_steps: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate each policy variant via hierarchical control.

    For each variant, runs the full hierarchical control loop and
    collects the final score, steps taken, and convergence status.

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key.
    variants : list of Policy
        Policy variants to evaluate.
    objective : dict
        Global objective weights.
    multistate_result : dict, optional
        Precomputed multistate result.
    coupled_result : dict, optional
        Precomputed coupled dynamics result.
    max_steps : int
        Maximum hierarchical control iterations per variant.

    Returns
    -------
    dict
        Mapping from variant name to ``{"score": float, "steps": int}``.
        Sorted deterministically by variant name.
    """
    from qec.analysis.hierarchical_control import run_hierarchical_control
    from qec.analysis.strategy_adapter import (
        run_coupled_dynamics_analysis,
        run_multistate_analysis,
    )

    if multistate_result is None:
        multistate_result = run_multistate_analysis(runs)
    if coupled_result is None:
        coupled_result = run_coupled_dynamics_analysis(
            runs, multistate_result=multistate_result,
        )

    results: Dict[str, Dict[str, Any]] = {}

    for variant in sorted(variants, key=lambda p: p.name):
        policy_dict = variant.to_dict()

        hc_result = run_hierarchical_control(
            runs,
            objective,
            policy_dict,
            max_steps=max_steps,
            multistate_result=multistate_result,
            coupled_result=coupled_result,
        )

        scores = hc_result.get("scores", [0.0])
        final_score = _round(scores[-1]) if scores else 0.0
        steps = int(hc_result.get("steps_taken", 0))

        results[variant.name] = {
            "score": final_score,
            "steps": steps,
        }

    return results


# ---------------------------------------------------------------------------
# Public API — Best Variant Selection
# ---------------------------------------------------------------------------


def select_best_variant(
    results: Dict[str, Dict[str, Any]],
    variants: List[Policy],
) -> Policy:
    """Select the best variant from evaluation results.

    Selection rule (deterministic):
    - Highest score wins.
    - Tie: fewer steps (faster convergence preferred).
    - Still tied: lexicographic variant name order.

    Parameters
    ----------
    results : dict
        Output of ``evaluate_policy_variants``.
    variants : list of Policy
        The evaluated variants (used to return the Policy object).

    Returns
    -------
    Policy
        The best variant.

    Raises
    ------
    ValueError
        If *results* is empty.
    """
    if not results:
        raise ValueError("Cannot select from empty results")

    variant_map = {v.name: v for v in variants}
    candidates = sorted(results.keys())

    best_name = candidates[0]
    best_score = float(results[best_name].get("score", 0.0))
    best_steps = int(results[best_name].get("steps", 0))

    for name in candidates[1:]:
        score = float(results[name].get("score", 0.0))
        steps = int(results[name].get("steps", 0))

        if (
            score > best_score
            or (score == best_score and steps < best_steps)
            or (
                score == best_score
                and steps == best_steps
                and name < best_name
            )
        ):
            best_name = name
            best_score = score
            best_steps = steps

    return variant_map[best_name]


# ---------------------------------------------------------------------------
# Public API — Refinement Loop
# ---------------------------------------------------------------------------


def refine_policy(
    runs: List[Dict[str, Any]],
    base_policy: Policy,
    objective: Dict[str, Any],
    max_iters: int = REFINEMENT_MAX_ITERS,
    *,
    delta: float = REFINEMENT_DELTA,
    multistate_result: Optional[Dict[str, Any]] = None,
    coupled_result: Optional[Dict[str, Any]] = None,
    max_steps: int = 5,
) -> Dict[str, Any]:
    """Run the iterative policy refinement loop.

    Pipeline::

        policy
        -> generate variants
        -> evaluate
        -> select best
        -> repeat (up to max_iters)

    Tracks the evolution of policies and scores across iterations.

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key.
    base_policy : Policy
        Starting policy. Not mutated.
    objective : dict
        Global objective weights.
    max_iters : int
        Maximum refinement iterations (default: 3).
    delta : float
        Perturbation step size for variant generation.
    multistate_result : dict, optional
        Precomputed multistate result.
    coupled_result : dict, optional
        Precomputed coupled dynamics result.
    max_steps : int
        Maximum hierarchical control iterations per variant evaluation.

    Returns
    -------
    dict
        Contains:

        - ``"policies"`` : list of Policy — best policy at each iteration
        - ``"scores"`` : list of float — best score at each iteration
        - ``"best_policy"`` : Policy — final best policy
        - ``"iterations"`` : int — number of iterations executed
        - ``"improved"`` : bool — whether refinement improved over base
    """
    from qec.analysis.strategy_adapter import (
        run_coupled_dynamics_analysis,
        run_multistate_analysis,
    )

    if multistate_result is None:
        multistate_result = run_multistate_analysis(runs)
    if coupled_result is None:
        coupled_result = run_coupled_dynamics_analysis(
            runs, multistate_result=multistate_result,
        )

    current_policy = Policy(
        name=base_policy.name,
        mode=base_policy.mode,
        priority=base_policy.priority,
        thresholds=dict(base_policy.thresholds),
    )

    policies: List[Policy] = []
    scores: List[float] = []

    for iteration in range(max_iters):
        # Generate variants around current policy.
        variants = generate_policy_variants(current_policy, delta=delta)

        # Evaluate all variants.
        results = evaluate_policy_variants(
            runs,
            variants,
            objective,
            multistate_result=multistate_result,
            coupled_result=coupled_result,
            max_steps=max_steps,
        )

        # Select best variant.
        best = select_best_variant(results, variants)
        best_score = float(results[best.name]["score"])

        policies.append(best)
        scores.append(_round(best_score))

        # Update current policy for next iteration.
        current_policy = Policy(
            name=base_policy.name,
            mode=best.mode,
            priority=best.priority,
            thresholds=dict(best.thresholds),
        )

    # Determine if refinement improved over the base.
    base_score = scores[0] if scores else 0.0
    final_score = scores[-1] if scores else 0.0
    improved = final_score > base_score

    return {
        "policies": policies,
        "scores": scores,
        "best_policy": current_policy,
        "iterations": len(policies),
        "improved": improved,
    }


# ---------------------------------------------------------------------------
# Public API — Formatting
# ---------------------------------------------------------------------------


def format_policy_refinement_summary(result: Dict[str, Any]) -> str:
    """Format policy refinement results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``refine_policy``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    lines.append("=== Policy Refinement ===")

    best_policy = result.get("best_policy")
    if best_policy is not None:
        lines.append("")
        lines.append(f"Base: {best_policy.name}")

    policies = result.get("policies", [])
    scores = result.get("scores", [])

    for i in range(len(scores)):
        lines.append("")
        lines.append(f"Iteration {i + 1}:")
        if i == 0:
            lines.append(f"  Score: {scores[i]:.2f}")
        else:
            lines.append(f"  Score: {scores[i - 1]:.2f} -> {scores[i]:.2f}")

    # Show final best policy thresholds.
    if best_policy is not None:
        lines.append("")
        lines.append("Best Policy:")
        for key in sorted(best_policy.thresholds.keys()):
            val = best_policy.thresholds[key]
            lines.append(f"  {key}_threshold={val:.1f}")

    improved = result.get("improved", False)
    lines.append("")
    if improved:
        lines.append("Refinement improved policy.")
    else:
        lines.append("Refinement did not improve policy (base is optimal).")

    return "\n".join(lines)


__all__ = [
    "REFINEMENT_DELTA",
    "REFINEMENT_MAX_ITERS",
    "THRESHOLD_MAX",
    "THRESHOLD_MIN",
    "evaluate_policy_variants",
    "format_policy_refinement_summary",
    "generate_policy_variants",
    "refine_policy",
    "select_best_variant",
]
