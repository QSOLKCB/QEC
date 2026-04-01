"""Adaptive Policy Search — v134.7.0.

Deterministic policy search layer.

Given benchmark results from control_policy_benchmark,
selects the best-performing policy and generates a refined
candidate damping factor for the next iteration.

This module is:
- pure
- immutable
- deterministic
- replay-safe
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from qec.sims.control_policy_benchmark import PolicyBenchmarkResult


# ---------------------------------------------------------------------------
# Deterministic damping refinement map
# ---------------------------------------------------------------------------

_REFINEMENT_FACTORS: Tuple[Tuple[str, float], ...] = (
    ("aggressive_damping", 0.9),
    ("nominal", 0.95),
    ("recovery_first", 0.8),
)


def _refinement_factor_for(policy_name: str) -> float:
    """Return the deterministic refinement multiplier for *policy_name*.

    Unknown policies receive a neutral factor of 1.0.
    """
    for name, factor in _REFINEMENT_FACTORS:
        if name == policy_name:
            return factor
    return 1.0


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptivePolicySearchResult:
    """Immutable record of an adaptive policy search iteration."""

    best_policy_name: str
    best_score: float
    candidate_damping_factor: float
    search_iteration: int
    improvement_delta: float


# ---------------------------------------------------------------------------
# Search function
# ---------------------------------------------------------------------------

def search_adaptive_policy(
    benchmark_results: Tuple[PolicyBenchmarkResult, ...],
    *,
    search_iteration: int = 0,
    base_damping: float = 1.0,
) -> AdaptivePolicySearchResult:
    """Select the best policy and produce a refined candidate damping factor.

    Parameters
    ----------
    benchmark_results:
        Tuple of ``PolicyBenchmarkResult`` from ``run_policy_benchmark``.
        Must contain at least one result.
    search_iteration:
        Monotonic iteration counter for the search process.
    base_damping:
        Starting damping factor before refinement is applied.

    Returns
    -------
    AdaptivePolicySearchResult
        Frozen record with best policy, score, candidate damping,
        iteration index, and improvement delta.

    Raises
    ------
    ValueError
        If *benchmark_results* is empty.
    """
    if not benchmark_results:
        raise ValueError("benchmark_results must not be empty")

    # Deterministic best-policy selection.
    # Sort by (score descending, policy_name ascending) for stability.
    sorted_results: Tuple[PolicyBenchmarkResult, ...] = tuple(
        sorted(
            benchmark_results,
            key=lambda r: (-r.score, r.policy_name),
        )
    )

    best = sorted_results[0]

    # Improvement delta: difference between best and second-best score.
    if len(sorted_results) >= 2:
        improvement_delta = best.score - sorted_results[1].score
    else:
        improvement_delta = 0.0

    # Deterministic candidate damping refinement.
    factor = _refinement_factor_for(best.policy_name)
    candidate_damping = base_damping * factor

    return AdaptivePolicySearchResult(
        best_policy_name=best.policy_name,
        best_score=best.score,
        candidate_damping_factor=candidate_damping,
        search_iteration=search_iteration,
        improvement_delta=improvement_delta,
    )
