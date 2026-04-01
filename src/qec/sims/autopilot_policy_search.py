# SPDX-License-Identifier: MIT
"""Deterministic autopilot route policy search.

Evaluates candidate propulsion schedules against a universe snapshot,
scores each route using additive deterministic scoring, and selects
the best policy with deterministic tie-breaking.

All operations are pure, immutable, and replay-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from qec.sims.qutrit_propulsion_universe import (
    UniverseSnapshot,
    evolve_universe,
)
from qec.sims.trajectory_stability_analysis import (
    TrajectoryStabilityReport,
    analyze_trajectory_stability,
)

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

STABILITY_BONUS: float = 5.0
SLINGSHOT_BONUS: float = 8.0
OSCILLATORY_PENALTY: float = -4.0
DIVERGENT_PENALTY: float = -10.0

DISTANCE_WEIGHT: float = 1.0

# Energy penalty weights
FIELD_ENERGY_DECAY_WEIGHT: float = 1.0
PROPULSION_INTENSITY_WEIGHT: float = 0.5


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AutopilotPolicyReport:
    """Frozen report of autopilot route policy search results."""

    best_schedule: Tuple[int, ...]
    best_score: float
    distance_score: float
    energy_efficiency_score: float
    stability_score: float
    slingshot_bonus: float
    steps_evaluated: int
    candidate_count: int


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _compute_energy_penalty(
    schedule: Tuple[int, ...],
    initial_field_energy: float,
    final_field_energy: float,
) -> float:
    """Deterministic energy penalty from field decay and propulsion intensity.

    Energy penalty = field_energy_decay + propulsion_intensity.
    field_energy_decay = initial - final field energy (always >= 0).
    propulsion_intensity = 0.5 * sum(mode values in schedule).
    """
    field_decay = max(0.0, initial_field_energy - final_field_energy)
    propulsion_sum = sum(schedule)
    return (
        FIELD_ENERGY_DECAY_WEIGHT * field_decay
        + PROPULSION_INTENSITY_WEIGHT * propulsion_sum
    )


def _score_stability(report: TrajectoryStabilityReport) -> float:
    """Return stability component from trajectory label."""
    label = report.stability_label
    if label == "stable":
        return STABILITY_BONUS
    if label == "oscillatory":
        return OSCILLATORY_PENALTY
    if label == "divergent":
        return DIVERGENT_PENALTY
    # slingshot label gives no stability bonus (slingshot handled separately)
    return 0.0


def _score_candidate(
    snapshot: UniverseSnapshot,
    schedule: Tuple[int, ...],
    steps: int,
) -> Tuple[float, float, float, float, float]:
    """Score a single candidate schedule.

    Returns (total_score, distance_score, energy_efficiency_score,
             stability_score, slingshot_bonus).
    """
    evolved = evolve_universe(snapshot, steps, propulsion_schedule=schedule)
    report = analyze_trajectory_stability(evolved)

    distance_score = report.net_distance * DISTANCE_WEIGHT
    stability_score = _score_stability(report)
    slingshot_bonus = SLINGSHOT_BONUS if report.slingshot_detected else 0.0
    energy_penalty = _compute_energy_penalty(
        schedule,
        snapshot.craft_state.field_energy,
        evolved.craft_state.field_energy,
    )
    # energy_efficiency_score is the negated penalty for reporting
    energy_efficiency_score = -energy_penalty

    total = distance_score + stability_score + slingshot_bonus - energy_penalty
    return (total, distance_score, energy_efficiency_score,
            stability_score, slingshot_bonus)


# ---------------------------------------------------------------------------
# Tie-breaking
# ---------------------------------------------------------------------------

def _tie_break_key(schedule: Tuple[int, ...]) -> Tuple[int, Tuple[int, ...]]:
    """Deterministic tie-break: lower propulsion sum, then lexicographic."""
    return (sum(schedule), schedule)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_best_route_policy(
    snapshot: UniverseSnapshot,
    candidate_schedules: Tuple[Tuple[int, ...], ...],
    steps: int = 10,
) -> AutopilotPolicyReport:
    """Evaluate candidate propulsion schedules and select the best route.

    Parameters
    ----------
    snapshot : UniverseSnapshot
        Initial universe state.
    candidate_schedules : tuple of tuple of int
        Each inner tuple is a propulsion schedule of length ``steps``.
    steps : int
        Number of evolution steps per candidate (default 10).

    Returns
    -------
    AutopilotPolicyReport
        Frozen report with best schedule, scores, and metadata.

    Raises
    ------
    ValueError
        If candidate_schedules is empty or a schedule has wrong length.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not candidate_schedules:
        raise ValueError("candidate_schedules must not be empty")

    best_idx = 0
    best_total = float("-inf")
    best_tie_key = None
    results: list = []

    for i, schedule in enumerate(candidate_schedules):
        if len(schedule) != steps:
            raise ValueError(
                f"Schedule {i} has length {len(schedule)}, expected {steps}"
            )
        scores = _score_candidate(snapshot, schedule, steps)
        total = scores[0]
        tie_key = _tie_break_key(schedule)

        is_better = False
        if total > best_total:
            is_better = True
        elif total == best_total and tie_key < best_tie_key:
            is_better = True

        if is_better:
            best_idx = i
            best_total = total
            best_tie_key = tie_key

        results.append(scores)

    best_scores = results[best_idx]
    return AutopilotPolicyReport(
        best_schedule=candidate_schedules[best_idx],
        best_score=best_scores[0],
        distance_score=best_scores[1],
        energy_efficiency_score=best_scores[2],
        stability_score=best_scores[3],
        slingshot_bonus=best_scores[4],
        steps_evaluated=steps,
        candidate_count=len(candidate_schedules),
    )
