"""Deterministic Experiment Selection Engine (v97.7.0).

Maps the law system over metric space, identifies uncertainty, conflict,
and gaps, prioritizes high-value experiment regions, selects non-redundant
experiments, and feeds results back into the law system.

Pipeline: law_set -> uncertainty -> targeted experiments -> improved theory

All algorithms are pure, deterministic, and use only stdlib + numpy.
No mutation of inputs. No randomness. No probabilistic exploration.
"""

from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple
import itertools
import math

import numpy as np

from qec.analysis.law_promotion import Condition, Law


# ---------------------------------------------------------------------------
# FLOAT PRECISION GUARD
# ---------------------------------------------------------------------------

_NORM_DIGITS = 12


def _norm(x: float) -> float:
    return round(float(x), _NORM_DIGITS)


# ---------------------------------------------------------------------------
# STEP 1 — METRIC SPACE GRID
# ---------------------------------------------------------------------------


def build_grid(
    metric_ranges: Dict[str, Tuple[float, float]],
    bins_per_metric: int,
) -> Dict[str, Any]:
    """Build a discretized metric space grid.

    Parameters
    ----------
    metric_ranges : dict
        Mapping metric_name -> (min_val, max_val).
    bins_per_metric : int
        Number of bins along each metric axis.

    Returns
    -------
    dict with keys:
        "metrics" : sorted list of metric names
        "bins"    : int
        "ranges"  : dict of metric -> (min, max)
        "centers" : dict of metric -> list of bin center values
        "cells"   : dict of cell_key (tuple of ints) -> cell_data dict
    """
    if bins_per_metric < 1:
        raise ValueError("bins_per_metric must be >= 1")
    if not metric_ranges:
        raise ValueError("metric_ranges must be non-empty")

    metrics = sorted(metric_ranges.keys())
    centers: Dict[str, List[float]] = {}
    for m in metrics:
        lo, hi = metric_ranges[m]
        if hi <= lo:
            raise ValueError(f"Invalid range for {m}: ({lo}, {hi})")
        step = (hi - lo) / bins_per_metric
        centers[m] = [_norm(lo + step * (i + 0.5)) for i in range(bins_per_metric)]

    # Build cells via cartesian product of bin indices.
    index_ranges = [range(bins_per_metric) for _ in metrics]
    cells: Dict[Tuple[int, ...], Dict[str, Any]] = {}
    for idx_tuple in itertools.product(*index_ranges):
        cells[idx_tuple] = {
            "coverage": 0,
            "sum_outcome": 0.0,
            "sum_outcome_sq": 0.0,
        }

    return {
        "metrics": metrics,
        "bins": bins_per_metric,
        "ranges": dict(metric_ranges),
        "centers": centers,
        "cells": cells,
    }


def cell_center(grid: Dict[str, Any], cell_key: Tuple[int, ...]) -> Dict[str, float]:
    """Return the metric-space center point of a grid cell."""
    metrics = grid["metrics"]
    return {
        metrics[i]: grid["centers"][metrics[i]][cell_key[i]]
        for i in range(len(metrics))
    }


# ---------------------------------------------------------------------------
# STEP 2 — LAW APPLICABILITY GRID
# ---------------------------------------------------------------------------


def compute_applicability(
    laws: List[Law],
    grid: Dict[str, Any],
) -> Dict[Tuple[int, ...], List[int]]:
    """Map each cell to indices of applicable laws.

    A law applies to a cell if all its conditions are satisfied
    by the cell's center point.
    """
    result: Dict[Tuple[int, ...], List[int]] = {}
    for key in sorted(grid["cells"].keys()):
        center = cell_center(grid, key)
        applicable = []
        for i, law in enumerate(laws):
            if law.evaluate(center):
                applicable.append(i)
        result[key] = applicable
    return result


# ---------------------------------------------------------------------------
# STEP 3 — CONFLICT GRID
# ---------------------------------------------------------------------------


def compute_conflict_grid(
    laws: List[Law],
    applicability: Dict[Tuple[int, ...], List[int]],
) -> Dict[Tuple[int, ...], bool]:
    """Identify cells where applicable laws prescribe different actions."""
    result: Dict[Tuple[int, ...], bool] = {}
    for key in sorted(applicability.keys()):
        indices = applicability[key]
        if len(indices) < 2:
            result[key] = False
            continue
        actions = {laws[i].action for i in indices}
        result[key] = len(actions) > 1
    return result


# ---------------------------------------------------------------------------
# STEP 4 — UNCERTAINTY SCORE
# ---------------------------------------------------------------------------


def compute_uncertainty(
    cell_data: Dict[str, Any],
    applicable_laws: List[Law],
    conflict: bool,
    max_coverage: int,
) -> float:
    """Compute uncertainty score in [0, 1] for a single cell.

    Components (equally weighted at 0.25 each):
      - 1 - avg_confidence of applicable laws
      - conflict flag (1 if conflict, 0 otherwise)
      - sparsity = 1 - coverage / max_coverage
      - normalized outcome variance
    """
    coverage = cell_data["coverage"]

    if coverage == 0:
        return 1.0

    # Confidence component.
    if applicable_laws:
        confidences = []
        for law in applicable_laws:
            c = law.scores.get("confidence", law.scores.get("coverage", 0.5))
            confidences.append(c)
        avg_confidence = _norm(sum(confidences) / len(confidences))
    else:
        avg_confidence = 0.0

    # Conflict component.
    conflict_flag = 1.0 if conflict else 0.0

    # Sparsity component.
    safe_max = max(max_coverage, 1)
    sparsity = _norm(1.0 - coverage / safe_max)

    # Variance component.
    mean_outcome = cell_data["sum_outcome"] / coverage
    mean_sq = cell_data["sum_outcome_sq"] / coverage
    variance = max(0.0, mean_sq - mean_outcome * mean_outcome)
    # Normalize variance to [0, 1] using sigmoid-like mapping.
    norm_variance = _norm(variance / (1.0 + variance))

    uncertainty = _norm(
        0.25 * (1.0 - avg_confidence)
        + 0.25 * conflict_flag
        + 0.25 * sparsity
        + 0.25 * norm_variance
    )
    return max(0.0, min(1.0, uncertainty))


# ---------------------------------------------------------------------------
# STEP 5 — STATE CLASSIFICATION
# ---------------------------------------------------------------------------

STATE_UNEXPLORED = "unexplored"
STATE_STABLE = "stable"
STATE_UNSTABLE = "unstable"
STATE_INTERMEDIATE = "intermediate"


def classify_state(coverage: int, uncertainty: float) -> str:
    """Classify a cell's state based on coverage and uncertainty."""
    if coverage == 0:
        return STATE_UNEXPLORED
    if uncertainty <= 0.3:
        return STATE_STABLE
    if uncertainty >= 0.7:
        return STATE_UNSTABLE
    return STATE_INTERMEDIATE


# ---------------------------------------------------------------------------
# STEP 6 — BOUNDARY DETECTION
# ---------------------------------------------------------------------------


def _manhattan_neighbors(
    cell_key: Tuple[int, ...], bins: int,
) -> List[Tuple[int, ...]]:
    """Return Manhattan-adjacent cells within grid bounds."""
    neighbors = []
    for dim in range(len(cell_key)):
        for delta in (-1, 1):
            new_idx = cell_key[dim] + delta
            if 0 <= new_idx < bins:
                neighbor = list(cell_key)
                neighbor[dim] = new_idx
                neighbors.append(tuple(neighbor))
    return neighbors


def compute_boundary_cells(
    state_grid: Dict[Tuple[int, ...], str],
    bins: int,
) -> Dict[Tuple[int, ...], bool]:
    """Identify cells where any Manhattan neighbor has a different state."""
    result: Dict[Tuple[int, ...], bool] = {}
    for key in sorted(state_grid.keys()):
        my_state = state_grid[key]
        is_boundary = False
        for nb in _manhattan_neighbors(key, bins):
            if nb in state_grid and state_grid[nb] != my_state:
                is_boundary = True
                break
        result[key] = is_boundary
    return result


# ---------------------------------------------------------------------------
# STEP 7 — PRIORITY SCORE
# ---------------------------------------------------------------------------


def compute_priority(uncertainty: float, boundary: bool) -> float:
    """Priority = uncertainty + 0.2 * boundary_flag."""
    return _norm(uncertainty + 0.2 * (1.0 if boundary else 0.0))


def is_candidate(state: str, boundary: bool) -> bool:
    """A cell is a candidate if it is not stable OR is a boundary cell."""
    return state != STATE_STABLE or boundary


# ---------------------------------------------------------------------------
# STEP 8 — NOVELTY DETECTION
# ---------------------------------------------------------------------------


def is_novel(
    cell_data: Dict[str, Any],
    applicable_laws: List[Law],
    deviation_threshold: float = 0.5,
) -> bool:
    """Detect if a cell represents novel territory.

    Novel if:
      - coverage == 0 (unexplored), OR
      - observed average outcome deviates from all law predictions
        (we use action hash as a proxy for predicted outcome class)
    """
    if cell_data["coverage"] == 0:
        return True
    if not applicable_laws:
        return True
    # If we have observations, check deviation from expected.
    avg_outcome = cell_data["sum_outcome"] / cell_data["coverage"]
    mean_sq = cell_data["sum_outcome_sq"] / cell_data["coverage"]
    variance = max(0.0, mean_sq - avg_outcome * avg_outcome)
    std = math.sqrt(variance)
    # High variance relative to the mean signals novelty.
    if abs(avg_outcome) < 1e-12:
        return std > deviation_threshold
    cv = std / abs(avg_outcome)
    return cv > deviation_threshold


# ---------------------------------------------------------------------------
# STEP 9 — EXPERIMENT SELECTION
# ---------------------------------------------------------------------------


def select_experiments(
    candidates: List[Tuple[Tuple[int, ...], float]],
    n: int,
    bins: int,
) -> List[Tuple[int, ...]]:
    """Select up to N non-adjacent cells by descending priority.

    Parameters
    ----------
    candidates : list of (cell_key, priority) pairs
    n : maximum number of experiments to select
    bins : grid dimension size (for adjacency checking)

    Returns
    -------
    List of selected cell keys, ordered by priority descending.
    """
    # Sort by priority descending, then by cell_key for determinism.
    sorted_cands = sorted(candidates, key=lambda x: (-x[1], x[0]))

    selected: List[Tuple[int, ...]] = []
    selected_set: Set[Tuple[int, ...]] = set()

    for cell_key, priority in sorted_cands:
        if len(selected) >= n:
            break
        # Check adjacency with already-selected cells.
        adjacent = False
        for sel in selected:
            if _is_adjacent(cell_key, sel):
                adjacent = True
                break
        if not adjacent:
            selected.append(cell_key)
            selected_set.add(cell_key)

    return selected


def _is_adjacent(a: Tuple[int, ...], b: Tuple[int, ...]) -> bool:
    """Check Manhattan adjacency (distance == 1)."""
    dist = sum(abs(ai - bi) for ai, bi in zip(a, b))
    return dist == 1


# ---------------------------------------------------------------------------
# STEP 10 — UPDATE AFTER EXPERIMENTS
# ---------------------------------------------------------------------------


def update_cells(
    grid: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Update grid cells with experiment results (returns new grid).

    Each result must have:
      "cell_key": tuple of ints
      "outcome": float

    Returns a new grid dict with updated cell data.
    """
    import copy

    new_grid = copy.deepcopy(grid)
    for result in results:
        key = tuple(result["cell_key"])
        outcome = float(result["outcome"])
        if key not in new_grid["cells"]:
            continue
        cell = new_grid["cells"][key]
        cell["coverage"] += 1
        cell["sum_outcome"] = _norm(cell["sum_outcome"] + outcome)
        cell["sum_outcome_sq"] = _norm(cell["sum_outcome_sq"] + outcome * outcome)

    return new_grid


# ---------------------------------------------------------------------------
# STEP 11 — FULL STRATEGY LOOP
# ---------------------------------------------------------------------------


def run_strategy(
    laws: List[Law],
    grid: Dict[str, Any],
    n: int,
) -> Dict[str, Any]:
    """Execute the full experiment selection strategy.

    Steps:
      1. Compute applicability
      2. Compute conflict grid
      3. Compute uncertainty for each cell
      4. Classify states
      5. Compute boundary cells
      6. Compute priority scores
      7. Select experiments

    Returns dict with:
      "selected"       : list of selected cell keys
      "applicability"  : cell -> law index list
      "conflicts"      : cell -> bool
      "uncertainty"    : cell -> float
      "states"         : cell -> str
      "boundaries"     : cell -> bool
      "priorities"     : cell -> float
    """
    bins = grid["bins"]

    # 1. Applicability.
    applicability = compute_applicability(laws, grid)

    # 2. Conflict grid.
    conflicts = compute_conflict_grid(laws, applicability)

    # 3. Compute max coverage for normalization.
    max_coverage = max(
        (grid["cells"][k]["coverage"] for k in grid["cells"]), default=1
    )
    max_coverage = max(max_coverage, 1)

    # 4. Uncertainty and state classification.
    uncertainty: Dict[Tuple[int, ...], float] = {}
    states: Dict[Tuple[int, ...], str] = {}
    for key in sorted(grid["cells"].keys()):
        cell_data = grid["cells"][key]
        app_laws = [laws[i] for i in applicability.get(key, [])]
        conflict = conflicts.get(key, False)
        u = compute_uncertainty(cell_data, app_laws, conflict, max_coverage)
        uncertainty[key] = u
        states[key] = classify_state(cell_data["coverage"], u)

    # 5. Boundary detection.
    boundaries = compute_boundary_cells(states, bins)

    # 6. Priority scores.
    priorities: Dict[Tuple[int, ...], float] = {}
    candidates: List[Tuple[Tuple[int, ...], float]] = []
    for key in sorted(grid["cells"].keys()):
        p = compute_priority(uncertainty[key], boundaries.get(key, False))
        priorities[key] = p
        if is_candidate(states[key], boundaries.get(key, False)):
            candidates.append((key, p))

    # 7. Selection.
    selected = select_experiments(candidates, n, bins)

    return {
        "selected": selected,
        "applicability": applicability,
        "conflicts": conflicts,
        "uncertainty": uncertainty,
        "states": states,
        "boundaries": boundaries,
        "priorities": priorities,
    }
