"""v102.4.0 — Strategy evolution and transition analysis.

Tracks how strategies transition between taxonomy types over time,
detecting evolution patterns like stability, adaptation, volatility,
convergence, divergence, and cycling.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_type_trajectory(
    runs: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """Build per-strategy type trajectories from multiple runs.

    For each run, computes the taxonomy type for each strategy using
    the v102.3 pipeline (history -> trajectory_metrics -> regime ->
    taxonomy), then collects the type sequence per strategy.

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key with a list of
        strategy dicts (each having ``"name"`` and ``"metrics"``).

    Returns
    -------
    dict
        Keyed by strategy name (sorted).  Each value is a list of
        taxonomy type strings in run order.  Strategies missing from
        a run are skipped (no padding).
    """
    from qec.analysis.regime_classification import classify_regime
    from qec.analysis.strategy_history import build_strategy_history
    from qec.analysis.strategy_taxonomy import classify_strategy_type
    from qec.analysis.trajectory_metrics import compute_trajectory_metrics

    # Collect all strategy names across all runs for deterministic ordering.
    all_names: set[str] = set()
    for run in runs:
        for strat in run.get("strategies", []):
            name = strat.get("name")
            if name is not None:
                all_names.add(name)
    sorted_names = sorted(all_names)

    # Build per-run taxonomy by treating each run as a single-run history.
    trajectories: Dict[str, List[str]] = {name: [] for name in sorted_names}

    for run in runs:
        single_run = [run]
        history = build_strategy_history(single_run)
        traj_metrics = compute_trajectory_metrics(history)
        regimes = classify_regime(traj_metrics)
        taxonomy = classify_strategy_type(traj_metrics, regimes)

        for name in sorted_names:
            if name in taxonomy:
                trajectories[name].append(taxonomy[name]["type"])

    return trajectories


def compute_transition_metrics(
    trajectories: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    """Compute transition metrics for each strategy's type trajectory.

    Parameters
    ----------
    trajectories : dict
        Output of ``build_type_trajectory``.  Maps strategy names to
        lists of taxonomy type strings.

    Returns
    -------
    dict
        Keyed by strategy name (sorted).  Each value contains:

        - ``num_transitions`` : int — count of type changes
        - ``transition_rate`` : float — transitions / (len - 1), 0.0 if len < 2
        - ``unique_types`` : int — number of distinct types visited
        - ``dominant_type`` : str — most frequent type (tie: first occurrence)
        - ``stability_score`` : float — 1 / (1 + transition_rate)
        - ``longest_streak`` : int — max consecutive identical types

        All floats rounded to 12 decimal places.
    """
    result: Dict[str, Dict[str, Any]] = {}

    for name in sorted(trajectories.keys()):
        seq = trajectories[name]

        num_transitions = _count_transitions(seq)
        transition_rate = round(
            num_transitions / (len(seq) - 1) if len(seq) >= 2 else 0.0,
            12,
        )
        unique_types = len(dict.fromkeys(seq))
        dominant_type = _dominant_type(seq)
        stability_score = round(1.0 / (1.0 + transition_rate), 12)
        longest_streak = _longest_streak(seq)

        result[name] = {
            "num_transitions": num_transitions,
            "transition_rate": transition_rate,
            "unique_types": unique_types,
            "dominant_type": dominant_type,
            "stability_score": stability_score,
            "longest_streak": longest_streak,
        }

    return result


def classify_evolution_pattern(
    metrics: Dict[str, Dict[str, Any]],
    trajectories: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    """Classify each strategy's evolution pattern.

    Parameters
    ----------
    metrics : dict
        Output of ``compute_transition_metrics``.
    trajectories : dict
        Output of ``build_type_trajectory``.

    Returns
    -------
    dict
        Keyed by strategy name (sorted).  Each value contains:

        - ``pattern`` : str — evolution pattern label
        - ``stability_score`` : float
        - ``num_transitions`` : int
        - ``dominant_type`` : str

    Evolution patterns (applied in priority order; first match wins).
    Structural patterns dominate statistical ones:

    1. ``"cycling"`` — detects repeating pattern (e.g. A->B->A->B)
    2. ``"converging"`` — last 3 types identical (sequence length >= 3)
    3. ``"volatile"`` — transition_rate > 0.5
    4. ``"diverging"`` — unique_types >= 4 and no repetition
    5. ``"adaptive"`` — transition_rate >= 0.1 and <= 0.5 and unique_types >= 2
    6. ``"stable_evolver"`` — transition_rate < 0.1 and unique_types <= 2
    7. ``"transitional"`` — fallback
    """
    result: Dict[str, Dict[str, Any]] = {}

    for name in sorted(metrics.keys()):
        m = metrics[name]
        seq = trajectories.get(name, [])
        pattern = _classify_single_pattern(m, seq)

        result[name] = {
            "pattern": pattern,
            "stability_score": m["stability_score"],
            "num_transitions": m["num_transitions"],
            "dominant_type": m["dominant_type"],
        }

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_transitions(seq: List[str]) -> int:
    """Count the number of type changes in a sequence."""
    if len(seq) < 2:
        return 0
    count = 0
    for i in range(len(seq) - 1):
        if seq[i] != seq[i + 1]:
            count += 1
    return count


def _dominant_type(seq: List[str]) -> str:
    """Return the most frequent type.  Tie-break: first occurrence."""
    if not seq:
        return "unknown"
    # Count frequencies preserving insertion order.
    counts: Dict[str, int] = {}
    for t in seq:
        counts[t] = counts.get(t, 0) + 1
    max_count = max(counts.values())
    # First occurrence with max count wins.
    for t in seq:
        if counts[t] == max_count:
            return t
    return seq[0]  # pragma: no cover


def _longest_streak(seq: List[str]) -> int:
    """Return the length of the longest consecutive identical type run."""
    if not seq:
        return 0
    best = 1
    current = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current += 1
            if current > best:
                best = current
        else:
            current = 1
    return best


def _has_cycling_pattern(seq: List[str]) -> bool:
    """Detect repeating patterns in a sequence.

    Checks for period lengths 2..len//2.  A period *p* is a cycle if
    every element at index *i* equals the element at index *i mod p*
    for all *i* in the sequence.  The period itself must contain at
    least 2 distinct types; a constant sequence is not a cycle.
    """
    n = len(seq)
    if n < 4:
        return False
    for period in range(2, n // 2 + 1):
        # Skip if the period window is constant (not a true cycle).
        if len(set(seq[:period])) < 2:
            continue
        is_cycle = True
        for i in range(period, n):
            if seq[i] != seq[i % period]:
                is_cycle = False
                break
        if is_cycle:
            return True
    return False


def _has_no_repetition(seq: List[str]) -> bool:
    """Return True if no type appears more than once."""
    seen: set[str] = set()
    for t in seq:
        if t in seen:
            return False
        seen.add(t)
    return True


def _classify_single_pattern(
    m: Dict[str, Any],
    seq: List[str],
) -> str:
    """Classify a single strategy's evolution pattern.

    Structural patterns (cycling, converging, diverging) are checked
    before statistical ones (volatile, adaptive, stable_evolver) so
    that observable structure is never masked by aggregate statistics.
    """
    transition_rate = m["transition_rate"]
    unique_types = m["unique_types"]

    # 1. Cycling — repeating structural pattern dominates all.
    if _has_cycling_pattern(seq):
        return "cycling"
    # 2. Converging — last 3 types identical, but only if there was
    #    at least one transition (a constant sequence is not converging).
    if (
        len(seq) >= 3
        and seq[-1] == seq[-2] == seq[-3]
        and m["num_transitions"] > 0
    ):
        return "converging"
    # 3. Volatile.
    if transition_rate > 0.5:
        return "volatile"
    # 4. Diverging — many unique types, no repetition.
    if unique_types >= 4 and _has_no_repetition(seq):
        return "diverging"
    # 5. Adaptive.
    if 0.1 <= transition_rate <= 0.5 and unique_types >= 2:
        return "adaptive"
    # 6. Stable evolver.
    if transition_rate < 0.1 and unique_types <= 2:
        return "stable_evolver"
    # 7. Transitional (fallback).
    return "transitional"


__all__ = [
    "build_type_trajectory",
    "compute_transition_metrics",
    "classify_evolution_pattern",
]
