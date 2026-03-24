"""v101.5.0 — Trust-aware strategy selection.

Scores strategies using design metrics, confidence efficiency,
temporal stability, and trust modulation (global + regime).
Selects the most trustworthy strategy deterministically.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs in [0, 1]
- pure analysis signals

Dependencies: stdlib only.  No randomness, no mutation, no ML.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Weights — fixed, deterministic, must sum to 1.0
# ---------------------------------------------------------------------------

SCORE_WEIGHTS: Dict[str, float] = {
    "design_score": 0.35,
    "confidence_efficiency": 0.25,
    "temporal_stability": 0.20,
    "trust_modulation": 0.20,
}


# ---------------------------------------------------------------------------
# Task 1 — Strategy Scoring
# ---------------------------------------------------------------------------


def score_strategy(
    metrics: Dict[str, float],
    trust_signals: Dict[str, float],
) -> float:
    """Score a single strategy using metrics and trust signals.

    Components (all in [0, 1]):
    - design_score: base quality from decoder design metrics
    - confidence_efficiency: mean confidence across nodes
    - temporal_stability: stability of confidence over time
    - trust_modulation: blended global + regime trust

    Final score = weighted sum, rounded to 12 decimals.

    Parameters
    ----------
    metrics : dict[str, float]
        Must contain ``design_score`` and ``confidence_efficiency``.
    trust_signals : dict[str, float]
        May contain ``stability``, ``global_trust``, ``regime_trust``,
        ``blended_trust``.

    Returns
    -------
    float
        Composite score in [0.0, 1.0].
    """
    # Base design score
    design = max(0.0, min(1.0, float(metrics.get("design_score", 0.0))))

    # Confidence efficiency
    confidence = max(0.0, min(1.0, float(metrics.get("confidence_efficiency", 0.0))))

    # Temporal stability (default 0.5 = neutral if unavailable)
    stability = max(0.0, min(1.0, float(trust_signals.get("stability", 0.5))))

    # Trust modulation: prefer blended, fall back to global, then default 0.5
    trust = float(
        trust_signals.get(
            "blended_trust",
            trust_signals.get("global_trust", 0.5),
        )
    )
    trust = max(0.0, min(1.0, trust))

    w = SCORE_WEIGHTS
    score = (
        w["design_score"] * design
        + w["confidence_efficiency"] * confidence
        + w["temporal_stability"] * stability
        + w["trust_modulation"] * trust
    )

    score = round(score, 12)
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Task 2 — Strategy Selection
# ---------------------------------------------------------------------------


def select_strategy(
    strategies: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Select the best strategy deterministically.

    Each strategy dict must contain:
    - ``name`` or ``id``: string identifier for tie-breaking
    - ``metrics``: dict with design_score, confidence_efficiency
    - ``trust_signals``: dict with stability, trust values

    Tie-breaking: lowest name/id in sorted order wins.

    Parameters
    ----------
    strategies : list of dict
        Candidate strategies with metrics and trust_signals.

    Returns
    -------
    dict
        The selected strategy dict, augmented with ``_score``.

    Raises
    ------
    ValueError
        If strategies list is empty.
    """
    if not strategies:
        raise ValueError("strategies list must not be empty")

    # Score all strategies
    scored: List[Tuple[float, str, int, Dict[str, Any]]] = []
    for idx, strat in enumerate(strategies):
        metrics = strat.get("metrics", {})
        trust_signals = strat.get("trust_signals", {})
        s = score_strategy(metrics, trust_signals)
        name = str(strat.get("name", strat.get("id", "")))
        scored.append((s, name, idx, strat))

    # Sort: highest score first, then lowest name (deterministic tie-break)
    scored.sort(key=lambda t: (-t[0], t[1]))

    best_score, _name, _idx, best_strat = scored[0]

    # Build result without mutating input
    result = dict(best_strat)
    result["_score"] = best_score

    return result


def rank_strategies(
    strategies: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank all strategies by score (descending), with deterministic ordering.

    Parameters
    ----------
    strategies : list of dict
        Candidate strategies with metrics and trust_signals.

    Returns
    -------
    list of dict
        Strategies sorted by score descending, each augmented with ``_score``
        and ``_rank`` (1-based).
    """
    if not strategies:
        return []

    scored: List[Tuple[float, str, int, Dict[str, Any]]] = []
    for idx, strat in enumerate(strategies):
        metrics = strat.get("metrics", {})
        trust_signals = strat.get("trust_signals", {})
        s = score_strategy(metrics, trust_signals)
        name = str(strat.get("name", strat.get("id", "")))
        scored.append((s, name, idx, strat))

    scored.sort(key=lambda t: (-t[0], t[1]))

    ranked = []
    for rank, (s, _name, _idx, strat) in enumerate(scored, 1):
        entry = dict(strat)
        entry["_score"] = s
        entry["_rank"] = rank
        ranked.append(entry)

    return ranked


__all__ = [
    "SCORE_WEIGHTS",
    "score_strategy",
    "select_strategy",
    "rank_strategies",
]
