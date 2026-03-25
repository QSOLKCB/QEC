"""v102.2.0 — Strategy history tracking across multiple runs.

Builds per-strategy metric histories from a sequence of run results,
enabling trajectory analysis and regime classification.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs

Dependencies: stdlib only.
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_strategy_history(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    """Build per-strategy metric histories from multiple runs.

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key with a list of
        strategy dicts.  Each strategy dict must have ``"name"`` (str)
        and ``"metrics"`` (dict with float values).

    Returns
    -------
    dict
        Keyed by strategy name (sorted deterministically).  Each value
        is a dict mapping metric name to a list of float values across
        runs (in run order).  Only strategies present in a given run
        contribute a value; missing strategies are skipped (no padding).

    Tracked metrics
    ---------------
    - design_score
    - confidence_efficiency
    - consistency_gap
    - revival_strength
    """
    tracked_metrics = (
        "confidence_efficiency",
        "consistency_gap",
        "design_score",
        "revival_strength",
    )

    # First pass: collect all strategy names for deterministic ordering.
    all_names: set[str] = set()
    for run in runs:
        for strat in run.get("strategies", []):
            name = strat.get("name")
            if name is not None:
                all_names.add(name)

    sorted_names = sorted(all_names)

    # Second pass: build histories.
    history: Dict[str, Dict[str, List[float]]] = {}
    for name in sorted_names:
        history[name] = {m: [] for m in tracked_metrics}

    for run in runs:
        # Index strategies by name for O(1) lookup.
        by_name: Dict[str, Dict[str, Any]] = {}
        for strat in run.get("strategies", []):
            sname = strat.get("name")
            if sname is not None:
                by_name[sname] = strat

        for name in sorted_names:
            if name not in by_name:
                continue
            strat = by_name[name]
            metrics = strat.get("metrics", {})
            for m in tracked_metrics:
                if m in metrics:
                    history[name][m].append(float(metrics[m]))

    return history


__all__ = ["build_strategy_history"]
