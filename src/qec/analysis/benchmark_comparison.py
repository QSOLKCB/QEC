"""v101.1.0 — Benchmark comparison snapshot support.

Provides helpers to extract comparable final scores from benchmark
results for use by the self-evaluation layer.

All functions are:
- deterministic (identical inputs → identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs

Dependencies: stdlib only.  No randomness, no mutation, no ML.
"""

from __future__ import annotations

from typing import Any, Dict


def summarize_baseline_finals(
    results: Dict[str, Any],
) -> Dict[str, float]:
    """Extract comparable final scores from benchmark results.

    Accepts a dict mapping baseline names to result dicts.  Each
    result dict should contain a ``"final_score"`` key.  Missing
    or non-numeric scores are omitted from the output.

    Parameters
    ----------
    results : dict[str, Any]
        Mapping of baseline name → result dict with ``"final_score"``.

    Returns
    -------
    dict[str, float]
        Mapping of baseline name → final score (float).
        Only baselines with valid numeric scores are included.
    """
    finals: Dict[str, float] = {}
    for name in sorted(results.keys()):
        entry = results[name]
        if not isinstance(entry, dict):
            continue
        score = entry.get("final_score")
        if score is None:
            continue
        try:
            finals[name] = float(score)
        except (TypeError, ValueError):
            continue
    return finals


__all__ = [
    "summarize_baseline_finals",
]
