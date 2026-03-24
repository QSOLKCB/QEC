"""v101.5.0 — Strategy adapter for trust-aware selection.

Wraps outputs from the v101.4 ternary bosonic pipeline into
candidate strategies, scores and ranks them, and returns the
selected strategy. Non-invasive integration hook.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- opt-in only

Dependencies: stdlib only (plus sibling analysis modules).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from qec.analysis.strategy_selection import rank_strategies, select_strategy


def build_candidate_strategies(
    experiment_result: Dict[str, Any],
    trust_signals: Optional[Dict[str, float]] = None,
    strategy_configs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build candidate strategies from a bosonic experiment result.

    If *strategy_configs* is provided, each config is paired with the
    experiment metrics and trust signals to form a candidate.

    If *strategy_configs* is None, a single default candidate is built
    from the experiment result.

    Parameters
    ----------
    experiment_result : dict
        Output of ``run_concatenated_bosonic_experiment``.
    trust_signals : dict, optional
        Trust signals (stability, global_trust, regime_trust, blended_trust).
        Defaults to neutral values if omitted.
    strategy_configs : list of dict, optional
        Strategy definitions, each with at least ``name`` and optional
        ``threshold``, ``rounds``.

    Returns
    -------
    list of dict
        Candidate strategies ready for ``select_strategy``.
    """
    metrics = dict(experiment_result.get("metrics", {}))
    ts = dict(trust_signals) if trust_signals else {}

    if strategy_configs is None:
        return [
            {
                "name": "default",
                "metrics": metrics,
                "trust_signals": ts,
                "config": {
                    "threshold": experiment_result.get("threshold", 0.3),
                    "rounds": experiment_result.get("rounds", 3),
                },
            }
        ]

    candidates = []
    for cfg in strategy_configs:
        candidate = {
            "name": str(cfg.get("name", cfg.get("id", "unnamed"))),
            "metrics": metrics,
            "trust_signals": ts,
            "config": {k: v for k, v in cfg.items() if k not in ("name", "id")},
        }
        # Allow per-strategy metric overrides
        if "metrics" in cfg:
            candidate["metrics"] = dict(cfg["metrics"])
        if "trust_signals" in cfg:
            candidate["trust_signals"] = dict(cfg["trust_signals"])
        candidates.append(candidate)

    return candidates


def run_strategy_selection(
    experiment_result: Dict[str, Any],
    trust_signals: Optional[Dict[str, float]] = None,
    strategy_configs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run full trust-aware strategy selection pipeline.

    Parameters
    ----------
    experiment_result : dict
        Output of ``run_concatenated_bosonic_experiment``.
    trust_signals : dict, optional
        Trust signals for modulation.
    strategy_configs : list of dict, optional
        Candidate strategy definitions.

    Returns
    -------
    dict
        Contains ``candidates`` (ranked list), ``selected`` (best strategy),
        and ``n_candidates`` count.
    """
    candidates = build_candidate_strategies(
        experiment_result, trust_signals, strategy_configs,
    )
    ranked = rank_strategies(candidates)
    selected = select_strategy(candidates)

    return {
        "candidates": ranked,
        "selected": selected,
        "n_candidates": len(ranked),
    }


def format_selection_summary(selection_result: Dict[str, Any]) -> str:
    """Format a human-readable summary of strategy selection.

    Parameters
    ----------
    selection_result : dict
        Output of ``run_strategy_selection``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines = [
        "",
        "=== Strategy Selection ===",
        f"Candidates: {selection_result['n_candidates']}",
        "",
    ]

    for c in selection_result["candidates"]:
        marker = " <-- selected" if c.get("_rank") == 1 else ""
        lines.append(f"  #{c.get('_rank', '?')} {c['name']}: "
                      f"score={c.get('_score', 0.0):.6f}{marker}")

    selected = selection_result["selected"]
    lines.append(f"\nSelected: {selected['name']} "
                 f"(score={selected.get('_score', 0.0):.6f})")

    return "\n".join(lines)


__all__ = [
    "build_candidate_strategies",
    "run_strategy_selection",
    "format_selection_summary",
]
