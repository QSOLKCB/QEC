"""v102.4.0 — Strategy adapter for trust-aware selection.

Wraps outputs from the v101.4 ternary bosonic pipeline into
candidate strategies, scores and ranks them, and returns the
selected strategy. Non-invasive integration hook.

v101.6.0 adds ``run_generation_selection_pipeline`` which integrates
deterministic strategy generation (27 candidates) with evaluation,
scoring, and selection.

v101.7.0 adds dual-system (ternary + quaternary) support with
``run_dual_generation_pipeline`` and ``format_comparison_summary``.

v101.8.0 adds deterministic dominance pruning (Pareto frontier) via
``run_pruned_pipeline`` and ``format_pruning_summary``.

v101.9.0 adds structure-aware dominance with consistency gap enrichment,
temporal revival detection, pairwise correlation-based redundancy pruning,
and ``run_structure_aware_pipeline`` with ``format_structure_aware_summary``.

v102.2.0 adds trajectory tracking and regime analysis via
``run_trajectory_analysis`` and ``format_trajectory_summary``.

v102.4.0 adds evolution analysis via ``run_evolution_analysis``
and ``format_evolution_summary``.

v102.7.0 adds flow geometry embedding via ``run_flow_geometry_analysis``
and ``format_flow_geometry_summary``.

All functions are:
- deterministic (identical inputs -> identical outputs)
- side-effect free (no mutation of inputs)
- bounded outputs
- opt-in only

Dependencies: stdlib only (plus sibling analysis modules).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from qec.analysis.consistency_metrics import enrich_with_consistency_gap
from qec.analysis.regime_classification import classify_regime
from qec.analysis.strategy_history import build_strategy_history
from qec.analysis.strategy_taxonomy import classify_strategy_type
from qec.analysis.trajectory_metrics import compute_trajectory_metrics
from qec.analysis.dominance_pruning import pareto_prune, pruning_stats
from qec.analysis.pareto_analysis import compute_pareto_front
from qec.analysis.pareto_explanation import explain_pareto
from qec.analysis.representation_analysis import compare_representations
from qec.analysis.strategy_clustering import cluster_strategies
from qec.analysis.strategy_correlation import prune_redundant
from qec.analysis.strategy_embedding import embed_strategies_2d
from qec.analysis.strategy_explanation import compare_strategies, explain_strategy
from qec.analysis.strategy_generation import generate_strategies
from qec.analysis.strategy_selection import rank_strategies, select_strategy
from qec.analysis.temporal_patterns import enrich_with_revival


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


def run_generation_selection_pipeline(
    base_strategy: Dict[str, Any],
    trust_signals: Optional[Dict[str, float]] = None,
    *,
    include_quaternary: bool = False,
) -> Dict[str, Any]:
    """Run the full generate -> evaluate -> score -> rank -> select pipeline.

    Generates 27 (or 54 with quaternary) deterministic candidate strategies
    from ``base_strategy``, wraps each as a scored candidate using the
    existing v101.5 scoring infrastructure, ranks them, and selects the best.

    Parameters
    ----------
    base_strategy : dict
        Base strategy with at least ``"config"`` key.
    trust_signals : dict, optional
        Trust signals for score modulation.
    include_quaternary : bool
        If True, generate both ternary and quaternary strategies (54 total).

    Returns
    -------
    dict
        Contains ``candidates`` (all with scores), ``ranked`` (sorted),
        and ``selected`` (best strategy).
    """
    ts = dict(trust_signals) if trust_signals else {}

    # Step 1: generate candidates (27 or 54)
    generated = generate_strategies(
        base_strategy, include_quaternary=include_quaternary,
    )

    # Step 2: wrap each as a scorable candidate
    candidates: List[Dict[str, Any]] = []
    for g in generated:
        candidate = {
            "name": g["name"],
            "metrics": dict(base_strategy.get("metrics", {})),
            "trust_signals": ts,
            "config": g["config"],
            "origin": g["origin"],
            "state_system": g.get("state_system", "ternary"),
        }
        candidates.append(candidate)

    # Step 3: rank (scores + sorts)
    ranked = rank_strategies(candidates)

    # Step 4: select best
    selected = select_strategy(candidates)

    return {
        "candidates": candidates,
        "ranked": ranked,
        "selected": selected,
    }


def format_generation_summary(result: Dict[str, Any]) -> str:
    """Format a human-readable summary of strategy generation + selection.

    Parameters
    ----------
    result : dict
        Output of ``run_generation_selection_pipeline``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    ranked = result["ranked"]
    selected = result["selected"]
    scores = [r["_score"] for r in ranked]

    lines = [
        "",
        "=== Strategy Generation + Selection ===",
        f"Total candidates: {len(result['candidates'])}",
        "",
        "Top 5:",
    ]

    for entry in ranked[:5]:
        marker = " <-- selected" if entry["name"] == selected["name"] else ""
        lines.append(f"  #{entry['_rank']} {entry['name']}: "
                     f"score={entry['_score']:.6f}{marker}")

    lines.append("")
    lines.append(f"Selected: {selected['name']} "
                 f"(score={selected.get('_score', 0.0):.6f})")

    if scores:
        lines.append(f"Score distribution: "
                     f"min={min(scores):.6f} "
                     f"max={max(scores):.6f} "
                     f"mean={sum(scores) / len(scores):.6f}")

    return "\n".join(lines)


def run_dual_generation_pipeline(
    base_strategy: Dict[str, Any],
    raw_signals: Any,
    trust_signals: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run ternary and quaternary pipelines, evaluate, and compare.

    Generates 54 strategies (27 ternary + 27 quaternary), routes each
    to its correct pipeline runner, scores, ranks, and selects the
    overall best as well as per-system bests.

    Parameters
    ----------
    base_strategy : dict
        Base strategy with at least ``"config"`` key and ``"metrics"``.
    raw_signals : array-like
        Raw analog signals for the quaternary pipeline.
    trust_signals : dict, optional
        Trust signals for score modulation.

    Returns
    -------
    dict
        Contains full ranked results, per-system bests, and overall winner.
    """
    import numpy as np

    from qec.experiments.concatenated_bosonic_decoder import (
        run_concatenated_bosonic_experiment,
    )
    from qec.experiments.quaternary_bosonic_decoder import (
        run_quaternary_bosonic_experiment,
    )

    ts = dict(trust_signals) if trust_signals else {}
    raw = np.asarray(raw_signals, dtype=np.float64)

    # Generate 54 strategies
    generated = generate_strategies(
        base_strategy, include_quaternary=True,
    )

    # Route each strategy to correct pipeline and collect metrics
    candidates: List[Dict[str, Any]] = []
    for g in generated:
        state_sys = g.get("state_system", "ternary")
        cfg = g["config"]
        rounds = cfg.get("rounds", 3)
        confidence_scale = cfg.get("confidence_scale", 1.0)
        neutral_bias = cfg.get("neutral_bias", 0.0)

        if state_sys == "quaternary":
            result = run_quaternary_bosonic_experiment(
                raw, rounds=rounds,
                confidence_scale=confidence_scale,
                neutral_bias=neutral_bias,
            )
        else:
            threshold = cfg.get("threshold", 0.3)
            result = run_concatenated_bosonic_experiment(
                raw, threshold=threshold, rounds=rounds,
                confidence_scale=confidence_scale,
                neutral_bias=neutral_bias,
            )

        candidate = {
            "name": g["name"],
            "metrics": dict(result.get("metrics", {})),
            "trust_signals": ts,
            "config": cfg,
            "origin": g["origin"],
            "state_system": state_sys,
        }
        candidates.append(candidate)

    # Rank all 54
    ranked = rank_strategies(candidates)
    selected = select_strategy(candidates)

    # Per-system bests
    ternary_ranked = [c for c in ranked if c.get("state_system") == "ternary"]
    quaternary_ranked = [c for c in ranked if c.get("state_system") == "quaternary"]

    best_ternary = ternary_ranked[0] if ternary_ranked else None
    best_quaternary = quaternary_ranked[0] if quaternary_ranked else None

    return {
        "candidates": candidates,
        "ranked": ranked,
        "selected": selected,
        "best_ternary": best_ternary,
        "best_quaternary": best_quaternary,
        "n_ternary": len([c for c in candidates if c.get("state_system") == "ternary"]),
        "n_quaternary": len([c for c in candidates if c.get("state_system") == "quaternary"]),
    }


def format_comparison_summary(result: Dict[str, Any]) -> str:
    """Format a human-readable comparison of ternary vs quaternary.

    Parameters
    ----------
    result : dict
        Output of ``run_dual_generation_pipeline``.

    Returns
    -------
    str
        Multi-line comparison summary string.
    """
    ranked = result["ranked"]
    selected = result["selected"]
    best_t = result["best_ternary"]
    best_q = result["best_quaternary"]

    n_total = len(result["candidates"])
    n_ternary = result["n_ternary"]
    n_quaternary = result["n_quaternary"]

    lines = [
        "",
        "=== State System Comparison ===",
        f"Total strategies: {n_total} ({n_ternary} ternary + {n_quaternary} quaternary)",
        "",
    ]

    if best_t:
        lines.append(f"Ternary best:     {best_t['_score']:.4f}  ({best_t['name']})")
    if best_q:
        lines.append(f"Quaternary best:  {best_q['_score']:.4f}  ({best_q['name']})")

    if best_t and best_q:
        t_score = best_t["_score"]
        q_score = best_q["_score"]
        winner_sys = selected.get("state_system", "ternary")

        if t_score == q_score:
            diff_str = "tied"
        else:
            winner_score = max(t_score, q_score)
            loser_score = min(t_score, q_score)
            if loser_score > 0:
                pct = (winner_score - loser_score) / loser_score * 100.0
            else:
                pct = 100.0
            diff_str = f"+{pct:.1f}%"

        lines.append(f"\nWinner: {winner_sys} ({diff_str})")

    lines.append("")
    lines.append("Top 5 overall:")
    for entry in ranked[:5]:
        marker = " <-- selected" if entry["name"] == selected["name"] else ""
        lines.append(
            f"  #{entry['_rank']} [{entry.get('state_system', '?')}] "
            f"{entry['name']}: score={entry['_score']:.6f}{marker}"
        )

    return "\n".join(lines)


def run_pruned_pipeline(
    base_strategy: Dict[str, Any],
    raw_signals: Any,
    trust_signals: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Generate, evaluate, prune (Pareto), rank, and select.

    Pipeline:
    1. Generate 54 strategies (27 ternary + 27 quaternary)
    2. Evaluate each through the correct pipeline
    3. Prune dominated strategies (Pareto frontier)
    4. Rank remaining by score
    5. Select best

    Parameters
    ----------
    base_strategy : dict
        Base strategy with at least ``"config"`` and ``"metrics"``.
    raw_signals : array-like
        Raw analog signals for quaternary pipeline.
    trust_signals : dict, optional
        Trust signals for score modulation.

    Returns
    -------
    dict
        Contains full results including pruning statistics.
    """
    # Step 1-2: generate and evaluate all 54
    dual_result = run_dual_generation_pipeline(
        base_strategy, raw_signals, trust_signals,
    )

    candidates = dual_result["candidates"]

    # Step 3: Pareto prune
    pruned = pareto_prune(candidates)

    # Step 4: rank pruned set
    ranked = rank_strategies(pruned)

    # Step 5: select best from pruned set
    selected = select_strategy(pruned)

    # Per-system bests from pruned set
    ternary_ranked = [c for c in ranked if c.get("state_system") == "ternary"]
    quaternary_ranked = [c for c in ranked if c.get("state_system") == "quaternary"]

    best_ternary = ternary_ranked[0] if ternary_ranked else None
    best_quaternary = quaternary_ranked[0] if quaternary_ranked else None

    stats = pruning_stats(candidates, pruned)

    return {
        "candidates": candidates,
        "pruned": pruned,
        "ranked": ranked,
        "selected": selected,
        "best_ternary": best_ternary,
        "best_quaternary": best_quaternary,
        "n_total": len(candidates),
        "n_pruned": len(pruned),
        "stats": stats,
    }


def format_pruning_summary(result: Dict[str, Any]) -> str:
    """Format a human-readable summary of pruned pipeline results.

    Parameters
    ----------
    result : dict
        Output of ``run_pruned_pipeline``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    selected = result["selected"]
    best_t = result["best_ternary"]
    best_q = result["best_quaternary"]
    stats = result["stats"]

    lines = [
        "",
        "=== Dominance Pruning (Pareto Frontier) ===",
        f"Total strategies:        {result['n_total']}",
        f"After pruning:           {result['n_pruned']}",
        f"Dominance ratio:         {stats['dominance_ratio']:.4f}",
        "",
    ]

    if best_t:
        lines.append(
            f"Best ternary:            {best_t['name']} "
            f"(score={best_t.get('_score', 0.0):.6f})"
        )
    if best_q:
        lines.append(
            f"Best quaternary:         {best_q['name']} "
            f"(score={best_q.get('_score', 0.0):.6f})"
        )

    lines.append(
        f"Selected (global best):  {selected['name']} "
        f"(score={selected.get('_score', 0.0):.6f})"
    )

    return "\n".join(lines)


def enrich_strategies(
    candidates: List[Dict[str, Any]],
    score_histories: Optional[Dict[str, List[float]]] = None,
) -> List[Dict[str, Any]]:
    """Enrich candidates with consistency gap and optional revival info.

    Does not mutate inputs.

    Parameters
    ----------
    candidates : list of dict
        Strategy candidates.
    score_histories : dict, optional
        Mapping of strategy name to score history list.

    Returns
    -------
    list of dict
        Enriched copies of candidates.
    """
    histories = score_histories or {}
    enriched = []
    for c in candidates:
        e = enrich_with_consistency_gap(c)
        name = e.get("name", "")
        if name in histories:
            e = enrich_with_revival(e, histories[name])
        enriched.append(e)
    return enriched


def run_structure_aware_pipeline(
    base_strategy: Dict[str, Any],
    raw_signals: Any,
    trust_signals: Optional[Dict[str, float]] = None,
    *,
    score_histories: Optional[Dict[str, List[float]]] = None,
    redundancy_threshold: float = 0.98,
) -> Dict[str, Any]:
    """Generate, evaluate, enrich, prune (structure-aware), and select.

    Pipeline:
    1. Generate 54 strategies (27 ternary + 27 quaternary)
    2. Evaluate each through the correct pipeline
    3. Enrich with consistency_gap and revival info
    4. Structure-aware Pareto prune
    5. Correlation-based redundancy prune
    6. Rank remaining by score
    7. Select best

    Parameters
    ----------
    base_strategy : dict
        Base strategy with at least ``"config"`` and ``"metrics"``.
    raw_signals : array-like
        Raw analog signals for quaternary pipeline.
    trust_signals : dict, optional
        Trust signals for score modulation.
    score_histories : dict, optional
        Mapping of strategy name to list of historical scores.
    redundancy_threshold : float
        Correlation threshold for redundancy pruning (default 0.98).

    Returns
    -------
    dict
        Contains full results including enrichment and pruning statistics.
    """
    # Step 1-2: generate and evaluate all 54
    dual_result = run_dual_generation_pipeline(
        base_strategy, raw_signals, trust_signals,
    )

    candidates = dual_result["candidates"]

    # Step 3: enrich
    enriched = enrich_strategies(candidates, score_histories)

    # Step 4: structure-aware Pareto prune
    pareto_pruned = pareto_prune(enriched, structure_aware=True)

    # Step 5: correlation-based redundancy prune
    final_pruned = prune_redundant(pareto_pruned, threshold=redundancy_threshold)

    # Step 6: rank
    ranked = rank_strategies(final_pruned)

    # Step 7: select
    selected = select_strategy(final_pruned)

    # Per-system bests
    ternary_ranked = [c for c in ranked if c.get("state_system") == "ternary"]
    quaternary_ranked = [c for c in ranked if c.get("state_system") == "quaternary"]

    best_ternary = ternary_ranked[0] if ternary_ranked else None
    best_quaternary = quaternary_ranked[0] if quaternary_ranked else None

    pareto_stats = pruning_stats(enriched, pareto_pruned)
    redundancy_removed = len(pareto_pruned) - len(final_pruned)

    # Compute avg consistency gap
    gaps = [
        float(c.get("metrics", {}).get("consistency_gap", 0.0))
        for c in enriched
    ]
    avg_gap = round(sum(gaps) / len(gaps), 12) if gaps else 0.0

    # Count revivals
    revival_count = sum(
        1 for c in enriched
        if c.get("metrics", {}).get("has_revival", False)
    )

    return {
        "candidates": enriched,
        "pareto_pruned": pareto_pruned,
        "final_pruned": final_pruned,
        "ranked": ranked,
        "selected": selected,
        "best_ternary": best_ternary,
        "best_quaternary": best_quaternary,
        "n_total": len(enriched),
        "n_after_pareto": len(pareto_pruned),
        "n_after_redundancy": len(final_pruned),
        "pareto_stats": pareto_stats,
        "redundancy_removed": redundancy_removed,
        "avg_consistency_gap": avg_gap,
        "revival_count": revival_count,
    }


def format_structure_aware_summary(result: Dict[str, Any]) -> str:
    """Format a human-readable summary of structure-aware pipeline results.

    Parameters
    ----------
    result : dict
        Output of ``run_structure_aware_pipeline``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    selected = result["selected"]
    best_t = result["best_ternary"]
    best_q = result["best_quaternary"]
    pareto_stats = result["pareto_stats"]

    lines = [
        "",
        "=== Structure-Aware Dominance Pruning ===",
        f"Total strategies:        {result['n_total']}",
        f"After Pareto pruning:    {result['n_after_pareto']}",
        f"After redundancy prune:  {result['n_after_redundancy']}",
        f"Dominance ratio:         {pareto_stats['dominance_ratio']:.4f}",
        f"Redundant removed:       {result['redundancy_removed']}",
        f"Avg consistency gap:     {result['avg_consistency_gap']:.6f}",
        f"Strategies with revival: {result['revival_count']} / {result['n_total']}",
        "",
    ]

    if best_t:
        lines.append(
            f"Best ternary:            {best_t['name']} "
            f"(score={best_t.get('_score', 0.0):.6f})"
        )
    if best_q:
        lines.append(
            f"Best quaternary:         {best_q['name']} "
            f"(score={best_q.get('_score', 0.0):.6f})"
        )

    lines.append(
        f"Selected (global best):  {selected['name']} "
        f"(score={selected.get('_score', 0.0):.6f})"
    )

    return "\n".join(lines)


def run_analysis_pipeline(
    base_strategy: Dict[str, Any],
    raw_signals: Any,
    trust_signals: Optional[Dict[str, float]] = None,
    *,
    score_histories: Optional[Dict[str, List[float]]] = None,
    redundancy_threshold: float = 0.98,
    cluster_threshold: float = 0.9,
    structure_aware: bool = True,
) -> Dict[str, Any]:
    """Run the full analysis pipeline: generate -> evaluate -> enrich -> prune -> cluster -> embed -> analyze.

    Extends the structure-aware pipeline with clustering, embedding,
    representation comparison, and Pareto front extraction for
    interpretability.

    Parameters
    ----------
    base_strategy : dict
        Base strategy with at least ``"config"`` and ``"metrics"``.
    raw_signals : array-like
        Raw analog signals for quaternary pipeline.
    trust_signals : dict, optional
        Trust signals for score modulation.
    score_histories : dict, optional
        Mapping of strategy name to list of historical scores.
    redundancy_threshold : float
        Correlation threshold for redundancy pruning (default 0.98).
    cluster_threshold : float
        Correlation threshold for clustering (default 0.9).
    structure_aware : bool
        If True, use structure-aware dominance (default True).

    Returns
    -------
    dict
        Contains structure-aware pipeline results plus analysis outputs:
        ``embedding``, ``clusters``, ``pareto_front``, ``representation``.
    """
    # Steps 1-7: run the structure-aware pipeline
    sa_result = run_structure_aware_pipeline(
        base_strategy,
        raw_signals,
        trust_signals,
        score_histories=score_histories,
        redundancy_threshold=redundancy_threshold,
    )

    final_strategies = sa_result["final_pruned"]
    all_enriched = sa_result["candidates"]

    # Step: cluster
    clusters = cluster_strategies(final_strategies, threshold=cluster_threshold)

    # Step: embed
    embedding = embed_strategies_2d(final_strategies)

    # Step: Pareto front (from enriched set, structure-aware)
    pareto_front = compute_pareto_front(
        all_enriched, structure_aware=structure_aware,
    )

    # Step: representation comparison (on all enriched candidates)
    representation = compare_representations(all_enriched)

    # Step: explain selected strategy
    selected_explanation = explain_strategy(sa_result["selected"])

    # Step: explain Pareto front
    pareto_explanations = explain_pareto(pareto_front)

    sa_result["embedding"] = embedding
    sa_result["clusters"] = clusters
    sa_result["pareto_front"] = pareto_front
    sa_result["representation"] = representation
    sa_result["selected_explanation"] = selected_explanation
    sa_result["pareto_explanations"] = pareto_explanations

    return sa_result


def format_analysis_summary(
    result: Dict[str, Any],
    *,
    show_pareto: bool = True,
    show_clusters: bool = True,
    show_map: bool = False,
    show_explanation: bool = False,
) -> str:
    """Format a human-readable summary of the analysis pipeline.

    Parameters
    ----------
    result : dict
        Output of ``run_analysis_pipeline``.
    show_pareto : bool
        If True, include the Pareto front section (default True).
    show_clusters : bool
        If True, include the clusters section (default True).
    show_map : bool
        If True, include ASCII strategy map (default False).
    show_explanation : bool
        If True, include strategy explanation and Pareto explanation
        sections (default False).

    Returns
    -------
    str
        Multi-line summary string.
    """
    from qec.visualization.strategy_map import render_strategy_map

    lines = [format_structure_aware_summary(result)]

    # Clusters
    if show_clusters and "clusters" in result:
        clusters = result["clusters"].get("clusters", [])
        lines.append("")
        lines.append("=== Strategy Clusters ===")
        lines.append(f"Total clusters: {len(clusters)}")
        for c in clusters:
            if c["size"] == 1:
                lines.append(f"  [{c['representative']}] (singleton)")
            else:
                others = [m for m in c["members"] if m != c["representative"]]
                lines.append(
                    f"  [{c['representative']}] + {len(others)} similar: "
                    + ", ".join(others[:3])
                    + ("..." if len(others) > 3 else "")
                )

    # Pareto front
    if show_pareto and "pareto_front" in result:
        pareto = result["pareto_front"]
        lines.append("")
        lines.append("=== Pareto Front ===")
        lines.append(f"Non-dominated strategies: {len(pareto)}")
        for p in pareto[:5]:
            ds = float(p.get("metrics", {}).get("design_score", 0.0))
            name = p.get("name", "")
            sys_type = p.get("state_system", "?")
            lines.append(f"  [{sys_type}] {name}: design_score={ds:.6f}")

    # Representation comparison
    rep = result.get("representation", {})
    lines.append("")
    lines.append("=== Representation Comparison ===")
    for sys_name in ("ternary", "quaternary"):
        info = rep.get(sys_name, {})
        count = info.get("count", 0)
        avg = info.get("avg_design_score", 0.0)
        best = info.get("best", "none")
        lines.append(f"  {sys_name}: count={count}, avg_design_score={avg:.6f}, best={best}")

    # Strategy map
    if show_map and "embedding" in result:
        lines.append("")
        lines.append(render_strategy_map(result["embedding"]))

    # Explanation
    if show_explanation:
        explanation = result.get("selected_explanation")
        if explanation:
            lines.append("")
            lines.append("=== Explanation ===")
            lines.append(f"Best strategy: {explanation['name']}")
            lines.append(f"Score: {explanation['score']}")
            lines.append(f"Dominant factors: {', '.join(explanation['dominant_factors'])}")
            comps = explanation.get("components", {})
            for k, v in sorted(comps.items()):
                lines.append(f"  {k}: {v:.6f}")

        pareto_expl = result.get("pareto_explanations")
        if pareto_expl:
            lines.append("")
            lines.append("=== Pareto Explanation ===")
            for pe in pareto_expl:
                strengths = ", ".join(pe["strengths"]) if pe["strengths"] else "none"
                lines.append(f"  {pe['name']}: strengths={strengths}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trajectory analysis (v102.2.0)
# ---------------------------------------------------------------------------


def run_trajectory_analysis(
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the full trajectory analysis pipeline.

    Pipeline: runs -> history -> trajectory_metrics -> regime_classification

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key with a list of
        strategy dicts (each having ``"name"`` and ``"metrics"``).

    Returns
    -------
    dict
        Contains ``"history"``, ``"trajectory_metrics"``, and ``"regimes"``.
    """
    history = build_strategy_history(runs)
    traj_metrics = compute_trajectory_metrics(history)
    regimes = classify_regime(traj_metrics)

    return {
        "history": history,
        "trajectory_metrics": traj_metrics,
        "regimes": regimes,
    }


def format_trajectory_summary(result: Dict[str, Any]) -> str:
    """Format trajectory analysis results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_trajectory_analysis``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    traj = result.get("trajectory_metrics", {})
    regimes = result.get("regimes", {})

    lines.append("=== Trajectory Summary ===")

    for name in sorted(traj.keys()):
        m = traj[name]
        regime = regimes.get(name, "unknown")
        lines.append(f"Strategy: {name}")
        lines.append(f"  Mean: {m['mean_score']:.6f}")
        lines.append(f"  Variance: {m['variance_score']:.6f}")
        lines.append(f"  Stability: {m['stability']:.6f}")
        lines.append(f"  Trend: {m['trend']:.6f}")
        lines.append(f"  Oscillation: {m['oscillation']}")
        lines.append(f"  Regime: {regime}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Taxonomy analysis (v102.3.0)
# ---------------------------------------------------------------------------


def run_taxonomy_analysis(
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the full taxonomy analysis pipeline.

    Pipeline: runs -> trajectory -> taxonomy

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key with a list of
        strategy dicts (each having ``"name"`` and ``"metrics"``).

    Returns
    -------
    dict
        Contains ``"history"``, ``"trajectory_metrics"``, ``"regimes"``,
        and ``"taxonomy"``.
    """
    result = run_trajectory_analysis(runs)
    taxonomy = classify_strategy_type(
        result["trajectory_metrics"],
        result["regimes"],
    )
    result["taxonomy"] = taxonomy
    return result


def format_taxonomy_summary(result: Dict[str, Any]) -> str:
    """Format taxonomy analysis results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_taxonomy_analysis``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    taxonomy = result.get("taxonomy", {})

    lines.append("=== Strategy Taxonomy ===")

    for name in sorted(taxonomy.keys()):
        entry = taxonomy[name]
        lines.append(f"Strategy: {name}")
        lines.append(f"  Type: {entry['type']}")
        lines.append(f"  Confidence: {entry['confidence']:.2f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evolution analysis (v102.4.0)
# ---------------------------------------------------------------------------


def run_evolution_analysis(
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the full evolution analysis pipeline.

    Pipeline: runs -> trajectory types (v102.3) -> transition metrics
    -> evolution pattern classification (v102.4)

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key with a list of
        strategy dicts (each having ``"name"`` and ``"metrics"``).

    Returns
    -------
    dict
        Contains ``"history"``, ``"trajectory_metrics"``, ``"regimes"``,
        ``"taxonomy"``, ``"type_trajectories"``, ``"transition_metrics"``,
        and ``"evolution"``.
    """
    from qec.analysis.strategy_evolution import (
        build_type_trajectory,
        classify_evolution_pattern,
        compute_transition_metrics,
    )

    result = run_taxonomy_analysis(runs)
    type_trajectories = build_type_trajectory(runs)
    transition_metrics = compute_transition_metrics(type_trajectories)
    evolution = classify_evolution_pattern(transition_metrics, type_trajectories)

    result["type_trajectories"] = type_trajectories
    result["transition_metrics"] = transition_metrics
    result["evolution"] = evolution
    return result


def format_evolution_summary(result: Dict[str, Any]) -> str:
    """Format evolution analysis results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_evolution_analysis``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    evolution = result.get("evolution", {})

    lines.append("=== Evolution Analysis ===")

    for name in sorted(evolution.keys()):
        entry = evolution[name]
        lines.append(f"Strategy: {name}")
        lines.append(f"  Pattern: {entry['pattern']}")
        lines.append(f"  Transitions: {entry['num_transitions']}")
        lines.append(f"  Stability: {entry['stability_score']:.2f}")
        lines.append(f"  Dominant: {entry['dominant_type']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Transition graph analysis (v102.5.0)
# ---------------------------------------------------------------------------


def run_transition_graph_analysis(
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the full transition graph analysis pipeline.

    Pipeline: runs -> taxonomy (v102.3) -> evolution trajectories (v102.4)
    -> transition graph (v102.5)

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key with a list of
        strategy dicts (each having ``"name"`` and ``"metrics"``).

    Returns
    -------
    dict
        Contains all keys from ``run_evolution_analysis`` plus
        ``"transition_graph"``, ``"node_stats"``, ``"ranked_transitions"``,
        and ``"transition_patterns"``.
    """
    from qec.analysis.transition_graph import (
        build_transition_graph,
        compute_node_stats,
        detect_transition_patterns,
        rank_transitions,
    )

    result = run_evolution_analysis(runs)
    type_trajectories = result["type_trajectories"]

    graph = build_transition_graph(type_trajectories)
    node_stats = compute_node_stats(graph)
    ranked = rank_transitions(graph)
    patterns = detect_transition_patterns(graph, node_stats=node_stats)

    result["transition_graph"] = graph
    result["node_stats"] = node_stats
    result["ranked_transitions"] = ranked
    result["transition_patterns"] = patterns
    return result


def format_transition_graph_summary(
    result: Dict[str, Any],
    show_transition_graph: bool = True,
) -> str:
    """Format transition graph analysis results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_transition_graph_analysis``.
    show_transition_graph : bool
        Whether to include the transition graph section (default True).

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []

    if not show_transition_graph:
        return format_evolution_summary(result)

    lines.append("=== Transition Graph ===")

    # Top transitions.
    ranked = result.get("ranked_transitions", [])
    if ranked:
        lines.append("Top Transitions:")
        for src, tgt, count in ranked:
            lines.append(f"  {src} → {tgt} : {count}")

    # Node stats.
    node_stats = result.get("node_stats", {})
    if node_stats:
        lines.append("Node Stats:")
        for node in sorted(node_stats.keys()):
            s = node_stats[node]
            lines.append(
                f"  {node}: in={s['in_degree']} "
                f"out={s['out_degree']} flow={s['total_flow']}"
            )

    # Patterns.
    patterns = result.get("transition_patterns", {})
    has_patterns = any(
        bool(patterns.get(k))
        for k in ("bidirectional", "self_loops", "sources", "sinks")
    )
    if has_patterns:
        lines.append("Patterns:")
        for a, b in patterns.get("bidirectional", []):
            lines.append(f"  bidirectional: {a} ↔ {b}")
        for s in patterns.get("self_loops", []):
            lines.append(f"  self-loop: {s}")
        for s in patterns.get("sources", []):
            lines.append(f"  source: {s}")
        for s in patterns.get("sinks", []):
            lines.append(f"  sink: {s}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase space analysis (v102.6.0)
# ---------------------------------------------------------------------------


def run_phase_space_analysis(
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the full phase space analysis pipeline.

    Pipeline: runs -> taxonomy (v102.3) -> evolution trajectories (v102.4)
    -> transition graph (v102.5) -> phase space (v102.6)

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key with a list of
        strategy dicts (each having ``"name"`` and ``"metrics"``).

    Returns
    -------
    dict
        Contains all keys from ``run_transition_graph_analysis`` plus
        ``"attractors"``, ``"basins"``, ``"escape_dynamics"``, and
        ``"phase_classification"``.
    """
    from qec.analysis.phase_space import (
        classify_phase_state,
        detect_attractors,
        detect_basins,
        detect_escape_dynamics,
    )

    result = run_transition_graph_analysis(runs)
    graph = result["transition_graph"]
    node_stats = result["node_stats"]

    attractors = detect_attractors(graph, node_stats)
    basins = detect_basins(graph, node_stats)
    escape = detect_escape_dynamics(graph, node_stats)
    classification = classify_phase_state(attractors, basins, escape)

    result["attractors"] = attractors
    result["basins"] = basins
    result["escape_dynamics"] = escape
    result["phase_classification"] = classification
    return result


def format_phase_space_summary(
    result: Dict[str, Any],
) -> str:
    """Format phase space analysis results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_phase_space_analysis``.

    Returns
    -------
    str
        Multi-line summary string.
    """
    lines: List[str] = []
    lines.append("=== Phase Space Analysis ===")

    attractors = result.get("attractors", {})
    basins = result.get("basins", {})
    escape = result.get("escape_dynamics", {})
    classification = result.get("phase_classification", {})

    all_nodes = sorted(
        set(
            list(attractors.keys())
            + list(basins.keys())
            + list(escape.keys())
            + list(classification.keys())
        )
    )

    for node in all_nodes:
        att = attractors.get(node, {})
        bas = basins.get(node, {})
        esc = escape.get(node, {})
        cls = classification.get(node, {})

        lines.append(f"Type: {node}")
        lines.append(f"  Attractor: {att.get('is_attractor', False)}")
        lines.append(f"  Score: {att.get('score', 0)}")
        lines.append(f"  Basin Strength: {bas.get('basin_strength', 0)}")
        lines.append(f"  Escape Rate: {esc.get('escape_rate', 0)}")
        lines.append(f"  Phase: {cls.get('phase', 'neutral')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Flow geometry analysis (v102.7.0)
# ---------------------------------------------------------------------------


def run_flow_geometry_analysis(
    runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the full flow geometry analysis pipeline.

    Pipeline: runs -> taxonomy (v102.3) -> evolution trajectories (v102.4)
    -> transition graph (v102.5) -> phase space (v102.6)
    -> flow geometry (v102.7)

    Parameters
    ----------
    runs : list of dict
        Each run must contain a ``"strategies"`` key with a list of
        strategy dicts (each having ``"name"`` and ``"metrics"``).

    Returns
    -------
    dict
        Contains all keys from ``run_phase_space_analysis`` plus
        ``"flow_geometry"`` with ``coordinates``, ``metrics``, and ``nodes``.
    """
    from qec.analysis.flow_geometry import compute_flow_geometry

    result = run_phase_space_analysis(runs)
    graph = result["transition_graph"]
    geometry = compute_flow_geometry(graph)

    result["flow_geometry"] = geometry
    return result


def format_flow_geometry_summary(
    result: Dict[str, Any],
    *,
    show_map: bool = False,
) -> str:
    """Format flow geometry analysis results as a human-readable summary.

    Parameters
    ----------
    result : dict
        Output of ``run_flow_geometry_analysis``.
    show_map : bool
        If True, include ASCII geometry map (default False).

    Returns
    -------
    str
        Multi-line summary string.
    """
    from qec.analysis.flow_geometry import render_ascii_map

    lines: List[str] = []
    lines.append("=== Flow Geometry ===")

    geometry = result.get("flow_geometry", {})
    coords = geometry.get("coordinates", {})
    metrics = geometry.get("metrics", {})

    for name in sorted(coords.keys()):
        x, y = coords[name]
        m = metrics.get(name, {})
        dist = m.get("distance_from_center", 0.0)
        cluster = m.get("cluster_score", 0.0)
        lines.append(f"Type: {name}")
        lines.append(f"  Coord: ({x:.2f}, {y:.2f})")
        lines.append(f"  Dist: {dist:.2f}")
        lines.append(f"  Cluster: {cluster:.2f}")

    if show_map and coords:
        lines.append("")
        lines.append(render_ascii_map(coords))

    return "\n".join(lines)


__all__ = [
    "build_candidate_strategies",
    "run_strategy_selection",
    "format_selection_summary",
    "run_generation_selection_pipeline",
    "format_generation_summary",
    "run_dual_generation_pipeline",
    "format_comparison_summary",
    "run_pruned_pipeline",
    "format_pruning_summary",
    "enrich_strategies",
    "run_structure_aware_pipeline",
    "format_structure_aware_summary",
    "run_analysis_pipeline",
    "format_analysis_summary",
    "explain_strategy",
    "compare_strategies",
    "explain_pareto",
    "run_trajectory_analysis",
    "format_trajectory_summary",
    "run_taxonomy_analysis",
    "format_taxonomy_summary",
    "run_evolution_analysis",
    "format_evolution_summary",
    "run_transition_graph_analysis",
    "format_transition_graph_summary",
    "run_phase_space_analysis",
    "format_phase_space_summary",
    "run_flow_geometry_analysis",
    "format_flow_geometry_summary",
]
