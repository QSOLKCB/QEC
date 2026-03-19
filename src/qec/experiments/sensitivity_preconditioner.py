"""
v7.6.0 — Sensitivity-Based Preconditioner for Graph Optimization.

Optional preconditioned graph optimizer that uses per-edge instability
sensitivity scores to guide edge-swap candidate generation.

When enabled, high-sensitivity edges are preferentially targeted for
swaps, improving optimization efficiency.  When disabled (default),
baseline graph optimization behavior is preserved exactly.

Pipeline:
  graph
  -> sensitivity map (proxy scores)
  -> sensitivity-weighted candidate ranking
  -> guided edge swaps
  -> optimized graph

Architecture:
  Layer 5 — Experiment.
  Consumes Layer 3 (sensitivity_map diagnostics).
  Consumes Layer 4 (spectral_graph_optimizer for baseline optimization).
  Does not import or modify the decoder (Layer 1).

Fully deterministic: no randomness, no global state, no input mutation.
All floats rounded to 12 decimal places.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from qec.diagnostics.sensitivity_map import (
    compute_proxy_sensitivity_scores,
    compute_sensitivity_map,
    _compute_instability_score_for_H,
)
from qec.experiments.tanner_graph_repair import (
    _extract_edges,
    _build_edge_set,
    _apply_swap,
    _edges_to_H,
)

_ROUND = 12


# ── Core: Sensitivity-Weighted Candidate Generation ───────────────────


def _generate_sensitivity_weighted_candidates(
    H: np.ndarray,
    proxy_scores: list[dict[str, Any]],
    *,
    max_candidates: int = 10,
    top_k_fraction: float = 0.5,
) -> list[dict[str, Any]]:
    """Generate edge-swap candidates weighted by sensitivity scores.

    Selects high-sensitivity edges as removal targets and low-sensitivity
    edges as swap partners, producing degree-preserving edge swaps.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    proxy_scores : list[dict[str, Any]]
        Per-edge proxy sensitivity scores from
        ``compute_proxy_sensitivity_scores``, sorted by sensitivity DESC.
    max_candidates : int
        Maximum number of swap candidates to generate.
    top_k_fraction : float
        Fraction of edges considered "high sensitivity".

    Returns
    -------
    list[dict[str, Any]]
        List of swap candidate descriptors with keys:
        - ``remove`` : list of edge pairs to remove
        - ``add`` : list of edge pairs to add
        - ``description`` : human-readable swap description
        - ``sensitivity_score`` : combined sensitivity of removed edges
    """
    m, n = H.shape

    if not proxy_scores:
        return []

    num_edges = len(proxy_scores)
    top_k = max(1, int(num_edges * top_k_fraction))

    high_sens = proxy_scores[:top_k]
    low_sens = proxy_scores[top_k:]

    if not low_sens:
        mid = max(1, num_edges // 2)
        high_sens = proxy_scores[:mid]
        low_sens = proxy_scores[mid:]

    # Build edge set for duplicate checking.
    all_edges = set()
    for rec in proxy_scores:
        all_edges.add((rec["variable_node"], rec["check_node"]))

    candidates: list[dict[str, Any]] = []

    for h_rec in high_sens:
        v1 = h_rec["variable_node"]
        c1 = h_rec["check_node"]
        s1 = h_rec["proxy_sensitivity"]

        # Ensure v1 is variable node.
        if v1 >= n:
            v1, c1 = c1, v1
        if v1 >= n:
            continue

        for l_rec in low_sens:
            v2 = l_rec["variable_node"]
            c2 = l_rec["check_node"]

            if v2 >= n:
                v2, c2 = c2, v2
            if v2 >= n:
                continue

            if v1 == v2 or c1 == c2:
                continue

            new_e1 = (min(v1, c2), max(v1, c2))
            new_e2 = (min(v2, c1), max(v2, c1))

            if new_e1 in all_edges or new_e2 in all_edges:
                continue

            combined_sensitivity = round(
                s1 + l_rec["proxy_sensitivity"], _ROUND,
            )

            candidates.append({
                "remove": [(v1, c1), (v2, c2)],
                "add": [list(new_e1), list(new_e2)],
                "description": (
                    f"sens-swap ({v1},{c1})+({v2},{c2})"
                    f" -> ({new_e1[0]},{new_e1[1]})"
                    f"+({new_e2[0]},{new_e2[1]})"
                ),
                "sensitivity_score": combined_sensitivity,
            })

            if len(candidates) >= max_candidates:
                # Sort by combined sensitivity DESC for determinism,
                # then description ASC for tie-breaking.
                candidates.sort(
                    key=lambda c: (-c["sensitivity_score"], c["description"]),
                )
                return candidates

    # Sort by combined sensitivity DESC, description ASC.
    candidates.sort(
        key=lambda c: (-c["sensitivity_score"], c["description"]),
    )
    return candidates


# ── Core: Preconditioned Optimization Loop ────────────────────────────


def run_sensitivity_preconditioned_optimization(
    H: np.ndarray,
    *,
    max_iterations: int = 10,
    improvement_threshold: float = 1e-6,
    instability_target: float = 0.1,
    max_candidates: int = 10,
    top_k_fraction: float = 0.5,
    enable_preconditioner: bool = True,
) -> dict[str, Any]:
    """Run sensitivity-preconditioned graph optimization loop.

    When ``enable_preconditioner=True``, uses per-edge proxy sensitivity
    scores to weight candidate generation, prioritizing high-sensitivity
    edges for swaps.

    When ``enable_preconditioner=False``, falls back to the baseline
    spectral gradient candidate generation from v7.5.

    Algorithm:
        H_current = H_initial
        score_current = instability(H_current)

        for iteration in range(max_iterations):
            if enable_preconditioner:
                proxy_scores = compute_proxy_sensitivity(H_current)
                candidates = sensitivity_weighted_candidates(proxy_scores)
            else:
                # Fallback: use v7.5 gradient-based candidates.
                from qec.experiments.spectral_graph_optimizer import (
                    _compute_spectral_edge_gradient,
                    _generate_guided_repair_candidates,
                )
                gradient = _compute_spectral_edge_gradient(H_current)
                candidates = _generate_guided_repair_candidates(
                    H_current, gradient,
                    max_candidates=max_candidates,
                    top_k_fraction=top_k_fraction,
                )

            evaluate candidates deterministically
            best_candidate = argmin(instability_score)

            if improvement <= threshold:
                stop

            apply repair
            update H_current, score_current

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    max_iterations : int
        Maximum optimization iterations.
    improvement_threshold : float
        Minimum improvement to continue.
    instability_target : float
        Stop if instability drops below this.
    max_candidates : int
        Maximum candidates per iteration.
    top_k_fraction : float
        Fraction of edges considered high-sensitivity/gradient.
    enable_preconditioner : bool
        If True (default), use sensitivity-weighted candidates.
        If False, use baseline v7.5 gradient candidates.

    Returns
    -------
    dict[str, Any]
        JSON-serializable optimization artifact with keys:

        - ``initial_instability_score`` : float
        - ``final_instability_score`` : float
        - ``iterations`` : int
        - ``swaps_applied`` : list[dict]
        - ``preconditioner_enabled`` : bool
        - ``optimizer_success`` : bool
        - ``sensitivity_map_summary`` : dict or None
    """
    H_current = np.array(H, dtype=np.float64)
    m, n = H_current.shape

    # Initial instability evaluation.
    score_current = _compute_instability_score_for_H(H_current)
    initial_score = score_current

    swaps_applied: list[dict[str, Any]] = []
    iterations_run = 0

    # Compute initial sensitivity map summary if preconditioner enabled.
    initial_sensitivity_summary: dict[str, Any] | None = None
    if enable_preconditioner:
        initial_proxy = compute_proxy_sensitivity_scores(H_current)
        if initial_proxy:
            proxy_vals = [r["proxy_sensitivity"] for r in initial_proxy]
            initial_sensitivity_summary = {
                "num_edges": len(initial_proxy),
                "max_proxy_sensitivity": max(proxy_vals),
                "mean_proxy_sensitivity": round(
                    sum(proxy_vals) / len(proxy_vals), _ROUND,
                ),
            }

    for iteration in range(max_iterations):
        iterations_run = iteration + 1

        if score_current <= instability_target:
            break

        if enable_preconditioner:
            proxy_scores = compute_proxy_sensitivity_scores(H_current)
            candidates = _generate_sensitivity_weighted_candidates(
                H_current,
                proxy_scores,
                max_candidates=max_candidates,
                top_k_fraction=top_k_fraction,
            )
        else:
            from qec.experiments.spectral_graph_optimizer import (
                _compute_spectral_edge_gradient,
                _generate_guided_repair_candidates,
            )
            gradient = _compute_spectral_edge_gradient(H_current)
            candidates = _generate_guided_repair_candidates(
                H_current,
                gradient,
                max_candidates=max_candidates,
                top_k_fraction=top_k_fraction,
            )

        if not candidates:
            break

        # Evaluate each candidate deterministically.
        best_candidate = None
        best_score = score_current
        best_H = None

        edges = _extract_edges(H_current)
        for candidate in candidates:
            trial_edges = _apply_swap(edges, candidate)
            H_trial = _edges_to_H(trial_edges, m, n)
            trial_score = _compute_instability_score_for_H(H_trial)

            if trial_score < best_score:
                best_score = trial_score
                best_candidate = candidate
                best_H = H_trial

        # Check improvement.
        improvement = round(score_current - best_score, _ROUND)
        if improvement <= improvement_threshold or best_candidate is None:
            break

        # Apply best repair.
        swap_record: dict[str, Any] = {
            "iteration": iteration,
            "swap": {
                "remove": [list(e) for e in best_candidate["remove"]],
                "add": best_candidate["add"],
                "description": best_candidate["description"],
            },
            "score_before": round(score_current, _ROUND),
            "score_after": round(best_score, _ROUND),
            "improvement": improvement,
        }
        if "sensitivity_score" in best_candidate:
            swap_record["sensitivity_score"] = best_candidate["sensitivity_score"]

        swaps_applied.append(swap_record)

        H_current = best_H
        score_current = best_score

    # Final instability evaluation.
    final_score = _compute_instability_score_for_H(H_current)
    optimizer_success = final_score < initial_score

    return {
        "initial_instability_score": round(initial_score, _ROUND),
        "final_instability_score": round(final_score, _ROUND),
        "iterations": iterations_run,
        "swaps_applied": swaps_applied,
        "preconditioner_enabled": enable_preconditioner,
        "optimizer_success": optimizer_success,
        "sensitivity_map_summary": initial_sensitivity_summary,
    }


# ── Harness Integration: Sensitivity Experiment ───────────────────────


def run_sensitivity_preconditioner_experiment(
    H: np.ndarray,
    *,
    max_iterations: int = 10,
    max_candidates: int = 10,
    top_k_fraction: float = 0.5,
) -> dict[str, Any]:
    """Run comparative experiment: baseline vs sensitivity-preconditioned.

    Runs the graph optimization loop twice — once without preconditioner
    (v7.5 baseline) and once with sensitivity preconditioning — and
    returns a comparative report.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    max_iterations : int
        Maximum optimization iterations for each run.
    max_candidates : int
        Maximum candidates per iteration.
    top_k_fraction : float
        Fraction of edges considered high-sensitivity/gradient.

    Returns
    -------
    dict[str, Any]
        JSON-serializable experiment report with keys:

        - ``baseline`` : dict — optimization without preconditioner
        - ``preconditioned`` : dict — optimization with preconditioner
        - ``sensitivity_map`` : dict — full sensitivity map artifact
        - ``comparison`` : dict — delta metrics between runs
    """
    H_arr = np.array(H, dtype=np.float64)

    # Baseline (v7.5 gradient-based).
    baseline_result = run_sensitivity_preconditioned_optimization(
        H_arr,
        max_iterations=max_iterations,
        max_candidates=max_candidates,
        top_k_fraction=top_k_fraction,
        enable_preconditioner=False,
    )

    # Preconditioned (v7.6 sensitivity-based).
    precond_result = run_sensitivity_preconditioned_optimization(
        H_arr,
        max_iterations=max_iterations,
        max_candidates=max_candidates,
        top_k_fraction=top_k_fraction,
        enable_preconditioner=True,
    )

    # Full sensitivity map for the original graph.
    sensitivity_map = compute_sensitivity_map(H_arr)

    # Comparison metrics.
    baseline_final = baseline_result["final_instability_score"]
    precond_final = precond_result["final_instability_score"]
    improvement_delta = round(baseline_final - precond_final, _ROUND)

    return {
        "baseline": baseline_result,
        "preconditioned": precond_result,
        "sensitivity_map": sensitivity_map,
        "comparison": {
            "baseline_final_score": baseline_final,
            "preconditioned_final_score": precond_final,
            "improvement_delta": improvement_delta,
            "preconditioner_better": precond_final < baseline_final,
            "baseline_iterations": baseline_result["iterations"],
            "preconditioned_iterations": precond_result["iterations"],
            "baseline_swaps": len(baseline_result["swaps_applied"]),
            "preconditioned_swaps": len(precond_result["swaps_applied"]),
        },
    }
