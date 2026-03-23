"""v83.1.0 — Target Sweep Engine (Invariant Query Grid).

Runs ``run_hybrid_inverse_design`` over a list of target specs and returns
structured results.  This is a thin deterministic loop — no new logic, no
parallelism, no reordering.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.  Read-only on inputs.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from qec.experiments.hybrid_inverse_design import run_hybrid_inverse_design


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _best_pair_identity(best_pair: Dict[str, Any]) -> tuple:
    """Extract deterministic identity key from a best_pair record."""
    theta = best_pair.get("theta", [])
    sequence = best_pair.get("sequence")
    return (tuple(theta), sequence)


def detect_transitions(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detect phase boundaries where the best candidate changes.

    Iterates consecutive result pairs and records transitions where the
    best_pair identity (theta_tuple, sequence) differs.

    Returns list of transition records with from/to index and best_pair info.
    """
    transitions: List[Dict[str, Any]] = []
    for i in range(len(results) - 1):
        id_a = _best_pair_identity(results[i].get("best_pair", {}))
        id_b = _best_pair_identity(results[i + 1].get("best_pair", {}))
        if id_a != id_b:
            from_best = results[i]["best_pair"]
            to_best = results[i + 1]["best_pair"]
            record: Dict[str, Any] = {
                "from_index": i,
                "to_index": i + 1,
                "from_best": from_best,
                "to_best": to_best,
                "delta_score": to_best.get("score", 0.0) - from_best.get("score", 0.0),
                "delta_compatibility": (
                    to_best.get("compatibility", 0.0)
                    - from_best.get("compatibility", 0.0)
                ),
                "class_change": from_best.get("class") != to_best.get("class"),
                "phase_change": from_best.get("phase") != to_best.get("phase"),
            }
            # Optional normalized score delta when both sides provide it.
            if "normalized_score" in from_best and "normalized_score" in to_best:
                record["delta_normalized_score"] = (
                    to_best["normalized_score"] - from_best["normalized_score"]
                )
            transitions.append(record)
    return transitions


def summarize_transitions(transitions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate transition metrics into a compact phase-map summary.

    Pure deterministic aggregation — no side effects, no randomness.
    """
    n = len(transitions)
    if n == 0:
        return {
            "n_transitions": 0,
            "mean_delta_score": 0.0,
            "max_delta_score": 0.0,
            "mean_delta_compatibility": 0.0,
            "max_delta_compatibility": 0.0,
            "class_change_count": 0,
            "phase_change_count": 0,
            "degenerate_count": 0,
        }
    delta_scores = [t["delta_score"] for t in transitions]
    delta_compats = [t["delta_compatibility"] for t in transitions]
    return {
        "n_transitions": n,
        "mean_delta_score": sum(delta_scores) / n,
        "max_delta_score": max(abs(d) for d in delta_scores),
        "mean_delta_compatibility": sum(delta_compats) / n,
        "max_delta_compatibility": max(abs(d) for d in delta_compats),
        "class_change_count": sum(1 for t in transitions if t["class_change"]),
        "phase_change_count": sum(1 for t in transitions if t["phase_change"]),
        "degenerate_count": sum(1 for t in transitions if t["delta_score"] == 0.0),
    }


def extract_regimes(
    results: List[Dict[str, Any]],
    transitions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Segment sweep results into phase regions between transitions.

    Boundaries are drawn at each ``transition["to_index"]``.  The dominant
    entry for each regime is its *first* result (deterministic, no voting).
    """
    n = len(results)
    if n == 0:
        return []

    # Build boundary list: [0, t0_to, t1_to, ..., n]
    boundaries = [0]
    for t in transitions:
        boundaries.append(t["to_index"])
    boundaries.append(n)

    regimes: List[Dict[str, Any]] = []
    for k in range(len(boundaries) - 1):
        start = boundaries[k]
        end = boundaries[k + 1] - 1  # inclusive end index
        length = end - start + 1
        dominant = results[start]["best_pair"]
        scores = [r["best_pair"].get("score", 0.0) for r in results[start:end + 1]]
        compats = [r["best_pair"].get("compatibility", 0.0) for r in results[start:end + 1]]
        regimes.append({
            "start_index": start,
            "end_index": end,
            "length": length,
            "dominant_theta": dominant.get("theta", []),
            "dominant_sequence": dominant.get("sequence"),
            "dominant_class": dominant.get("class"),
            "dominant_phase": dominant.get("phase"),
            "mean_score": sum(scores) / length,
            "mean_compatibility": sum(compats) / length,
        })
    return regimes


def compare_regimes(regimes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare adjacent regimes to quantify inter-regime deltas.

    Pure post-processing — no side effects, no randomness.
    """
    comparisons: List[Dict[str, Any]] = []
    for i in range(len(regimes) - 1):
        fr = regimes[i]
        to = regimes[i + 1]
        comparisons.append({
            "from_index": i,
            "to_index": i + 1,
            "from_range": [fr["start_index"], fr["end_index"]],
            "to_range": [to["start_index"], to["end_index"]],
            "delta_mean_score": to["mean_score"] - fr["mean_score"],
            "delta_mean_compatibility": to["mean_compatibility"] - fr["mean_compatibility"],
            "class_shift": fr["dominant_class"] != to["dominant_class"],
            "phase_shift": fr["dominant_phase"] != to["dominant_phase"],
            "structure_shift": (
                tuple(fr["dominant_theta"]) != tuple(to["dominant_theta"])
                or fr["dominant_sequence"] != to["dominant_sequence"]
            ),
        })
    return comparisons


def run_target_sweep(
    targets: List[Union[str, Dict[str, Any]]],
    theta_grid: List[List[float]],
    sequences: List[Any],
    *,
    v_circ_fn: Optional[Callable[..., np.ndarray]] = None,
    pipeline_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Sweep *targets* through the hybrid inverse-design engine.

    Parameters
    ----------
    targets:
        Ordered list of target behaviour classes (strings like ``"stable"``)
        or parametric target-spec dicts.  Iterated in input order.
    theta_grid:
        UFF parameter vectors forwarded to each design call.
    sequences:
        Deterministic sequence inputs forwarded to each design call.
    v_circ_fn:
        Optional velocity-curve generator (passed through).
    pipeline_fn:
        Pipeline function (passed through, required by the engine).
    top_k:
        Number of best candidates per target (default 5).

    Returns
    -------
    dict with keys ``n_targets``, ``targets``, ``results``.
    """
    # Defensive copy of inputs so callers can verify no mutation.
    targets_copy = copy.deepcopy(targets)
    theta_copy = copy.deepcopy(theta_grid)
    seq_copy = copy.deepcopy(sequences)

    results: List[Dict[str, Any]] = []

    for target in targets_copy:
        single = run_hybrid_inverse_design(
            target=target,
            theta_grid=theta_copy,
            sequences=seq_copy,
            v_circ_fn=v_circ_fn,
            pipeline_fn=pipeline_fn,
            top_k=top_k,
        )
        results.append({
            "target_spec": single["target_spec"],
            "best_pair": single["best_pair"],
            "top_k": single["top_k"],
            "score_distribution": single["score_distribution"],
        })

    transitions = detect_transitions(results)
    regimes = extract_regimes(results, transitions)
    return {
        "n_targets": len(results),
        "targets": [r["target_spec"] for r in results],
        "results": results,
        "transitions": transitions,
        "transition_summary": summarize_transitions(transitions),
        "regimes": regimes,
        "regime_comparisons": compare_regimes(regimes),
    }
