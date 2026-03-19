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

    return {
        "n_targets": len(results),
        "targets": [r["target_spec"] for r in results],
        "results": results,
        "transitions": detect_transitions(results),
    }
