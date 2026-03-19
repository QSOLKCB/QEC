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
    }
