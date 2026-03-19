"""
v7.9.0 — Incremental Spectral Benchmark Experiment.

Compares full NB recomputation, incremental warm-start update, and
localized incremental update on a Tanner graph repair candidate.

Records runtime, spectral agreement, and SIS drift metrics.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from qec.diagnostics.spectral_nb import compute_nb_spectrum
from qec.diagnostics.spectral_repair import (
    apply_repair_candidate,
    propose_repair_candidates,
)
from qec.diagnostics.spectral_incremental import (
    detect_edge_swap,
    identify_affected_nb_edges,
    update_nb_eigenpair_incremental,
    update_nb_eigenpair_localized,
)


_BENCHMARK_SCHEMA_VERSION = 1
_ROUND = 12


def run_incremental_spectral_benchmark(
    H: np.ndarray,
    *,
    top_k_edges: int = 10,
    max_candidates: int = 10,
    max_iter: int = 30,
    tol: float = 1e-10,
) -> dict[str, Any]:
    """Run benchmark comparing full vs incremental spectral updates.

    Selects the first valid repair candidate, applies it, and
    compares three methods for computing the post-repair eigenpair:

    1. Full recomputation via ``compute_nb_spectrum``
    2. Warm-start incremental power iteration
    3. Localized incremental power iteration

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    top_k_edges : int
        Number of hot edges for candidate generation.
    max_candidates : int
        Maximum candidates to generate.
    max_iter : int
        Maximum iterations for incremental solvers.
    tol : float
        Convergence tolerance for incremental solvers.

    Returns
    -------
    dict[str, Any]
        Benchmark artifact with runtime and accuracy metrics.
    """
    H_arr = np.asarray(H, dtype=np.float64)

    # Compute original spectrum
    orig = compute_nb_spectrum(H_arr)
    previous_eigenvector = orig["eigenvector"]

    # Generate repair candidates
    candidates = propose_repair_candidates(
        H_arr, top_k_edges=top_k_edges, max_candidates=max_candidates,
    )

    if not candidates:
        return _empty_artifact()

    # Use first valid candidate
    candidate = candidates[0]
    H_repaired = apply_repair_candidate(H_arr, candidate)

    # --- Full recomputation ---
    t0 = time.perf_counter()
    full_result = compute_nb_spectrum(H_repaired)
    runtime_full = time.perf_counter() - t0

    # --- Incremental warm-start ---
    t0 = time.perf_counter()
    incr_result = update_nb_eigenpair_incremental(
        H_repaired, previous_eigenvector,
        max_iter=max_iter, tol=tol,
    )
    runtime_incremental = time.perf_counter() - t0

    # --- Localized incremental ---
    affected = identify_affected_nb_edges(H_arr, H_repaired)

    t0 = time.perf_counter()
    local_result = update_nb_eigenpair_localized(
        H_repaired, previous_eigenvector, affected,
        max_iter=max_iter, tol=tol,
    )
    runtime_localized = time.perf_counter() - t0

    # Compute SIS for full result
    sis_full = full_result["sis"]

    # Compute SIS for incremental result
    incr_eigvec = incr_result["eigenvector"]
    incr_ipr = float(np.sum(np.abs(incr_eigvec) ** 4) /
                      (np.sum(np.abs(incr_eigvec) ** 2) ** 2))
    incr_edge_energy = np.abs(incr_eigvec) ** 2
    from qec.diagnostics.spectral_nb import _compute_eeec, _compute_sis
    incr_eeec = _compute_eeec(incr_edge_energy)
    sis_incremental = _compute_sis(
        incr_result["spectral_radius"], incr_ipr, incr_eeec,
    )

    # Spectral radius difference
    sr_diff = abs(
        full_result["spectral_radius"] - incr_result["spectral_radius"]
    )

    # Eigenvector cosine similarity
    v_full = full_result["eigenvector"]
    v_incr = incr_result["eigenvector"]
    cos_sim = _cosine_similarity(v_full, v_incr)

    # Speedup ratios (avoid division by zero)
    speedup_incremental = (
        runtime_full / runtime_incremental if runtime_incremental > 0 else 0.0
    )
    speedup_localized = (
        runtime_full / runtime_localized if runtime_localized > 0 else 0.0
    )

    # Delta SIS
    delta_sis = sis_incremental - sis_full

    artifact = {
        "schema_version": _BENCHMARK_SCHEMA_VERSION,
        "runtime_full": round(runtime_full, _ROUND),
        "runtime_incremental": round(runtime_incremental, _ROUND),
        "runtime_localized": round(runtime_localized, _ROUND),
        "speedup_incremental": round(speedup_incremental, _ROUND),
        "speedup_localized": round(speedup_localized, _ROUND),
        "spectral_radius_difference": round(sr_diff, _ROUND),
        "eigenvector_cosine_similarity": round(cos_sim, _ROUND),
        "sis_full": round(sis_full, _ROUND),
        "sis_incremental": round(sis_incremental, _ROUND),
        "delta_sis": round(delta_sis, _ROUND),
        "incremental_converged": incr_result["converged"],
        "incremental_iterations": incr_result["iterations"],
        "localized_converged": local_result["converged"],
        "localized_iterations": local_result["iterations"],
        "localized_used": local_result.get("localized", False),
        "num_affected_edges": len(affected),
        "candidate": candidate,
    }

    return artifact


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(abs(np.dot(a, b)) / (norm_a * norm_b))


def _empty_artifact() -> dict[str, Any]:
    """Return empty artifact when no candidates are available."""
    return {
        "schema_version": _BENCHMARK_SCHEMA_VERSION,
        "runtime_full": 0.0,
        "runtime_incremental": 0.0,
        "runtime_localized": 0.0,
        "speedup_incremental": 0.0,
        "speedup_localized": 0.0,
        "spectral_radius_difference": 0.0,
        "eigenvector_cosine_similarity": 0.0,
        "sis_full": 0.0,
        "sis_incremental": 0.0,
        "delta_sis": 0.0,
        "incremental_converged": False,
        "incremental_iterations": 0,
        "localized_converged": False,
        "localized_iterations": 0,
        "localized_used": False,
        "num_affected_edges": 0,
        "candidate": None,
    }


def serialize_benchmark_artifact(artifact: dict[str, Any]) -> str:
    """Serialize benchmark artifact to canonical JSON."""
    return json.dumps(artifact, sort_keys=True)
