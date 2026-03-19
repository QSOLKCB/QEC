"""
v7.5.0 — Deterministic Spectral Graph Optimization Pipeline.

Predictor-guided spectral graph optimizer that modifies Tanner graphs
before decoding begins to reduce BP instability.

Pipeline:
  graph
  → spectral diagnostics (NB spectrum)
  → instability prediction (spectral score)
  → spectral optimization (guided edge swaps)
  → optimized graph

Architecture:
  Layer 4 — Graph Optimization (OS services)
  Consumes Layer 2 (spectral diagnostics) and Layer 3 (instability predictor).
  Does not import or modify the decoder (Layer 5).

Consumes:
  - v6.0 NB spectral radius, eigenvectors
  - v6.1 IPR localization
  - v7.2 spectral instability score

Does not modify decoder internals.  Fully deterministic: no randomness,
no global state, no input mutation.  All floats rounded to 12 decimals.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from qec.diagnostics.non_backtracking_spectrum import (
    compute_non_backtracking_spectrum,
    _build_sparse_nb_matrix,
)
from qec.diagnostics.nb_localization import (
    compute_nb_localization_metrics,
)
from qec.experiments.tanner_graph_repair import (
    _extract_edges,
    _build_edge_set,
    _apply_swap,
    _edges_to_H,
)
from qec.experiments.spectral_instability_phase_map import (
    compute_spectral_instability_score,
)


# ── Internal: NB spectrum with eigenvectors ─────────────────────────


def _compute_nb_eigenvector(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute NB eigenvalues and eigenvectors for a parity-check matrix.

    Returns the dominant eigenvector (by magnitude) and the directed
    edge list used for eigenvector-to-edge mapping.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dominant_eigenvector : np.ndarray
        Eigenvector corresponding to the largest-magnitude eigenvalue.
    directed_edges : np.ndarray
        Directed edge list as (num_directed, 2) array.
    """
    H = np.asarray(H, dtype=np.float64)

    B_sparse, directed_edges = _build_sparse_nb_matrix(H)
    num_directed = len(directed_edges)
    if num_directed == 0:
        return np.array([]), np.array([])

    # Use dense eigensolver for deterministic eigenvectors.
    # Sparse construction ensures O(|E|) build cost; dense eigensolver
    # is needed because ARPACK eigs is non-deterministic for eigenvectors
    # of non-symmetric matrices.
    eigenvalues, eigenvectors = np.linalg.eig(B_sparse.toarray())

    magnitudes = np.abs(eigenvalues)

    # Deterministic sort: magnitude descending, real desc, imag desc.
    sort_keys = np.lexsort((
        -eigenvalues.imag,
        -eigenvalues.real,
        -magnitudes,
    ))

    dominant_vec = eigenvectors[:, sort_keys[0]]
    return dominant_vec, np.array(directed_edges)


# ── Core Feature 1: Instability Score Evaluation ────────────────────


def _compute_graph_instability_score(H: np.ndarray) -> dict[str, float]:
    """Compute graph instability score from NB spectrum.

    Evaluates:
    - NB spectral radius
    - spectral gap
    - IPR localization
    - instability ratio

    Returns a scalar instability score and component metrics.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, float]
        Dictionary with instability score and component metrics.
    """
    m, n = H.shape

    nb_result = compute_non_backtracking_spectrum(H)
    spectral_radius = nb_result.get("spectral_radius", 0.0)

    # Spectral gap: difference between first and second eigenvalue magnitudes.
    nb_eigs = nb_result.get("nb_eigenvalues", [])
    if len(nb_eigs) >= 2:
        mag_0 = math.sqrt(nb_eigs[0][0] ** 2 + nb_eigs[0][1] ** 2)
        mag_1 = math.sqrt(nb_eigs[1][0] ** 2 + nb_eigs[1][1] ** 2)
        spectral_gap = round(mag_0 - mag_1, 12)
    else:
        spectral_gap = 0.0

    # IPR localization.
    loc_result = compute_nb_localization_metrics(H, num_leading_modes=1)
    ipr_score = loc_result.get("max_ipr", 0.0)

    # Instability ratio.
    col_sums = np.sum(np.asarray(H, dtype=np.float64), axis=0)
    row_sums = np.sum(np.asarray(H, dtype=np.float64), axis=1)
    avg_var_degree = round(float(np.mean(col_sums)), 12) if n > 0 else 0.0
    avg_chk_degree = round(float(np.mean(row_sums)), 12) if m > 0 else 0.0
    avg_degree = (avg_var_degree + avg_chk_degree) / 2.0

    if avg_degree > 0.0:
        instability_ratio = round(spectral_radius / math.sqrt(avg_degree), 12)
    else:
        instability_ratio = 0.0

    # Composite instability score via v7.2 machinery.
    score = compute_spectral_instability_score(
        nb_spectral_radius=spectral_radius,
        spectral_instability_ratio=instability_ratio,
        ipr_localization_score=ipr_score,
        cluster_risk_scores=[],
        avg_variable_degree=avg_var_degree,
        avg_check_degree=avg_chk_degree,
    )

    return {
        "instability_score": round(score, 12),
        "spectral_radius": round(spectral_radius, 12),
        "spectral_gap": round(spectral_gap, 12),
        "ipr_localization": round(ipr_score, 12),
        "instability_ratio": round(instability_ratio, 12),
        "avg_variable_degree": avg_var_degree,
        "avg_check_degree": avg_chk_degree,
    }


# ── Core Feature 2: Spectral Edge Gradient ─────────────────────────


def _compute_spectral_edge_gradient(
    H: np.ndarray,
    eigenvector: np.ndarray | None = None,
) -> list[tuple[tuple[int, int], float]]:
    """Compute spectral edge gradient for instability attribution.

    Edge importance approximation:
        edge_gradient(i, j) = |v_i| * |v_j|

    where v is the dominant NB eigenvector.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    eigenvector : np.ndarray or None
        Pre-computed dominant NB eigenvector.  If None, computed
        internally.

    Returns
    -------
    list[tuple[tuple[int, int], float]]
        List of ((variable_node, check_node), gradient_score) tuples,
        sorted by gradient DESC, then edge (var, chk) ASC.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    if eigenvector is None or len(eigenvector) == 0:
        eigenvector, directed_edges_arr = _compute_nb_eigenvector(H_arr)
        if len(eigenvector) == 0:
            return []
    else:
        # Build directed edges to map eigenvector to undirected edges.
        directed_edges_list: list[tuple[int, int]] = []
        for ci in range(m):
            for vi in range(n):
                if H_arr[ci, vi] != 0:
                    directed_edges_list.append((vi, n + ci))
                    directed_edges_list.append((n + ci, vi))
        directed_edges_arr = np.array(directed_edges_list)

    if len(eigenvector) == 0:
        return []

    abs_v = np.abs(eigenvector)
    num_directed = len(directed_edges_arr)

    # Map each undirected edge to its gradient.
    # Undirected edges at positions 2*i (fwd) and 2*i+1 (rev).
    edge_gradients: list[tuple[tuple[int, int], float]] = []
    num_undirected = num_directed // 2

    for i in range(num_undirected):
        fwd_idx = 2 * i
        rev_idx = 2 * i + 1
        src = int(directed_edges_arr[fwd_idx, 0])
        dst = int(directed_edges_arr[fwd_idx, 1])

        # Use maximum of forward and reverse eigenvector components.
        grad = round(float(abs_v[fwd_idx] * abs_v[rev_idx]), 12)

        # Normalize to (variable_node, check_node) order.
        if src < n:
            edge = (src, dst)
        else:
            edge = (dst, src)

        edge_gradients.append((edge, grad))

    # Sort: gradient DESC, then edge ASC (deterministic).
    edge_gradients.sort(key=lambda x: (-x[1], x[0]))

    return edge_gradients


# ── Core Feature 3: Guided Repair Candidate Generation ──────────────


def _generate_guided_repair_candidates(
    H: np.ndarray,
    gradient: list[tuple[tuple[int, int], float]],
    *,
    max_candidates: int = 10,
    top_k_fraction: float = 0.5,
) -> list[dict[str, Any]]:
    """Generate degree-preserving swap candidates guided by spectral gradient.

    Selects high-gradient edges and generates swaps with low-gradient
    edges to redistribute spectral mass.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    gradient : list[tuple[tuple[int, int], float]]
        Sorted edge gradient list from _compute_spectral_edge_gradient.
    max_candidates : int
        Maximum number of candidates to generate.
    top_k_fraction : float
        Fraction of edges considered "high gradient".

    Returns
    -------
    list[dict[str, Any]]
        List of swap candidate descriptors.
    """
    m, n = H.shape

    if not gradient:
        return []

    num_edges = len(gradient)
    top_k = max(1, int(num_edges * top_k_fraction))
    high_gradient_edges = [e for e, _ in gradient[:top_k]]
    low_gradient_edges = [e for e, _ in gradient[top_k:]]

    if not low_gradient_edges:
        # All edges are high-gradient; split in half.
        mid = max(1, num_edges // 2)
        high_gradient_edges = [e for e, _ in gradient[:mid]]
        low_gradient_edges = [e for e, _ in gradient[mid:]]

    edge_set = set(e for e, _ in gradient)
    candidates: list[dict[str, Any]] = []

    for v1, c1 in high_gradient_edges:
        # Ensure v1 is variable, c1 is check.
        if v1 >= n:
            v1, c1 = c1, v1
        if v1 >= n:
            continue

        for v2, c2 in low_gradient_edges:
            if v2 >= n:
                v2, c2 = c2, v2
            if v2 >= n:
                continue

            if v1 == v2 or c1 == c2:
                continue

            new_e1 = (min(v1, c2), max(v1, c2))
            new_e2 = (min(v2, c1), max(v2, c1))

            if new_e1 in edge_set or new_e2 in edge_set:
                continue

            candidates.append({
                "remove": [(v1, c1), (v2, c2)],
                "add": [list(new_e1), list(new_e2)],
                "description": (
                    f"swap ({v1},{c1})+({v2},{c2})"
                    f" -> ({new_e1[0]},{new_e1[1]})"
                    f"+({new_e2[0]},{new_e2[1]})"
                ),
            })

            if len(candidates) >= max_candidates:
                return candidates

    return candidates


# ── Core Feature 4: Belief Curvature Diagnostic ─────────────────────


def _compute_belief_curvature(trace: list[float]) -> dict[str, Any]:
    """Compute discrete belief curvature from a BP LLR trace.

    Uses the stencil [1, -2, 1] to detect oscillatory BP trajectories.

    Interpretation:
    - stable convergence → curvature ≈ 0
    - divergence → moderate curvature
    - trapping oscillation → large alternating curvature

    This is diagnostic only — never modifies BP messages.

    Parameters
    ----------
    trace : list[float]
        Per-iteration LLR values for a single variable node.

    Returns
    -------
    dict[str, Any]
        Curvature analysis with keys:
        - curvature_values: list of curvature at each interior point
        - max_abs_curvature: float
        - mean_abs_curvature: float
        - oscillation_detected: bool (alternating sign curvature)
        - num_sign_changes: int
    """
    n = len(trace)
    if n < 3:
        return {
            "curvature_values": [],
            "max_abs_curvature": 0.0,
            "mean_abs_curvature": 0.0,
            "oscillation_detected": False,
            "num_sign_changes": 0,
        }

    curvatures: list[float] = []
    for i in range(1, n - 1):
        c = round(trace[i - 1] - 2.0 * trace[i] + trace[i + 1], 12)
        curvatures.append(c)

    abs_curvatures = [abs(c) for c in curvatures]
    max_abs = round(max(abs_curvatures), 12) if abs_curvatures else 0.0
    mean_abs = round(sum(abs_curvatures) / len(abs_curvatures), 12) if abs_curvatures else 0.0

    # Count sign changes to detect oscillation.
    sign_changes = 0
    for i in range(1, len(curvatures)):
        if curvatures[i - 1] * curvatures[i] < 0:
            sign_changes += 1

    # Oscillation: majority of curvature values alternate sign.
    oscillation = sign_changes > len(curvatures) // 2 if curvatures else False

    return {
        "curvature_values": curvatures,
        "max_abs_curvature": max_abs,
        "mean_abs_curvature": mean_abs,
        "oscillation_detected": oscillation,
        "num_sign_changes": sign_changes,
    }


# ── Core Feature 5: Ternary Reliability Diagnostics ─────────────────


def _classify_ternary_llr(
    llr_values: list[float],
    *,
    threshold: float = 1.0,
) -> list[int]:
    """Classify LLR values into ternary reliability labels.

    Maps LLR values to:
        +1 → strong convergence to 0 (LLR > threshold)
        -1 → strong convergence to 1 (LLR < -threshold)
         0 → metastable / oscillating (|LLR| <= threshold)

    Parameters
    ----------
    llr_values : list[float]
        LLR values to classify.
    threshold : float
        Classification threshold for strong convergence.

    Returns
    -------
    list[int]
        Ternary labels: +1, -1, or 0 for each LLR value.
    """
    result: list[int] = []
    for v in llr_values:
        if v > threshold:
            result.append(1)
        elif v < -threshold:
            result.append(-1)
        else:
            result.append(0)
    return result


def _compute_ternary_cluster_stats(
    labels: list[int],
) -> dict[str, Any]:
    """Compute statistics from ternary classification labels.

    Parameters
    ----------
    labels : list[int]
        Ternary labels from _classify_ternary_llr.

    Returns
    -------
    dict[str, Any]
        Statistics including counts and fractions for each class.
    """
    n = len(labels)
    if n == 0:
        return {
            "num_strong_zero": 0,
            "num_strong_one": 0,
            "num_metastable": 0,
            "fraction_strong_zero": 0.0,
            "fraction_strong_one": 0.0,
            "fraction_metastable": 0.0,
            "total": 0,
        }

    strong_zero = sum(1 for l in labels if l == 1)
    strong_one = sum(1 for l in labels if l == -1)
    metastable = sum(1 for l in labels if l == 0)

    return {
        "num_strong_zero": strong_zero,
        "num_strong_one": strong_one,
        "num_metastable": metastable,
        "fraction_strong_zero": round(strong_zero / n, 12),
        "fraction_strong_one": round(strong_one / n, 12),
        "fraction_metastable": round(metastable / n, 12),
        "total": n,
    }


# ── Core: Deterministic Graph Optimization Loop ─────────────────────


def run_spectral_graph_optimization(
    H: np.ndarray,
    *,
    max_iterations: int = 10,
    improvement_threshold: float = 1e-6,
    instability_target: float = 0.1,
    max_candidates: int = 10,
    top_k_fraction: float = 0.5,
    bp_traces: list[list[float]] | None = None,
    final_llr: list[float] | None = None,
    ternary_threshold: float = 1.0,
) -> dict[str, Any]:
    """Run the deterministic spectral graph optimization loop.

    Algorithm:
        H_current = H_initial
        score_current = instability(H_current)

        for iteration in range(max_iterations):
            candidates = generate guided repair candidates
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
        Fraction of edges considered high-gradient.
    bp_traces : list[list[float]] or None
        Optional per-variable BP LLR traces for curvature analysis.
    final_llr : list[float] or None
        Optional final LLR values for ternary classification.
    ternary_threshold : float
        Threshold for ternary LLR classification.

    Returns
    -------
    dict[str, Any]
        JSON-serializable optimization artifact.
    """
    H_current = np.array(H, dtype=np.float64)
    m, n = H_current.shape

    # Initial instability evaluation.
    initial_metrics = _compute_graph_instability_score(H_current)
    score_current = initial_metrics["instability_score"]
    initial_score = score_current

    swaps_applied: list[dict[str, Any]] = []
    iterations_run = 0

    for iteration in range(max_iterations):
        iterations_run = iteration + 1

        # Check if we've reached the target.
        if score_current <= instability_target:
            break

        # Compute spectral gradient for current graph.
        gradient = _compute_spectral_edge_gradient(H_current)

        if not gradient:
            break

        # Generate guided repair candidates.
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
            trial_metrics = _compute_graph_instability_score(H_trial)
            trial_score = trial_metrics["instability_score"]

            if trial_score < best_score:
                best_score = trial_score
                best_candidate = candidate
                best_H = H_trial

        # Check improvement.
        improvement = round(score_current - best_score, 12)
        if improvement <= improvement_threshold or best_candidate is None:
            break

        # Apply best repair.
        swaps_applied.append({
            "iteration": iteration,
            "swap": {
                "remove": [list(e) for e in best_candidate["remove"]],
                "add": best_candidate["add"],
                "description": best_candidate["description"],
            },
            "score_before": round(score_current, 12),
            "score_after": round(best_score, 12),
            "improvement": improvement,
        })

        H_current = best_H
        score_current = best_score

    # Final instability evaluation.
    final_metrics = _compute_graph_instability_score(H_current)
    final_score = final_metrics["instability_score"]

    # Spectral radius reduction.
    spectral_radius_reduction = round(
        initial_metrics["spectral_radius"] - final_metrics["spectral_radius"],
        12,
    )

    # Optional curvature diagnostics.
    curvature_events_detected = 0
    if bp_traces is not None:
        for trace in bp_traces:
            curv = _compute_belief_curvature(trace)
            if curv["oscillation_detected"]:
                curvature_events_detected += 1

    # Optional ternary classification.
    ternary_clusters_detected = 0
    if final_llr is not None:
        ternary_labels = _classify_ternary_llr(
            final_llr, threshold=ternary_threshold,
        )
        ternary_stats = _compute_ternary_cluster_stats(ternary_labels)
        ternary_clusters_detected = ternary_stats["num_metastable"]

    optimizer_success = final_score < initial_score

    return {
        "initial_instability_score": round(initial_score, 12),
        "final_instability_score": round(final_score, 12),
        "iterations": iterations_run,
        "swaps_applied": swaps_applied,
        "spectral_radius_reduction": spectral_radius_reduction,
        "optimizer_success": optimizer_success,
        "curvature_events_detected": curvature_events_detected,
        "ternary_clusters_detected": ternary_clusters_detected,
        "initial_metrics": {
            k: round(v, 12) if isinstance(v, float) else v
            for k, v in initial_metrics.items()
        },
        "final_metrics": {
            k: round(v, 12) if isinstance(v, float) else v
            for k, v in final_metrics.items()
        },
    }
