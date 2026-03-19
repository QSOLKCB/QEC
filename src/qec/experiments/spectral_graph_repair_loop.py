"""
v7.3.0 — Deterministic Spectral Graph Repair Loop.

Attempts to reduce predicted BP instability before decoding by testing
small deterministic graph rewrites on the Tanner graph.

Pipeline:
  baseline graph
  → spectral diagnostics
  → instability prediction
  → generate repair candidates
  → evaluate repaired candidates (single-step or multi-step)
  → choose best repair (sequence)
  → optional baseline decode
  → optional repaired decode
  → compare outcomes

v7.3.x adds multi-step repair search with spectral-bound pruning:
  - depth=1: single-swap search (v7.3.0 behavior)
  - depth=2: two-step repair sequences with branch-and-bound pruning

Consumes:
  - v6.0 NB spectral radius
  - v6.1 IPR localization
  - v6.2 trapping-set candidates
  - v6.4 spectral failure risk
  - v7.2 spectral instability score

Does not modify decoder internals.  Fully deterministic: no randomness,
no global state, no input mutation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from qec.experiments.tanner_graph_repair import (
    _extract_edges,
    _build_edge_set,
    _build_adjacency,
    _get_cluster_edges,
    _get_boundary_edges,
    _generate_candidate_swaps,
    _apply_swap,
    _edges_to_H,
    _experimental_bp_flooding,
    _compute_syndrome,
)
from qec.experiments.spectral_instability_phase_map import (
    compute_spectral_instability_score,
)


# ── Core Feature 1 — Deterministic Repair Candidate Generation ───────


def generate_repair_candidates(
    H: np.ndarray,
    top_risk_clusters: list[int],
    candidate_clusters: list[dict[str, Any]],
    node_risk_scores: list[list],
    cluster_risk_scores: list[float],
    max_candidates: int = 10,
) -> dict[str, Any]:
    """Generate deterministic degree-preserving edge-swap repair candidates.

    Identifies the highest-risk cluster and generates edge swaps between
    cluster edges and boundary edges.  Candidates are ordered
    deterministically: cluster edges sorted, boundary edges sorted,
    outer product in that order.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    top_risk_clusters : list[int]
        Cluster indices sorted by risk descending (from v6.4).
    candidate_clusters : list[dict[str, Any]]
        Cluster definitions from v6.2, each with ``variable_nodes``
        and ``check_nodes``.
    node_risk_scores : list[list]
        Per-node risk scores ``[[node_idx, score], ...]`` from v6.4.
    cluster_risk_scores : list[float]
        Per-cluster risk scores from v6.4.
    max_candidates : int
        Maximum number of candidate swaps to generate.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``candidates``: list of swap descriptors
        - ``cluster_nodes``: sorted list of cluster node indices
        - ``cluster_risk_score``: float risk score for selected cluster
        - ``selected_cluster_index``: int index of selected cluster
    """
    m, n = H.shape

    if not top_risk_clusters or not candidate_clusters:
        return {
            "candidates": [],
            "cluster_nodes": [],
            "cluster_risk_score": 0.0,
            "selected_cluster_index": -1,
        }

    # Select highest-risk cluster.
    cluster_idx = top_risk_clusters[0]
    if cluster_idx >= len(candidate_clusters):
        return {
            "candidates": [],
            "cluster_nodes": [],
            "cluster_risk_score": 0.0,
            "selected_cluster_index": cluster_idx,
        }

    cluster = candidate_clusters[cluster_idx]
    cluster_var_nodes = sorted(cluster.get("variable_nodes", []))
    cluster_chk_nodes = sorted(cluster.get("check_nodes", []))

    if not cluster_var_nodes:
        return {
            "candidates": [],
            "cluster_nodes": [],
            "cluster_risk_score": (
                cluster_risk_scores[cluster_idx]
                if cluster_idx < len(cluster_risk_scores) else 0.0
            ),
            "selected_cluster_index": cluster_idx,
        }

    # Build cluster node set: variable nodes as-is, check nodes offset.
    cluster_nodes: set[int] = set(cluster_var_nodes)
    for ci in cluster_chk_nodes:
        cluster_nodes.add(n + ci)

    # Also include adjacent check nodes from H for variable-only clusters.
    for vi in cluster_var_nodes:
        for ci in range(m):
            if H[ci, vi] != 0:
                cluster_nodes.add(n + ci)

    edges = _extract_edges(H)
    edge_set = _build_edge_set(edges)
    adjacency = _build_adjacency(edges)

    cluster_edges = _get_cluster_edges(edges, cluster_nodes)
    boundary_edges = _get_boundary_edges(edges, cluster_nodes, adjacency)

    candidates = _generate_candidate_swaps(
        cluster_edges, boundary_edges, edge_set, n,
        max_candidates=max_candidates,
    )

    cluster_risk = (
        cluster_risk_scores[cluster_idx]
        if cluster_idx < len(cluster_risk_scores) else 0.0
    )

    return {
        "candidates": candidates,
        "cluster_nodes": sorted(cluster_nodes),
        "cluster_risk_score": round(float(cluster_risk), 12),
        "selected_cluster_index": cluster_idx,
    }


# ── Core Feature 2 — Predicted Instability Scoring ──────────────────


def score_repair_candidate(
    H: np.ndarray,
    candidate: dict[str, Any],
    nb_spectral_radius: float,
    spectral_instability_ratio: float,
    ipr_localization_score: float,
    cluster_risk_scores: list[float],
    avg_variable_degree: float,
    avg_check_degree: float,
    instability_score_before: float,
    predicted_instability_before: bool,
    *,
    instability_threshold: float = 0.5,
) -> dict[str, Any]:
    """Score a repair candidate by recomputing instability prediction.

    For the candidate repaired graph, recomputes the spectral
    instability score using the same v7.2 scoring machinery, then
    compares with the baseline score.

    Parameters
    ----------
    H : np.ndarray
        Original parity-check matrix, shape (m, n).
    candidate : dict[str, Any]
        Swap descriptor with ``remove`` and ``add`` keys.
    nb_spectral_radius : float
        Baseline NB spectral radius.
    spectral_instability_ratio : float
        Baseline spectral instability ratio.
    ipr_localization_score : float
        Baseline IPR localization score.
    cluster_risk_scores : list[float]
        Baseline per-cluster risk scores.
    avg_variable_degree : float
        Average variable-node degree.
    avg_check_degree : float
        Average check-node degree.
    instability_score_before : float
        Baseline spectral instability score.
    predicted_instability_before : bool
        Baseline instability prediction.
    instability_threshold : float
        Threshold for instability prediction.

    Returns
    -------
    dict[str, Any]
        Per-candidate metrics dictionary.
    """
    m, n = H.shape

    # Apply swap to get repaired edge list and H.
    edges = _extract_edges(H)
    repaired_edges = _apply_swap(edges, candidate)
    H_repaired = _edges_to_H(repaired_edges, m, n)

    # Recompute NB spectral radius for repaired graph.
    from qec.diagnostics.non_backtracking_spectrum import (
        compute_non_backtracking_spectrum,
    )
    nb_result = compute_non_backtracking_spectrum(H_repaired)
    repaired_spectral_radius = nb_result.get("spectral_radius", 0.0)

    # Recompute instability ratio.
    repaired_avg_degree = (avg_variable_degree + avg_check_degree) / 2.0
    if repaired_avg_degree > 0.0:
        repaired_threshold = math.sqrt(repaired_avg_degree)
        repaired_instability_ratio = round(
            repaired_spectral_radius / repaired_threshold, 12,
        )
    else:
        repaired_instability_ratio = 0.0

    # Recompute spectral instability score using v7.2 machinery.
    repaired_score = compute_spectral_instability_score(
        nb_spectral_radius=repaired_spectral_radius,
        spectral_instability_ratio=repaired_instability_ratio,
        ipr_localization_score=ipr_localization_score,
        cluster_risk_scores=cluster_risk_scores,
        avg_variable_degree=avg_variable_degree,
        avg_check_degree=avg_check_degree,
    )

    predicted_instability_after = repaired_score > instability_threshold
    score_improvement = round(instability_score_before - repaired_score, 12)

    return {
        "swap": {
            "remove": [list(e) for e in candidate["remove"]],
            "add": candidate["add"],
            "description": candidate["description"],
        },
        "spectral_instability_score_before": round(instability_score_before, 12),
        "spectral_instability_score_after": round(repaired_score, 12),
        "predicted_instability_before": predicted_instability_before,
        "predicted_instability_after": predicted_instability_after,
        "score_improvement": score_improvement,
        "repaired_spectral_radius": round(repaired_spectral_radius, 12),
    }


# ── Core Feature 3 — Best Repair Selection ──────────────────────────


def select_best_repair(
    scored_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the best repair candidate deterministically.

    Primary objective: maximize score_improvement.

    Tie-breaking order:
      1. Largest instability score reduction (score_improvement desc)
      2. Lowest repaired instability score (score_after asc)
      3. Lexicographically smallest swap description (asc)

    Parameters
    ----------
    scored_candidates : list[dict[str, Any]]
        List of scored candidate metrics from
        :func:`score_repair_candidate`.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:

        - ``best_swap``: swap descriptor or None
        - ``best_candidate_metrics``: full metrics for best or None
        - ``num_candidates_evaluated``: int
    """
    num_evaluated = len(scored_candidates)

    if not scored_candidates:
        return {
            "best_swap": None,
            "best_candidate_metrics": None,
            "num_candidates_evaluated": 0,
        }

    # Filter to candidates with positive improvement.
    improving = [c for c in scored_candidates if c["score_improvement"] > 0.0]

    if not improving:
        return {
            "best_swap": None,
            "best_candidate_metrics": None,
            "num_candidates_evaluated": num_evaluated,
        }

    # Deterministic sort: score_improvement desc, score_after asc,
    # description asc.
    improving.sort(key=lambda c: (
        -c["score_improvement"],
        c["spectral_instability_score_after"],
        c["swap"]["description"],
    ))

    best = improving[0]

    return {
        "best_swap": best["swap"],
        "best_candidate_metrics": best,
        "num_candidates_evaluated": num_evaluated,
    }


# ── Multi-Step Repair Search with Spectral-Bound Pruning ────────────


def _compute_instability_score_for_H(
    H_candidate: np.ndarray,
    ipr_localization_score: float,
    cluster_risk_scores: list[float],
    avg_variable_degree: float,
    avg_check_degree: float,
) -> float:
    """Compute spectral instability score for a candidate parity matrix."""
    from qec.diagnostics.non_backtracking_spectrum import (
        compute_non_backtracking_spectrum,
    )
    nb_result = compute_non_backtracking_spectrum(H_candidate)
    repaired_spectral_radius = nb_result.get("spectral_radius", 0.0)

    repaired_avg_degree = (avg_variable_degree + avg_check_degree) / 2.0
    if repaired_avg_degree > 0.0:
        repaired_threshold = math.sqrt(repaired_avg_degree)
        repaired_instability_ratio = round(
            repaired_spectral_radius / repaired_threshold, 12,
        )
    else:
        repaired_instability_ratio = 0.0

    return compute_spectral_instability_score(
        nb_spectral_radius=repaired_spectral_radius,
        spectral_instability_ratio=repaired_instability_ratio,
        ipr_localization_score=ipr_localization_score,
        cluster_risk_scores=cluster_risk_scores,
        avg_variable_degree=avg_variable_degree,
        avg_check_degree=avg_check_degree,
    )


def _prune_branch(
    baseline_score: float,
    partial_score: float,
    best_improvement: float,
) -> bool:
    """Determine whether to prune a multi-step branch.

    If the upper bound on improvement from this branch (assuming
    step-2 can only maintain or worsen the partial score) is not
    better than the best complete sequence found so far, prune.
    """
    upper_bound = round(baseline_score - partial_score, 12)
    return upper_bound <= best_improvement


def _generate_repair_sequences(
    H: np.ndarray,
    candidates: list[dict[str, Any]],
    baseline_score: float,
    ipr_localization_score: float,
    cluster_risk_scores: list[float],
    avg_variable_degree: float,
    avg_check_degree: float,
    instability_threshold: float,
    predicted_instability_before: bool,
    max_depth: int,
    enable_pruning: bool,
    risk_result: dict[str, Any],
    max_candidates: int,
    candidate_clusters: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Generate and score deterministic multi-step repair sequences.

    For depth=1, scores each single candidate (identical to v7.3.0).
    For depth=2, explores two-step sequences with optional pruning.

    Returns
    -------
    tuple[list[dict[str, Any]], int]
        (scored_sequences, branches_pruned)
    """
    m, n = H.shape
    branches_pruned = 0

    if max_depth <= 1:
        # Single-step: score each candidate directly.
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            edges = _extract_edges(H)
            repaired_edges = _apply_swap(edges, candidate)
            H_repaired = _edges_to_H(repaired_edges, m, n)

            repaired_score = _compute_instability_score_for_H(
                H_repaired, ipr_localization_score, cluster_risk_scores,
                avg_variable_degree, avg_check_degree,
            )
            predicted_after = repaired_score > instability_threshold
            improvement = round(baseline_score - repaired_score, 12)

            scored.append({
                "repair_sequence": [
                    {
                        "remove": [list(e) for e in candidate["remove"]],
                        "add": candidate["add"],
                        "description": candidate["description"],
                    },
                ],
                "spectral_instability_score_before": round(baseline_score, 12),
                "spectral_instability_score_after": round(repaired_score, 12),
                "predicted_instability_before": predicted_instability_before,
                "predicted_instability_after": predicted_after,
                "score_improvement": improvement,
                "sequence_length": 1,
            })
        return scored, branches_pruned

    # Multi-step (depth >= 2): branch-and-bound.
    best_improvement = 0.0
    scored_sequences: list[dict[str, Any]] = []

    for swap1 in candidates:
        edges_0 = _extract_edges(H)
        edges_1 = _apply_swap(edges_0, swap1)
        H_1 = _edges_to_H(edges_1, m, n)

        score_after_swap1 = _compute_instability_score_for_H(
            H_1, ipr_localization_score, cluster_risk_scores,
            avg_variable_degree, avg_check_degree,
        )

        # Record depth-1 sequence.
        improvement_1 = round(baseline_score - score_after_swap1, 12)
        predicted_after_1 = score_after_swap1 > instability_threshold

        seq1_entry = {
            "repair_sequence": [
                {
                    "remove": [list(e) for e in swap1["remove"]],
                    "add": swap1["add"],
                    "description": swap1["description"],
                },
            ],
            "spectral_instability_score_before": round(baseline_score, 12),
            "spectral_instability_score_after": round(score_after_swap1, 12),
            "predicted_instability_before": predicted_instability_before,
            "predicted_instability_after": predicted_after_1,
            "score_improvement": improvement_1,
            "sequence_length": 1,
        }
        scored_sequences.append(seq1_entry)
        if improvement_1 > best_improvement:
            best_improvement = improvement_1

        # Pruning check before exploring depth-2.
        if enable_pruning and _prune_branch(
            baseline_score, score_after_swap1, best_improvement,
        ):
            branches_pruned += 1
            continue

        # Generate step-2 candidates from the repaired graph H_1.
        gen_2 = generate_repair_candidates(
            H_1,
            top_risk_clusters=risk_result.get("top_risk_clusters", []),
            candidate_clusters=candidate_clusters,
            node_risk_scores=risk_result.get("node_risk_scores", []),
            cluster_risk_scores=risk_result.get("cluster_risk_scores", []),
            max_candidates=max_candidates,
        )
        candidates_2 = gen_2["candidates"]

        for swap2 in candidates_2:
            edges_2 = _apply_swap(edges_1, swap2)
            H_2 = _edges_to_H(edges_2, m, n)

            score_after_swap2 = _compute_instability_score_for_H(
                H_2, ipr_localization_score, cluster_risk_scores,
                avg_variable_degree, avg_check_degree,
            )
            improvement_2 = round(baseline_score - score_after_swap2, 12)
            predicted_after_2 = score_after_swap2 > instability_threshold

            seq2_entry = {
                "repair_sequence": [
                    {
                        "remove": [list(e) for e in swap1["remove"]],
                        "add": swap1["add"],
                        "description": swap1["description"],
                    },
                    {
                        "remove": [list(e) for e in swap2["remove"]],
                        "add": swap2["add"],
                        "description": swap2["description"],
                    },
                ],
                "spectral_instability_score_before": round(baseline_score, 12),
                "spectral_instability_score_after": round(score_after_swap2, 12),
                "predicted_instability_before": predicted_instability_before,
                "predicted_instability_after": predicted_after_2,
                "score_improvement": improvement_2,
                "sequence_length": 2,
            }
            scored_sequences.append(seq2_entry)
            if improvement_2 > best_improvement:
                best_improvement = improvement_2

    return scored_sequences, branches_pruned


def _select_best_repair_sequence(
    scored_sequences: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select the best repair sequence deterministically.

    Tie-breaking:
      1. score_improvement DESC
      2. spectral_instability_score_after ASC
      3. swap1 description ASC
      4. swap2 description ASC (empty string if depth-1)
    """
    num_evaluated = len(scored_sequences)
    if not scored_sequences:
        return {
            "best_sequence": None,
            "best_sequence_metrics": None,
            "num_candidates_evaluated": 0,
        }

    improving = [s for s in scored_sequences if s["score_improvement"] > 0.0]
    if not improving:
        return {
            "best_sequence": None,
            "best_sequence_metrics": None,
            "num_candidates_evaluated": num_evaluated,
        }

    def _sort_key(s: dict[str, Any]) -> tuple:
        seq = s["repair_sequence"]
        desc1 = seq[0]["description"] if len(seq) > 0 else ""
        desc2 = seq[1]["description"] if len(seq) > 1 else ""
        return (
            -s["score_improvement"],
            s["spectral_instability_score_after"],
            desc1,
            desc2,
        )

    improving.sort(key=_sort_key)
    best = improving[0]

    return {
        "best_sequence": best["repair_sequence"],
        "best_sequence_metrics": best,
        "num_candidates_evaluated": num_evaluated,
    }


# ── Core Feature 4 — Spectral Repair Loop Experiment ────────────────


def run_spectral_graph_repair_loop(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    risk_result: dict[str, Any],
    *,
    nb_spectral_radius: float = 0.0,
    spectral_instability_ratio: float = 0.0,
    ipr_localization_score: float = 0.0,
    avg_variable_degree: float = 0.0,
    avg_check_degree: float = 0.0,
    instability_threshold: float = 0.5,
    max_candidates: int = 10,
    max_iters: int = 100,
    enable_decode_comparison: bool = True,
    candidate_clusters: list[dict[str, Any]] | None = None,
    max_repair_depth: int = 1,
    enable_multistep_repair: bool = False,
    enable_pruning: bool = True,
) -> dict[str, Any]:
    """Run the spectral graph repair loop experiment.

    Full pipeline:
      baseline graph
      → spectral diagnostics (reused from inputs)
      → instability prediction
      → generate repair candidates
      → evaluate repaired candidates (single or multi-step)
      → choose best repair (sequence)
      → optional baseline decode
      → optional repaired decode
      → compare outcomes

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    llr : np.ndarray
        Per-variable log-likelihood ratios, length n.
    syndrome_vec : np.ndarray
        Binary syndrome vector, length m.
    risk_result : dict[str, Any]
        Output of ``compute_spectral_failure_risk()`` (v6.4).
    nb_spectral_radius : float
        NB spectral radius from v6.0 diagnostics.
    spectral_instability_ratio : float
        Spectral instability ratio from v6.8 predictor.
    ipr_localization_score : float
        IPR localization score from v6.1.
    avg_variable_degree : float
        Average variable-node degree.
    avg_check_degree : float
        Average check-node degree.
    instability_threshold : float
        Threshold for instability prediction.
    max_candidates : int
        Maximum number of candidate swaps.
    max_iters : int
        Maximum BP iterations for decode comparison.
    enable_decode_comparison : bool
        If True, run baseline and repaired decodes for comparison.
    candidate_clusters : list[dict[str, Any]] or None
        Cluster definitions from v6.2.  If None, extracted from
        risk_result (if available).
    max_repair_depth : int
        Maximum repair sequence depth.  1 = single-swap (v7.3.0),
        2 = two-step sequences.  Default 1.
    enable_multistep_repair : bool
        If True, enables multi-step repair search.  Default False.
    enable_pruning : bool
        If True, enables spectral-bound branch pruning for
        multi-step search.  Default True.

    Returns
    -------
    dict[str, Any]
        JSON-serializable artifact with repair loop results.
    """
    m, n = H.shape
    llr = np.asarray(llr, dtype=np.float64)
    syndrome_vec = np.asarray(syndrome_vec, dtype=np.uint8)

    node_risk_scores = risk_result.get("node_risk_scores", [])
    cluster_risk_scores_raw = risk_result.get("cluster_risk_scores", [])
    top_risk_clusters = risk_result.get("top_risk_clusters", [])

    # Resolve candidate clusters.
    if candidate_clusters is None:
        candidate_clusters = risk_result.get("candidate_clusters", [])

    # Effective depth: only use multi-step if explicitly enabled.
    effective_depth = max_repair_depth if enable_multistep_repair else 1

    # Compute baseline instability score.
    instability_score_before = compute_spectral_instability_score(
        nb_spectral_radius=nb_spectral_radius,
        spectral_instability_ratio=spectral_instability_ratio,
        ipr_localization_score=ipr_localization_score,
        cluster_risk_scores=cluster_risk_scores_raw,
        avg_variable_degree=avg_variable_degree,
        avg_check_degree=avg_check_degree,
    )
    predicted_instability_before = instability_score_before > instability_threshold

    # ── Step 1: Generate repair candidates ────────────────────────
    gen_result = generate_repair_candidates(
        H,
        top_risk_clusters=top_risk_clusters,
        candidate_clusters=candidate_clusters,
        node_risk_scores=node_risk_scores,
        cluster_risk_scores=cluster_risk_scores_raw,
        max_candidates=max_candidates,
    )

    candidates = gen_result["candidates"]
    cluster_nodes = gen_result["cluster_nodes"]
    cluster_risk_score = gen_result["cluster_risk_score"]

    # Multistep artifact fields.
    multistep_fields = {
        "repair_depth": effective_depth,
        "multistep_enabled": enable_multistep_repair,
        "pruning_enabled": enable_pruning and enable_multistep_repair,
        "branches_pruned": 0,
    }

    # ── No candidates: return baseline-only result ────────────────
    if not candidates:
        result = _no_repair_result(
            H, llr, syndrome_vec, max_iters,
            instability_score_before, predicted_instability_before,
            cluster_nodes, cluster_risk_score,
            enable_decode_comparison,
        )
        result.update(multistep_fields)
        result["repair_sequence"] = []
        result["sequence_length"] = 0
        return result

    # ── Step 2: Score candidates (single or multi-step) ───────────
    if effective_depth <= 1:
        # Single-step: use existing v7.3.0 path.
        scored: list[dict[str, Any]] = []
        for candidate in candidates:
            metrics = score_repair_candidate(
                H, candidate,
                nb_spectral_radius=nb_spectral_radius,
                spectral_instability_ratio=spectral_instability_ratio,
                ipr_localization_score=ipr_localization_score,
                cluster_risk_scores=cluster_risk_scores_raw,
                avg_variable_degree=avg_variable_degree,
                avg_check_degree=avg_check_degree,
                instability_score_before=instability_score_before,
                predicted_instability_before=predicted_instability_before,
                instability_threshold=instability_threshold,
            )
            scored.append(metrics)

        selection = select_best_repair(scored)
        best_swap = selection["best_swap"]
        best_metrics = selection["best_candidate_metrics"]
        num_candidates_evaluated = selection["num_candidates_evaluated"]

        if best_swap is None:
            result = _no_repair_result(
                H, llr, syndrome_vec, max_iters,
                instability_score_before, predicted_instability_before,
                cluster_nodes, cluster_risk_score,
                enable_decode_comparison,
                num_candidates_evaluated=num_candidates_evaluated,
            )
            result.update(multistep_fields)
            result["repair_sequence"] = []
            result["sequence_length"] = 0
            return result

        instability_score_after = best_metrics[
            "spectral_instability_score_after"
        ]
        predicted_instability_after = best_metrics[
            "predicted_instability_after"
        ]
        score_improvement = best_metrics["score_improvement"]
        best_sequence = [best_swap]
    else:
        # Multi-step search with optional pruning.
        scored_sequences, branches_pruned = _generate_repair_sequences(
            H, candidates, instability_score_before,
            ipr_localization_score, cluster_risk_scores_raw,
            avg_variable_degree, avg_check_degree,
            instability_threshold, predicted_instability_before,
            effective_depth, enable_pruning,
            risk_result, max_candidates, candidate_clusters,
        )
        multistep_fields["branches_pruned"] = branches_pruned

        seq_selection = _select_best_repair_sequence(scored_sequences)
        best_seq = seq_selection["best_sequence"]
        best_seq_metrics = seq_selection["best_sequence_metrics"]
        num_candidates_evaluated = seq_selection[
            "num_candidates_evaluated"
        ]

        if best_seq is None:
            result = _no_repair_result(
                H, llr, syndrome_vec, max_iters,
                instability_score_before, predicted_instability_before,
                cluster_nodes, cluster_risk_score,
                enable_decode_comparison,
                num_candidates_evaluated=num_candidates_evaluated,
            )
            result.update(multistep_fields)
            result["repair_sequence"] = []
            result["sequence_length"] = 0
            return result

        instability_score_after = best_seq_metrics[
            "spectral_instability_score_after"
        ]
        predicted_instability_after = best_seq_metrics[
            "predicted_instability_after"
        ]
        score_improvement = best_seq_metrics["score_improvement"]
        best_sequence = best_seq

    # ── Step 3: Optional decode comparison ────────────────────────
    if enable_decode_comparison:
        # Baseline decode.
        baseline_correction, baseline_iters, _ = (
            _experimental_bp_flooding(H, llr, syndrome_vec, max_iters)
        )
        baseline_success = bool(np.array_equal(
            _compute_syndrome(H, baseline_correction), syndrome_vec,
        ))

        # Apply full repair sequence to build H_repaired.
        edges = _extract_edges(H)
        for swap_step in best_sequence:
            edges = _apply_swap(edges, swap_step)
        H_repaired = _edges_to_H(edges, m, n)

        repaired_correction, repaired_iters, _ = (
            _experimental_bp_flooding(
                H_repaired, llr, syndrome_vec, max_iters,
            )
        )
        repaired_success = bool(np.array_equal(
            _compute_syndrome(H_repaired, repaired_correction),
            syndrome_vec,
        ))

        decode_success_before = baseline_success
        decode_success_after = repaired_success
        iterations_before = baseline_iters
        iterations_after = repaired_iters
        delta_success = int(repaired_success) - int(baseline_success)
        delta_iterations = repaired_iters - baseline_iters
    else:
        decode_success_before = False
        decode_success_after = False
        iterations_before = 0
        iterations_after = 0
        delta_success = 0
        delta_iterations = 0

    # Build best_swap for backward compatibility (first swap in sequence).
    first_swap = best_sequence[0]

    result = {
        "repair_applied": True,
        "best_swap": {
            "remove": [list(e) for e in first_swap["remove"]],
            "add": first_swap["add"],
            "description": first_swap["description"],
        },
        "num_candidates_evaluated": num_candidates_evaluated,
        "spectral_instability_score_before": round(
            instability_score_before, 12,
        ),
        "spectral_instability_score_after": round(
            instability_score_after, 12,
        ),
        "score_improvement": round(score_improvement, 12),
        "predicted_instability_before": predicted_instability_before,
        "predicted_instability_after": predicted_instability_after,
        "decode_success_before": decode_success_before,
        "decode_success_after": decode_success_after,
        "delta_success": delta_success,
        "iterations_before": iterations_before,
        "iterations_after": iterations_after,
        "delta_iterations": delta_iterations,
        "cluster_nodes": cluster_nodes,
        "cluster_risk_score": round(cluster_risk_score, 12),
        "repair_sequence": best_sequence,
        "sequence_length": len(best_sequence),
    }
    result.update(multistep_fields)
    return result


def _no_repair_result(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    max_iters: int,
    instability_score_before: float,
    predicted_instability_before: bool,
    cluster_nodes: list[int],
    cluster_risk_score: float,
    enable_decode_comparison: bool,
    *,
    num_candidates_evaluated: int = 0,
) -> dict[str, Any]:
    """Build result when no valid repair improves the score."""
    if enable_decode_comparison:
        correction, iters, _ = _experimental_bp_flooding(
            H, llr, syndrome_vec, max_iters,
        )
        success = bool(np.array_equal(
            _compute_syndrome(H, correction), syndrome_vec,
        ))
        decode_success_before = success
        decode_success_after = success
        iterations_before = iters
        iterations_after = iters
    else:
        decode_success_before = False
        decode_success_after = False
        iterations_before = 0
        iterations_after = 0

    return {
        "repair_applied": False,
        "best_swap": None,
        "num_candidates_evaluated": num_candidates_evaluated,
        "spectral_instability_score_before": round(instability_score_before, 12),
        "spectral_instability_score_after": round(instability_score_before, 12),
        "score_improvement": 0.0,
        "predicted_instability_before": predicted_instability_before,
        "predicted_instability_after": predicted_instability_before,
        "decode_success_before": decode_success_before,
        "decode_success_after": decode_success_after,
        "delta_success": 0,
        "iterations_before": iterations_before,
        "iterations_after": iterations_after,
        "delta_iterations": 0,
        "cluster_nodes": cluster_nodes,
        "cluster_risk_score": round(cluster_risk_score, 12),
    }


# ── Phase Diagram Aggregation Metrics ────────────────────────────────


def compute_repair_loop_aggregate_metrics(
    trial_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate spectral graph repair loop results across trials.

    Parameters
    ----------
    trial_results : list[dict[str, Any]]
        List of per-trial results from
        :func:`run_spectral_graph_repair_loop`.

    Returns
    -------
    dict[str, Any]
        JSON-serializable aggregate metrics dictionary.
    """
    n = len(trial_results)
    if n == 0:
        return {
            "mean_repair_score_improvement": 0.0,
            "repair_activation_rate": 0.0,
            "repair_prediction_flip_rate": 0.0,
            "mean_repaired_instability_score": 0.0,
            "mean_delta_iterations_after_repair": 0.0,
            "mean_delta_success_after_repair": 0.0,
            "mean_candidates_evaluated": 0.0,
            "repair_decode_improvement_rate": 0.0,
            "num_trials": 0,
        }

    total_improvement = 0.0
    num_activated = 0
    num_prediction_flips = 0
    total_repaired_score = 0.0
    total_delta_iters = 0
    total_delta_success = 0
    total_candidates = 0
    num_decode_improvements = 0

    for trial in trial_results:
        total_improvement += trial.get("score_improvement", 0.0)
        if trial.get("repair_applied", False):
            num_activated += 1
        pred_before = trial.get("predicted_instability_before", False)
        pred_after = trial.get("predicted_instability_after", False)
        if pred_before != pred_after:
            num_prediction_flips += 1
        total_repaired_score += trial.get(
            "spectral_instability_score_after", 0.0,
        )
        total_delta_iters += trial.get("delta_iterations", 0)
        total_delta_success += trial.get("delta_success", 0)
        total_candidates += trial.get("num_candidates_evaluated", 0)
        if trial.get("delta_success", 0) > 0:
            num_decode_improvements += 1

    return {
        "mean_repair_score_improvement": round(total_improvement / n, 12),
        "repair_activation_rate": round(num_activated / n, 12),
        "repair_prediction_flip_rate": round(num_prediction_flips / n, 12),
        "mean_repaired_instability_score": round(total_repaired_score / n, 12),
        "mean_delta_iterations_after_repair": round(total_delta_iters / n, 12),
        "mean_delta_success_after_repair": round(total_delta_success / n, 12),
        "mean_candidates_evaluated": round(total_candidates / n, 12),
        "repair_decode_improvement_rate": round(num_decode_improvements / n, 12),
        "num_trials": n,
    }
