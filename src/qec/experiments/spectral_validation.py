"""
v7.6.1 — Spectral Diagnostic Validation Experiment.

Validates that non-backtracking eigenvector localization predicts
BP instability edges by comparing spectral edge rankings against
empirically measured LLR-variance instability rankings.

Computes Precision@k and Spearman rank correlation across
multiple deterministic trials with a random baseline for comparison.

Does not modify decoder internals.  Fully deterministic: no randomness
beyond seeded RNG for baseline, no global state, no input mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from typing import Any

import numpy as np
from scipy import stats

from qec.decoder.bp_decoder_reference import bp_decode
from qec.diagnostics.spectral_nb import (
    SPECTRAL_SCHEMA_VERSION,
    _TannerGraph,
    compute_nb_spectrum,
)
from qec.diagnostics._spectral_utils import build_directed_edges
from qec.experiments.eeec_anomaly_scan import (
    DEFAULT_EEEC_THRESHOLD,
    DEFAULT_IPR_THRESHOLD,
    DEFAULT_RADIUS_THRESHOLD,
)


_ROUND = 12


# ── Internal helpers ──────────────────────────────────────────────


def _compute_syndrome(H: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Compute binary syndrome s = H @ x (mod 2)."""
    return (H.astype(np.int32) @ x.astype(np.int32)) % 2


def _derive_seed(base_seed: int, label: str) -> int:
    """Deterministic sub-seed derivation via SHA-256."""
    data = struct.pack(">Q", base_seed) + label.encode("utf-8")
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h[:8], "big") % (2**31)


def _generate_error_vector(n: int, p: float, seed: int) -> np.ndarray:
    """Generate deterministic binary error vector."""
    rng = np.random.RandomState(seed)
    return (rng.random(n) < p).astype(np.uint8)


# ── Per-trial BP instability measurement ──────────────────────────


def _measure_edge_instability(
    H: np.ndarray,
    llr: np.ndarray,
    syndrome_vec: np.ndarray,
    max_iters: int,
    directed_edges: list[tuple[int, int]],
    n_vars: int,
) -> tuple[np.ndarray, bool, int]:
    """Run BP and measure per-edge LLR variance as instability signal.

    For each directed edge (u, v) where v is a variable node,
    the instability is the variance of LLR[v] across BP iterations.

    Returns
    -------
    edge_instability : np.ndarray
        Per-directed-edge instability score.
    bp_failed : bool
        Whether BP failed to converge.
    bp_iterations : int
        Number of BP iterations executed.
    """
    m, n = H.shape

    # Request full LLR history
    result = bp_decode(
        H,
        llr,
        max_iters=max_iters,
        syndrome_vec=syndrome_vec,
        llr_history=max_iters,
    )

    hard_decision = result[0]
    iterations = result[1]

    # Check if BP converged: syndrome must be satisfied
    residual_syndrome = _compute_syndrome(H, hard_decision)
    target_syndrome = np.asarray(syndrome_vec, dtype=np.int32)
    bp_failed = not np.array_equal(residual_syndrome, target_syndrome)

    # Extract LLR history
    if len(result) >= 3:
        llr_history = result[2]  # shape (k, n)
    else:
        # No history available, use uniform instability
        llr_history = llr.reshape(1, -1)

    # Compute per-variable-node LLR variance across iterations
    if llr_history.shape[0] > 1:
        var_per_variable = np.var(llr_history, axis=0)  # shape (n,)
    else:
        var_per_variable = np.zeros(n, dtype=np.float64)

    # Map to directed edges: for edge (u, v), use var of target node
    # If target is variable node (index < n), use its variance
    # If target is check node, use 0 (no direct LLR)
    num_edges = len(directed_edges)
    edge_instability = np.zeros(num_edges, dtype=np.float64)

    for i, (u, v) in enumerate(directed_edges):
        if v < n:
            edge_instability[i] = var_per_variable[v]
        elif u < n:
            edge_instability[i] = var_per_variable[u]

    return edge_instability, bp_failed, iterations


# ── Precision@k ───────────────────────────────────────────────────


def _precision_at_k(
    spectral_ranking: np.ndarray,
    instability_ranking: np.ndarray,
    k: int,
) -> float:
    """Compute precision@k between two edge rankings.

    Parameters
    ----------
    spectral_ranking : np.ndarray
        Edge indices sorted by spectral sensitivity (descending).
    instability_ranking : np.ndarray
        Edge indices sorted by instability (descending).
    k : int
        Number of top edges to compare.

    Returns
    -------
    float
        Fraction of top-k spectral edges that appear in top-k instability edges.
    """
    if k == 0:
        return 0.0

    top_k_spectral = set(spectral_ranking[:k].tolist())
    top_k_instability = set(instability_ranking[:k].tolist())

    overlap = len(top_k_spectral & top_k_instability)
    return overlap / k


# ── Spearman correlation ─────────────────────────────────────────


def _spearman_correlation(
    spectral_scores: np.ndarray,
    instability_scores: np.ndarray,
) -> float:
    """Compute Spearman rank correlation between two score vectors."""
    if len(spectral_scores) < 2:
        return 0.0

    # Handle constant arrays (no variance)
    if np.all(spectral_scores == spectral_scores[0]):
        return 0.0
    if np.all(instability_scores == instability_scores[0]):
        return 0.0

    corr, _ = stats.spearmanr(spectral_scores, instability_scores)

    if np.isnan(corr):
        return 0.0

    return float(corr)


# ── Random baseline ──────────────────────────────────────────────


def _random_precision_at_k(
    num_edges: int,
    k: int,
    instability_ranking: np.ndarray,
    seed: int,
) -> float:
    """Compute precision@k for a deterministic random ranking."""
    if num_edges == 0 or k == 0:
        return 0.0

    rng = np.random.RandomState(seed)
    random_order = np.arange(num_edges)
    rng.shuffle(random_order)

    return _precision_at_k(random_order, instability_ranking, k)


# ── Main experiment ──────────────────────────────────────────────


def run_spectral_validation_experiment(
    H: np.ndarray,
    *,
    trial_seeds: list[int] | None = None,
    p: float = 0.05,
    max_iters: int = 100,
) -> dict[str, Any]:
    """Run spectral diagnostic validation experiment.

    Executes multiple deterministic trials comparing spectral edge
    sensitivity rankings against empirically measured BP instability
    rankings.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).
    trial_seeds : list[int] or None
        Seed schedule for trials.  Defaults to [0,1,2,3,4,5,6,7].
    p : float
        Channel error probability for BSC syndrome channel.
    max_iters : int
        Maximum BP iterations per trial.

    Returns
    -------
    dict[str, Any]
        JSON-serializable validation artifact.
    """
    H_arr = np.asarray(H, dtype=np.float64)
    m, n = H_arr.shape

    if trial_seeds is None:
        trial_seeds = [0, 1, 2, 3, 4, 5, 6, 7]

    # Compute spectral diagnostics (once, deterministic)
    spectrum = compute_nb_spectrum(H_arr)
    spectral_radius = spectrum["spectral_radius"]
    ipr = spectrum["ipr"]
    eeec = spectrum["eeec"]
    sis = spectrum["sis"]
    edge_energy = spectrum["edge_energy"]

    # Build directed edges for alignment
    graph = _TannerGraph(H_arr)
    directed_edges = build_directed_edges(graph)
    num_edges = len(directed_edges)
    k = max(1, math.ceil(math.sqrt(num_edges))) if num_edges > 0 else 0

    # Spectral ranking: sort by edge_energy descending, index ascending
    spectral_indices = np.argsort(
        -edge_energy, kind="stable"
    )

    # Compute uniform LLR (syndrome-only channel)
    eps = 1e-30
    base_llr = np.log((1.0 - p + eps) / (p + eps))
    llr_base = np.full(n, base_llr, dtype=np.float64)

    # Per-trial results
    precision_values = []
    spearman_values = []
    random_precision_values = []
    bp_failed_any = False
    bp_iterations_last = 0

    for trial_seed in trial_seeds:
        # Generate deterministic error vector
        error_seed = _derive_seed(trial_seed, "error_vector")
        error_vector = _generate_error_vector(n, p, error_seed)

        # Compute syndrome
        syndrome = _compute_syndrome(H_arr, error_vector)

        # Measure edge instability via BP LLR trajectory
        edge_instability, bp_failed, bp_iters = _measure_edge_instability(
            H_arr, llr_base, syndrome, max_iters, directed_edges, n,
        )

        if bp_failed:
            bp_failed_any = True
        bp_iterations_last = bp_iters

        # Instability ranking: sort by instability descending, index ascending
        instability_indices = np.argsort(
            -edge_instability, kind="stable"
        )

        # Precision@k
        prec = _precision_at_k(spectral_indices, instability_indices, k)
        precision_values.append(prec)

        # Spearman correlation
        spearman = _spearman_correlation(edge_energy, edge_instability)
        spearman_values.append(spearman)

        # Random baseline
        random_seed = _derive_seed(trial_seed, "random_baseline")
        rand_prec = _random_precision_at_k(
            num_edges, k, instability_indices, random_seed,
        )
        random_precision_values.append(rand_prec)

    # Aggregate
    mean_precision = round(
        float(np.mean(precision_values)) if precision_values else 0.0,
        _ROUND,
    )
    mean_spearman = round(
        float(np.mean(spearman_values)) if spearman_values else 0.0,
        _ROUND,
    )
    mean_random_precision = round(
        float(np.mean(random_precision_values))
        if random_precision_values
        else 0.0,
        _ROUND,
    )

    # EEEC anomaly detection (inline threshold check)
    eeec_anomaly_detected = (
        spectral_radius < DEFAULT_RADIUS_THRESHOLD
        and eeec > DEFAULT_EEEC_THRESHOLD
        and ipr > DEFAULT_IPR_THRESHOLD
    )

    artifact = {
        "bp_failed": bp_failed_any,
        "bp_iterations": bp_iterations_last,
        "eeec_anomaly_detected": bool(eeec_anomaly_detected),
        "mean_precision_at_k": mean_precision,
        "mean_random_precision_at_k": mean_random_precision,
        "mean_spearman_correlation": mean_spearman,
        "nb_eeec": eeec,
        "nb_ipr": ipr,
        "nb_sis": sis,
        "nb_spectral_radius": spectral_radius,
        "num_edges": num_edges,
        "schema_version": SPECTRAL_SCHEMA_VERSION,
    }

    return artifact


def serialize_artifact(artifact: dict[str, Any]) -> str:
    """Serialize validation artifact to canonical JSON."""
    return json.dumps(artifact, sort_keys=True)
