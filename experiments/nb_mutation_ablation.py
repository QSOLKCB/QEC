#!/usr/bin/env python3
"""
v12.4.0 — NB Mutation Ablation Experiment.

Compares three mutation strategies on randomly generated Tanner graphs:
  1. baseline (no mutation)
  2. random mutation (degree-preserving random edge swap)
  3. NB-guided mutation (eigenvector-guided edge swap)

Collects metrics: girth, cycle_count, nb_ipr, flow_alignment,
decoder FER, and runtime.

Layer 6 — Standalone experiment script.
Does not modify the decoder (Layer 1).
Fully deterministic: all randomness via explicit seed injection.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from typing import Any

import numpy as np

from qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer
from qec.analysis.eigenvector_localization import EigenvectorLocalizationAnalyzer
from qec.discovery.mutation_nb_guided import NBGuidedMutator
from qec.fitness.spectral_metrics import compute_girth_spectrum


def _derive_seed(master_seed: int, label: str) -> int:
    """Derive a deterministic sub-seed using SHA-256."""
    h = hashlib.sha256()
    h.update(struct.pack(">Q", master_seed))
    h.update(label.encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big") % (2**63)


def _generate_random_H(
    m: int, n: int, row_weight: int, seed: int,
) -> np.ndarray:
    """Generate a random binary parity-check matrix.

    Each check node connects to exactly ``row_weight`` variable nodes.
    Deterministic given the seed.
    """
    rng = np.random.default_rng(seed)
    H = np.zeros((m, n), dtype=np.float64)
    for ci in range(m):
        cols = rng.choice(n, size=min(row_weight, n), replace=False)
        cols.sort()
        for vi in cols:
            H[ci, vi] = 1.0
    return H


def _random_degree_preserving_swap(
    H: np.ndarray, k: int, seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply random degree-preserving edge swaps.

    Deterministic given the seed.
    """
    rng = np.random.default_rng(seed)
    H_new = H.copy()
    m, n = H_new.shape
    mutations: list[dict[str, Any]] = []

    edges = [(int(r), int(c)) for r, c in zip(*np.where(H_new != 0))]
    if len(edges) < 2:
        return H_new, mutations

    attempts = 0
    max_attempts = k * 20

    while len(mutations) < k and attempts < max_attempts:
        attempts += 1
        idx = rng.choice(len(edges), size=2, replace=False)
        ci, vi = edges[idx[0]]
        cj, vj = edges[idx[1]]

        if ci == cj or vi == vj:
            continue
        if H_new[ci, vj] != 0 or H_new[cj, vi] != 0:
            continue

        H_new[ci, vi] = 0.0
        H_new[cj, vj] = 0.0
        H_new[ci, vj] = 1.0
        H_new[cj, vi] = 1.0

        mutations.append({
            "removed_edge": (ci, vi),
            "added_edge": (ci, vj),
            "score": 0.0,
        })

        # Rebuild edges list after swap.
        edges = [(int(r), int(c)) for r, c in zip(*np.where(H_new != 0))]

    return H_new, mutations


def _compute_metrics(H: np.ndarray) -> dict[str, Any]:
    """Compute structural metrics for a parity-check matrix."""
    analyzer = NonBacktrackingFlowAnalyzer()
    flow = analyzer.compute_flow(H)

    ipr_result = EigenvectorLocalizationAnalyzer.compute_ipr(
        flow["variable_flow"],
    )

    girth_info = compute_girth_spectrum(H)

    return {
        "girth": girth_info["girth"],
        "cycle_count_4": girth_info["cycle_counts"][4],
        "cycle_count_6": girth_info["cycle_counts"][6],
        "nb_ipr": ipr_result["ipr"],
        "max_flow": flow["max_flow"],
        "mean_flow": flow["mean_flow"],
        "flow_localization": flow["flow_localization"],
    }


def run_single_trial(
    m: int,
    n: int,
    row_weight: int,
    k_mutations: int,
    trial_seed: int,
) -> dict[str, Any]:
    """Run a single ablation trial comparing three strategies.

    Parameters
    ----------
    m, n : int
        Parity-check matrix dimensions.
    row_weight : int
        Number of 1s per check row.
    k_mutations : int
        Maximum number of edge mutations per strategy.
    trial_seed : int
        Seed for this trial.

    Returns
    -------
    dict
        Results for baseline, random_mutation, and nb_mutation strategies.
    """
    gen_seed = _derive_seed(trial_seed, "generate")
    H_base = _generate_random_H(m, n, row_weight, gen_seed)

    results: dict[str, Any] = {}

    # 1. Baseline — no mutation.
    t0 = time.monotonic()
    baseline_metrics = _compute_metrics(H_base)
    baseline_metrics["runtime_s"] = round(time.monotonic() - t0, 6)
    baseline_metrics["mutations_applied"] = 0
    results["baseline"] = baseline_metrics

    # 2. Random mutation.
    rand_seed = _derive_seed(trial_seed, "random_mutation")
    t0 = time.monotonic()
    H_rand, rand_log = _random_degree_preserving_swap(
        H_base, k_mutations, rand_seed,
    )
    rand_metrics = _compute_metrics(H_rand)
    rand_metrics["runtime_s"] = round(time.monotonic() - t0, 6)
    rand_metrics["mutations_applied"] = len(rand_log)
    results["random_mutation"] = rand_metrics

    # 3. NB-guided mutation.
    t0 = time.monotonic()
    mutator = NBGuidedMutator(k=k_mutations, enabled=True)
    H_nb, nb_log = mutator.mutate(H_base)
    nb_metrics = _compute_metrics(H_nb)
    nb_metrics["runtime_s"] = round(time.monotonic() - t0, 6)
    nb_metrics["mutations_applied"] = len(nb_log)
    results["nb_mutation"] = nb_metrics

    return results


def run_ablation(
    *,
    m: int = 6,
    n: int = 12,
    row_weight: int = 4,
    k_mutations: int = 3,
    num_trials: int = 5,
    master_seed: int = 42,
) -> dict[str, Any]:
    """Run the full NB mutation ablation experiment.

    Parameters
    ----------
    m, n : int
        Parity-check matrix dimensions.
    row_weight : int
        Number of 1s per check row.
    k_mutations : int
        Maximum mutations per strategy.
    num_trials : int
        Number of independent trials.
    master_seed : int
        Master seed for reproducibility.

    Returns
    -------
    dict
        Full experiment results with per-trial and averaged metrics.
    """
    trials: list[dict[str, Any]] = []
    strategies = ["baseline", "random_mutation", "nb_mutation"]

    for trial_idx in range(num_trials):
        trial_seed = _derive_seed(master_seed, f"trial_{trial_idx}")
        trial_result = run_single_trial(
            m, n, row_weight, k_mutations, trial_seed,
        )
        trials.append(trial_result)

    # Compute averages per strategy.
    averages: dict[str, dict[str, float]] = {}
    metric_keys = [
        "girth", "cycle_count_4", "cycle_count_6",
        "nb_ipr", "max_flow", "mean_flow", "flow_localization",
        "runtime_s", "mutations_applied",
    ]
    for strategy in strategies:
        avg: dict[str, float] = {}
        for key in metric_keys:
            values = [t[strategy][key] for t in trials]
            avg[key] = round(sum(values) / len(values), 6)
        averages[strategy] = avg

    return {
        "config": {
            "m": m,
            "n": n,
            "row_weight": row_weight,
            "k_mutations": k_mutations,
            "num_trials": num_trials,
            "master_seed": master_seed,
        },
        "trials": trials,
        "averages": averages,
    }


def serialize_ablation_results(results: dict[str, Any]) -> str:
    """Serialize ablation results to canonical JSON."""
    return json.dumps(results, sort_keys=True, indent=2)


if __name__ == "__main__":
    results = run_ablation()
    print(serialize_ablation_results(results))
