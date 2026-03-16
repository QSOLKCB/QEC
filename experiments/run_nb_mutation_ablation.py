#!/usr/bin/env python3
"""
v12.5.0 — NB Mutation Ablation with Phase Diagram Analysis.

Compares mutation strategies on randomly generated Tanner graphs:
  1. baseline (no mutation)
  2. random swap (degree-preserving random edge swap)
  3. NB swap (eigenvector-guided edge swap)
  4. NB x IPR swap (IPR-weighted eigenvector-guided edge swap)
  5. NB gradient (instability-gradient guided edge swap)

For each graph computes: spectral_radius, FER, girth, cycle_count.
Produces FER improvement vs spectral_radius across strategies.

Layer 6 — Standalone experiment script.
Does not modify the decoder (Layer 1).
Fully deterministic: all randomness via explicit seed injection.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from typing import Any

import numpy as np

sys.path.insert(0, ".")

from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer
from src.qec.analysis.eigenvector_localization import EigenvectorLocalizationAnalyzer
from src.qec.discovery.mutation_nb_guided import NBGuidedMutator
from src.qec.discovery.mutation_nb_gradient import NBGradientMutator
from src.qec.fitness.spectral_metrics import compute_girth_spectrum
from src.qec.experiments.spectral_phase_diagram import (
    _derive_seed,
    _run_decoder_trial,
)
from src.qec.experiments.constants import ABLATION_METRICS, MUTATION_STRATEGIES


_ROUND = 12

METRIC_ALIASES = {
    "fer": "FER",
    "spectral_radius": "spectral_radius",
    "ipr": "nb_ipr",
    "runtime": "runtime",
}

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

        edges = [(int(r), int(c)) for r, c in zip(*np.where(H_new != 0))]

    return H_new, mutations


def _compute_spectral_metrics(H: np.ndarray) -> dict[str, Any]:
    """Compute spectral and structural metrics for a parity-check matrix."""
    analyzer = NonBacktrackingFlowAnalyzer()
    flow = analyzer.compute_flow(H)

    ipr_result = EigenvectorLocalizationAnalyzer.compute_ipr(
        flow["variable_flow"],
    )
    girth_info = compute_girth_spectrum(H)

    return {
        "spectral_radius": round(float(flow["max_flow"]), _ROUND),
        "girth": girth_info["girth"],
        "cycle_count_4": girth_info["cycle_counts"].get(4, 0),
        "cycle_count_6": girth_info["cycle_counts"].get(6, 0),
        "nb_ipr": ipr_result["ipr"],
        "max_flow": round(float(flow["max_flow"]), _ROUND),
        "mean_flow": round(float(flow["mean_flow"]), _ROUND),
        "flow_localization": round(float(flow["flow_localization"]), _ROUND),
    }


def _compute_fer(
    H: np.ndarray,
    error_rate: float,
    num_trials: int,
    base_seed: int,
) -> float:
    """Compute FER for a parity-check matrix at a given error rate."""
    failures = 0
    for trial_idx in range(num_trials):
        trial_seed = _derive_seed(base_seed, f"fer_trial_{trial_idx}")
        result = _run_decoder_trial(H, error_rate, trial_seed)
        if not result["success"]:
            failures += 1
    return round(failures / num_trials, _ROUND) if num_trials > 0 else 0.0


def run_single_trial(
    m: int,
    n: int,
    row_weight: int,
    k_mutations: int,
    error_rate: float,
    fer_trials: int,
    trial_seed: int,
) -> dict[str, Any]:
    """Run a single ablation trial comparing mutation strategies.

    Parameters
    ----------
    m, n : int
        Parity-check matrix dimensions.
    row_weight : int
        Number of 1s per check row.
    k_mutations : int
        Maximum edge mutations per strategy.
    error_rate : float
        BSC error rate for FER measurement.
    fer_trials : int
        Number of decode trials for FER estimation.
    trial_seed : int
        Deterministic seed for this trial.

    Returns
    -------
    dict
        Results for baseline, random_swap, nb_swap, nb_ipr_swap.
    """
    gen_seed = _derive_seed(trial_seed, "generate")
    H_base = _generate_random_H(m, n, row_weight, gen_seed)

    results: dict[str, Any] = {}

    # 1. Baseline — no mutation.
    baseline_metrics = _compute_spectral_metrics(H_base)
    fer_seed = _derive_seed(trial_seed, "baseline_fer")
    baseline_metrics["FER"] = _compute_fer(
        H_base, error_rate, fer_trials, fer_seed,
    )
    baseline_metrics["mutations_applied"] = 0
    baseline_metrics["runtime"] = 0.0
    results["baseline"] = baseline_metrics

    # 2. Random swap.
    rand_seed = _derive_seed(trial_seed, "random_swap")
    H_rand, rand_log = _random_degree_preserving_swap(
        H_base, k_mutations, rand_seed,
    )
    rand_metrics = _compute_spectral_metrics(H_rand)
    fer_seed = _derive_seed(trial_seed, "random_fer")
    rand_metrics["FER"] = _compute_fer(
        H_rand, error_rate, fer_trials, fer_seed,
    )
    rand_metrics["mutations_applied"] = len(rand_log)
    rand_metrics["runtime"] = 0.0
    results["random_swap"] = rand_metrics

    # 3. NB swap.
    mutator_nb = NBGuidedMutator(k=k_mutations, enabled=True, use_ipr_weight=False)
    H_nb, nb_log = mutator_nb.mutate(H_base)
    nb_metrics = _compute_spectral_metrics(H_nb)
    fer_seed = _derive_seed(trial_seed, "nb_fer")
    nb_metrics["FER"] = _compute_fer(
        H_nb, error_rate, fer_trials, fer_seed,
    )
    nb_metrics["mutations_applied"] = len(nb_log)
    nb_metrics["runtime"] = 0.0
    results["nb_swap"] = nb_metrics

    # 4. NB x IPR swap.
    mutator_ipr = NBGuidedMutator(k=k_mutations, enabled=True, use_ipr_weight=True)
    H_ipr, ipr_log = mutator_ipr.mutate(H_base)
    ipr_metrics = _compute_spectral_metrics(H_ipr)
    fer_seed = _derive_seed(trial_seed, "nb_ipr_fer")
    ipr_metrics["FER"] = _compute_fer(
        H_ipr, error_rate, fer_trials, fer_seed,
    )
    ipr_metrics["mutations_applied"] = len(ipr_log)
    ipr_metrics["runtime"] = 0.0
    results["nb_ipr_swap"] = ipr_metrics


    # 5. NB gradient swap.
    mutator_grad = NBGradientMutator(enabled=True, avoid_4cycles=True)
    H_grad, grad_log = mutator_grad.mutate(H_base, steps=k_mutations)
    grad_runtime = 0.0
    grad_metrics = _compute_spectral_metrics(H_grad)
    fer_seed = _derive_seed(trial_seed, "nb_gradient_fer")
    grad_metrics["FER"] = _compute_fer(
        H_grad, error_rate, fer_trials, fer_seed,
    )
    grad_metrics["mutations_applied"] = len(grad_log)
    grad_metrics["runtime"] = grad_runtime
    results["nb_gradient"] = grad_metrics

    return results


def run_ablation(
    *,
    m: int = 6,
    n: int = 12,
    row_weight: int = 4,
    k_mutations: int = 5,
    num_graphs: int = 200,
    error_rate: float = 0.05,
    fer_trials: int = 50,
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
    num_graphs : int
        Number of independent Tanner graphs (default 200).
    error_rate : float
        BSC error rate (default 0.05).
    fer_trials : int
        Decode trials per FER measurement (default 50).
    master_seed : int
        Master seed for reproducibility.

    Returns
    -------
    dict
        Full experiment results with per-trial and averaged metrics.
    """
    _missing_aliases = [m for m in ABLATION_METRICS if m not in METRIC_ALIASES]
    _extra_aliases = [m for m in METRIC_ALIASES if m not in ABLATION_METRICS]
    if _missing_aliases or _extra_aliases:
        raise ValueError(
            "METRIC_ALIASES must match ABLATION_METRICS exactly; "
            f"missing={_missing_aliases}, extra={_extra_aliases}",
        )

    strategies = MUTATION_STRATEGIES
    trials: list[dict[str, Any]] = []

    for trial_idx in range(num_graphs):
        trial_seed = _derive_seed(master_seed, f"trial_{trial_idx}")
        trial_result = run_single_trial(
            m, n, row_weight, k_mutations,
            error_rate, fer_trials, trial_seed,
        )
        trials.append(trial_result)

    # Compute averages per strategy.
    metric_keys = [METRIC_ALIASES[name] for name in ABLATION_METRICS] + [
        "girth", "cycle_count_4", "cycle_count_6",
        "max_flow", "mean_flow", "flow_localization",
        "mutations_applied",
    ]

    averages: dict[str, dict[str, float]] = {}
    for strategy in strategies:
        avg: dict[str, float] = {}
        for key in metric_keys:
            values = [t[strategy][key] for t in trials]
            avg[key] = round(sum(values) / len(values), _ROUND)
        averages[strategy] = avg

    return {
        "config": {
            "m": m,
            "n": n,
            "row_weight": row_weight,
            "k_mutations": k_mutations,
            "num_graphs": num_graphs,
            "error_rate": error_rate,
            "fer_trials": fer_trials,
            "master_seed": master_seed,
        },
        "baseline": averages["baseline"],
        "random_swap": averages["random_swap"],
        "nb_swap": averages["nb_swap"],
        "nb_ipr_swap": averages["nb_ipr_swap"],
        "nb_gradient": averages["nb_gradient"],
        "trials": trials,
    }


def serialize_ablation_results(results: dict[str, Any]) -> str:
    """Serialize ablation results to canonical JSON."""
    return json.dumps(results, sort_keys=True, indent=2)


if __name__ == "__main__":
    # Small default run for quick testing.
    results = run_ablation(
        num_graphs=5,
        fer_trials=10,
    )
    print(serialize_ablation_results(results))
