"""
v8.4.0 — Generation Benchmark.

Runs the Tanner graph generator across multiple specifications and
records the best candidate for each.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

import json
import os
from typing import Any

from qec.generation.tanner_graph_generator import (
    _derive_seed,
    generate_tanner_graph_candidates,
)
from qec.generation.candidate_evaluation import evaluate_tanner_graph_candidate
from qec.generation.candidate_ranking import rank_tanner_graph_candidates
from src.utils.canonicalize import canonicalize


_ROUND = 12


def run_generation_benchmark(
    specs: list[dict[str, Any]],
    *,
    candidates_per_spec: int = 10,
    base_seed: int = 0,
    output_path: str = "artifacts/generation_benchmark.json",
) -> dict[str, Any]:
    """Run the generation benchmark across multiple graph specifications.

    Procedure for each specification:
    1. Generate candidate Tanner graphs.
    2. Evaluate all candidates via spectral metrics.
    3. Rank candidates and record the best.

    Parameters
    ----------
    specs : list[dict[str, Any]]
        List of generation specifications.
    candidates_per_spec : int
        Number of candidates per specification.
    base_seed : int
        Base seed for deterministic sub-seed derivation.
    output_path : str
        Path for JSON artifact output.

    Returns
    -------
    dict[str, Any]
        Benchmark results with best candidate per specification.
    """
    results: list[dict[str, Any]] = []

    for spec_idx, spec in enumerate(specs):
        spec_seed = _derive_seed(base_seed, f"spec_{spec_idx}")

        candidates = generate_tanner_graph_candidates(
            spec, candidates_per_spec, base_seed=spec_seed,
        )

        evaluated = []
        for c in candidates:
            evaluation = evaluate_tanner_graph_candidate(c["H"])
            evaluated.append({
                **evaluation,
                "candidate_id": c["candidate_id"],
            })

        ranked = rank_tanner_graph_candidates(evaluated)
        best = ranked[0]

        results.append({
            "spec_index": spec_idx,
            "num_variables": spec["num_variables"],
            "num_checks": spec["num_checks"],
            "variable_degree": spec["variable_degree"],
            "check_degree": spec["check_degree"],
            "best_candidate_id": best["candidate_id"],
            "best_instability_score": best["instability_score"],
            "best_spectral_radius": best["spectral_radius"],
            "best_entropy": best["entropy"],
            "best_bethe_margin": best["bethe_margin"],
            "best_predicted_regime": best["predicted_regime"],
        })

    artifact = {
        "num_specs": len(specs),
        "candidates_per_spec": candidates_per_spec,
        "base_seed": base_seed,
        "results": results,
    }

    artifact = canonicalize(artifact)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")

    return artifact
