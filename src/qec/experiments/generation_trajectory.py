"""
v8.4.0 — Generation Trajectory Experiment.

Runs an iterative Tanner graph generation trajectory:
generate → evaluate → rank → select best → perturb → repeat.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from qec.generation.tanner_graph_generator import (
    _derive_seed,
    generate_tanner_graph_candidates,
)
from qec.generation.candidate_evaluation import evaluate_tanner_graph_candidate
from qec.generation.candidate_ranking import rank_tanner_graph_candidates
from utils.canonicalize import canonicalize


def run_generation_trajectory(
    spec: dict[str, Any],
    num_steps: int,
    *,
    candidates_per_step: int = 5,
    base_seed: int = 0,
    output_path: str = "artifacts/generation_trajectory.json",
) -> list[dict[str, Any]]:
    """Run an iterative Tanner graph generation trajectory.

    Procedure at each step:
    1. Generate candidates (fresh or by perturbing the best so far).
    2. Evaluate each candidate via spectral metrics.
    3. Rank candidates deterministically.
    4. Select the best candidate.
    5. Use the best candidate as the seed graph for the next step.

    Parameters
    ----------
    spec : dict[str, Any]
        Generation specification (see ``generate_tanner_graph_candidates``).
    num_steps : int
        Number of trajectory steps.
    candidates_per_step : int
        Number of candidates generated per step.
    base_seed : int
        Base seed for deterministic sub-seed derivation.
    output_path : str
        Path for JSON artifact output.

    Returns
    -------
    list[dict[str, Any]]
        Trajectory records, one per step.
    """
    trajectory: list[dict[str, Any]] = []

    # Step 0: initial generation
    step_seed = _derive_seed(base_seed, "step_0")
    candidates = generate_tanner_graph_candidates(
        spec, candidates_per_step, base_seed=step_seed,
    )

    # Evaluate and rank
    evaluated = []
    for c in candidates:
        evaluation = evaluate_tanner_graph_candidate(c["H"])
        evaluated.append({**evaluation, "candidate_id": c["candidate_id"], "H": c["H"]})

    ranked = rank_tanner_graph_candidates(evaluated)
    best = ranked[0]

    trajectory.append({
        "step": 0,
        "best_candidate_id": best["candidate_id"],
        "best_score": best["instability_score"],
        "best_metrics": {
            k: best[k]
            for k in (
                "spectral_radius", "entropy", "spectral_gap",
                "bethe_margin", "sis", "instability_score",
                "predicted_regime",
            )
        },
    })

    best_H = best["H"]

    # Subsequent steps: perturb best candidate
    for step_idx in range(1, num_steps + 1):
        step_seed = _derive_seed(base_seed, f"step_{step_idx}")

        perturb_spec = {
            "num_variables": spec["num_variables"],
            "num_checks": spec["num_checks"],
            "variable_degree": spec["variable_degree"],
            "check_degree": spec["check_degree"],
            "seed_graph": best_H,
        }

        candidates = generate_tanner_graph_candidates(
            perturb_spec, candidates_per_step, base_seed=step_seed,
        )

        # Also include the current best as a candidate (elitism)
        candidates.append({
            "candidate_id": "elite_best",
            "H": best_H,
        })

        evaluated = []
        for c in candidates:
            evaluation = evaluate_tanner_graph_candidate(c["H"])
            evaluated.append({
                **evaluation,
                "candidate_id": c["candidate_id"],
                "H": c["H"],
            })

        ranked = rank_tanner_graph_candidates(evaluated)
        best = ranked[0]

        trajectory.append({
            "step": step_idx,
            "best_candidate_id": best["candidate_id"],
            "best_score": best["instability_score"],
            "best_metrics": {
                k: best[k]
                for k in (
                    "spectral_radius", "entropy", "spectral_gap",
                    "bethe_margin", "sis", "instability_score",
                    "predicted_regime",
                )
            },
        })

        best_H = best["H"]

    # Save artifact
    artifact = canonicalize(trajectory)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")

    return trajectory
