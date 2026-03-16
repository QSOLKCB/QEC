"""
v9.2.0 — Discovery Run Experiment.

Runs the structure discovery engine and saves generation-level
artifacts to JSON.  Includes reproducibility metadata.

Layer 5 — Experiments.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

import json
import os
from typing import Any

from src.qec.discovery.discovery_engine import run_structure_discovery
from src.qec.io.export_graph import (
    export_matrix_market,
    export_parity_check,
    export_json_adjacency,
)
from src.qec.utils.reproducibility import collect_environment_metadata
from src.utils.canonicalize import canonicalize


def run_discovery_experiment(
    spec: dict[str, Any],
    *,
    num_generations: int = 10,
    population_size: int = 8,
    base_seed: int = 0,
    archive_top_k: int = 5,
    output_path: str = "artifacts/discovery_run.json",
    enable_curriculum_learning: bool = False,
    curriculum_success_threshold: float = 0.2,
    curriculum_initial_tier: int = 0,
    enable_motif_clustering: bool = False,
    enable_operator_stability_guard: bool = True,
    enable_information_gain_scheduler: bool = False,
    enable_autonomous_scheduler: bool = False,
) -> dict[str, Any]:
    """Run a discovery experiment and save the artifact.

    Parameters
    ----------
    spec : dict[str, Any]
        Generation specification with keys: num_variables, num_checks,
        variable_degree, check_degree.
    num_generations : int
        Number of discovery generations.
    population_size : int
        Population size per generation.
    base_seed : int
        Deterministic base seed.
    archive_top_k : int
        Archive elites per category.
    output_path : str
        Path for JSON artifact output.

    Returns
    -------
    dict[str, Any]
        Discovery results artifact.
    """
    result = run_structure_discovery(
        spec,
        num_generations=num_generations,
        population_size=population_size,
        base_seed=base_seed,
        archive_top_k=archive_top_k,
        enable_curriculum_learning=enable_curriculum_learning,
        curriculum_success_threshold=curriculum_success_threshold,
        curriculum_initial_tier=curriculum_initial_tier,
        enable_motif_clustering=enable_motif_clustering,
        enable_operator_stability_guard=enable_operator_stability_guard,
        enable_information_gain_scheduler=enable_information_gain_scheduler,
        enable_autonomous_scheduler=enable_autonomous_scheduler,
    )

    metadata = collect_environment_metadata(
        spec=spec,
        generation_count=num_generations,
        population_size=population_size,
    )

    # Export best graph in standard formats
    best_H = result.get("best_H")
    if best_H is not None:
        artifact_dir = os.path.dirname(output_path) or "artifacts"
        export_matrix_market(best_H, os.path.join(artifact_dir, "best_graph.mtx"))
        export_parity_check(best_H, os.path.join(artifact_dir, "best_graph.txt"))
        export_json_adjacency(best_H, os.path.join(artifact_dir, "best_graph.json"))

    artifact = {
        "metadata": metadata,
        "results": {
            "spec": {
                "num_variables": spec["num_variables"],
                "num_checks": spec["num_checks"],
                "variable_degree": spec["variable_degree"],
                "check_degree": spec["check_degree"],
            },
            "config": {
                "num_generations": num_generations,
                "population_size": population_size,
                "base_seed": base_seed,
                "archive_top_k": archive_top_k,
                "enable_curriculum_learning": bool(enable_curriculum_learning),
                "curriculum_success_threshold": float(curriculum_success_threshold),
                "curriculum_initial_tier": int(curriculum_initial_tier),
                "enable_motif_clustering": bool(enable_motif_clustering),
                "enable_operator_stability_guard": bool(enable_operator_stability_guard),
                "enable_information_gain_scheduler": bool(enable_information_gain_scheduler),
                "enable_autonomous_scheduler": bool(enable_autonomous_scheduler),
            },
            "best_candidate": result["best_candidate"],
            "elite_history": result["elite_history"],
            "archive_summary": result["archive_summary"],
            "generation_summaries": result["generation_summaries"],
        },
    }


    if enable_curriculum_learning:
        artifact["results"]["curriculum_tier"] = int(result.get("curriculum_tier", curriculum_initial_tier))
        artifact["results"]["curriculum_progress"] = float(result.get("curriculum_progress", 0.0))
        artifact["results"]["curriculum_success_rate"] = float(result.get("curriculum_success_rate", 0.0))

    if enable_motif_clustering:
        artifact["results"]["motif_cluster_count"] = int(result.get("motif_cluster_count", 0))
        artifact["results"]["cluster_frequencies"] = [int(v) for v in result.get("cluster_frequencies", [])]
        artifact["results"]["cluster_centroids"] = [
            [float(x) for x in row] for row in result.get("cluster_centroids", [])
        ]

    artifact = canonicalize(artifact)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")

    # Export archive artifact with metadata
    archive_path = output_path.replace("discovery_run", "discovery_archive")
    if archive_path != output_path:
        archive_artifact = {
            "metadata": metadata,
            "archive": result["archive_summary"],
        }
        archive_artifact = canonicalize(archive_artifact)
        with open(archive_path, "w") as f:
            json.dump(archive_artifact, f, sort_keys=True, separators=(",", ":"))
            f.write("\n")

    return artifact
