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

    results_payload = {
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
        },
        "best_candidate": result["best_candidate"],
        "elite_history": result["elite_history"],
        "archive_summary": result["archive_summary"],
        "generation_summaries": result["generation_summaries"],
    }
    if "motif_library_size" in result:
        results_payload["motif_library_size"] = result["motif_library_size"]
        results_payload["motifs_used"] = result.get("motifs_used", [])
    if "operator_success_rates" in result:
        results_payload["operator_success_rates"] = result["operator_success_rates"]
        results_payload["adaptive_operator_weights"] = result.get("adaptive_operator_weights", {})

    artifact = {
        "metadata": metadata,
        "results": results_payload,
    }

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
