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
    enable_bayesian_model: bool = False,
    bayesian_length_scale: float = 1.0,
    bayesian_noise: float = 1e-6,
    output_path: str = "artifacts/discovery_run.json",
    enable_information_gain_scheduler: bool = False,
    information_gain_novelty_weight: float = 0.5,
    information_gain_uncertainty_weight: float = 0.5,
    enable_autonomous_scheduler: bool = False,
    scheduler_gap_radius: float = 0.3,
    scheduler_max_gaps: int = 16,
    scheduler_queue=None,
    enable_experiment_planner: bool = False,
    planner_uncertainty_threshold: float = 0.2,
    planner_max_targets: int = 10,
    enable_phase_diagram_3d: bool = False,
    phase_diagram_3d_path: str | None = None,
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
        enable_bayesian_model=enable_bayesian_model,
        bayesian_length_scale=bayesian_length_scale,
        bayesian_noise=bayesian_noise,
        enable_information_gain_scheduler=enable_information_gain_scheduler,
        information_gain_novelty_weight=information_gain_novelty_weight,
        information_gain_uncertainty_weight=information_gain_uncertainty_weight,
        enable_autonomous_scheduler=enable_autonomous_scheduler,
        scheduler_gap_radius=scheduler_gap_radius,
        scheduler_max_gaps=scheduler_max_gaps,
        scheduler_queue=scheduler_queue,
        enable_experiment_planner=enable_experiment_planner,
        planner_uncertainty_threshold=planner_uncertainty_threshold,
        planner_max_targets=planner_max_targets,
        enable_phase_diagram_3d=enable_phase_diagram_3d,
        phase_diagram_3d_path=phase_diagram_3d_path,
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

    artifact_results = {
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
            "enable_autonomous_scheduler": enable_autonomous_scheduler,
            "scheduler_gap_radius": scheduler_gap_radius,
            "scheduler_max_gaps": scheduler_max_gaps,
            "enable_experiment_planner": enable_experiment_planner,
            "planner_uncertainty_threshold": planner_uncertainty_threshold,
            "planner_max_targets": planner_max_targets,
            "enable_phase_diagram_3d": enable_phase_diagram_3d,
            "phase_diagram_3d_path": phase_diagram_3d_path,
        },
        "best_candidate": result["best_candidate"],
        "elite_history": result["elite_history"],
        "archive_summary": result["archive_summary"],
        "generation_summaries": result["generation_summaries"],
    }
    if enable_autonomous_scheduler:
        artifact_results["scheduled_target_spectrum"] = result.get("scheduled_target_spectrum")
        artifact_results["landscape_gap_count"] = int(result.get("landscape_gap_count", 0))
        artifact_results["scheduler_strategy"] = result.get(
            "scheduler_strategy", "landscape_exploration",
        )
    if enable_experiment_planner:
        artifact_results["experiment_plan"] = {
            "planner_iteration": int(result.get("planner_iteration", 0)),
            "phase_uncertainty_score": float(result.get("phase_uncertainty_score", 0.0)),
        }
        artifact_results["uncertainty_map"] = result.get("uncertainty_map")
        artifact_results["planned_targets"] = result.get("planned_experiment_targets", [])
    if enable_phase_diagram_3d:
        artifact_results["phase_diagram_3d_path"] = result.get("phase_diagram_3d_path")
        artifact_results["phase_diagram_3d_num_targets"] = int(
            result.get("phase_diagram_3d_num_targets", 0)
        )

    artifact = {
        "metadata": metadata,
        "results": artifact_results,
    }

    for key in (
        "agent_assignments",
        "agent_spacing",
        "cooperative_coverage",
        "frontier_exploration_rate",
        "agent_messages",
        "coordination_state",
        "agent_region_overlap",
    ):
        if key in result:
            artifact["results"][key] = result[key]

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
