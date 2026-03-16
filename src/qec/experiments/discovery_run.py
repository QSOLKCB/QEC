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
    enable_self_reflection: bool = False,
    reflection_interval: int = 50,
    hypothesis_weight: float = 0.5,
    reuse_landscape_kd_tree: bool = False,
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
    enable_theory_extraction: bool = False,
    theory_extraction_interval: int = 200,
    max_conjectures: int = 10,
    enable_theory_refinement: bool = False,
    theory_refinement_interval: int = 200,
    enable_phase_trajectory: bool = False,
    enable_spectral_geometry: bool = False,
    enable_basin_hopping: bool = False,
    enable_spectral_ridge_detection: bool = False,
    enable_phase_map_reconstruction: bool = False,
    enable_phase_guided_discovery: bool = False,
    enable_phase_novelty_discovery: bool = False,
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
        enable_self_reflection=enable_self_reflection,
        reflection_interval=reflection_interval,
        hypothesis_weight=hypothesis_weight,
        reuse_landscape_kd_tree=reuse_landscape_kd_tree,
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
        enable_theory_extraction=enable_theory_extraction,
        theory_extraction_interval=theory_extraction_interval,
        max_conjectures=max_conjectures,
        enable_theory_refinement=enable_theory_refinement,
        theory_refinement_interval=theory_refinement_interval,
        enable_phase_trajectory=enable_phase_trajectory,
        enable_spectral_geometry=enable_spectral_geometry,
        enable_basin_hopping=enable_basin_hopping,
        enable_spectral_ridge_detection=enable_spectral_ridge_detection,
        enable_phase_map_reconstruction=enable_phase_map_reconstruction,
        enable_phase_guided_discovery=enable_phase_guided_discovery,
        enable_phase_novelty_discovery=enable_phase_novelty_discovery,
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
            "enable_autonomous_scheduler": enable_autonomous_scheduler,
            "scheduler_gap_radius": scheduler_gap_radius,
            "scheduler_max_gaps": scheduler_max_gaps,
            "enable_experiment_planner": enable_experiment_planner,
            "planner_uncertainty_threshold": planner_uncertainty_threshold,
            "planner_max_targets": planner_max_targets,
            "enable_phase_diagram_3d": enable_phase_diagram_3d,
            "phase_diagram_3d_path": phase_diagram_3d_path,
            "enable_theory_extraction": enable_theory_extraction,
            "theory_extraction_interval": theory_extraction_interval,
            "max_conjectures": max_conjectures,
            "enable_theory_refinement": enable_theory_refinement,
            "theory_refinement_interval": theory_refinement_interval,
            "enable_phase_trajectory": enable_phase_trajectory,
            "enable_spectral_geometry": enable_spectral_geometry,
            "enable_basin_hopping": enable_basin_hopping,
            "enable_spectral_ridge_detection": enable_spectral_ridge_detection,
            "enable_phase_map_reconstruction": enable_phase_map_reconstruction,
            "enable_phase_guided_discovery": enable_phase_guided_discovery,
            "enable_phase_novelty_discovery": enable_phase_novelty_discovery,
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
    if enable_autonomous_scheduler:
        results_payload["scheduled_target_spectrum"] = result.get("scheduled_target_spectrum")
        results_payload["landscape_gap_count"] = int(result.get("landscape_gap_count", 0))
        results_payload["scheduler_strategy"] = result.get(
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
                "enable_self_reflection": enable_self_reflection,
                "reflection_interval": reflection_interval,
                "hypothesis_weight": hypothesis_weight,
                "reuse_landscape_kd_tree": reuse_landscape_kd_tree,
                "enable_phase_novelty_discovery": enable_phase_novelty_discovery,
            },
            "best_candidate": result["best_candidate"],
            "elite_history": result["elite_history"],
            "archive_summary": result["archive_summary"],
            "generation_summaries": result["generation_summaries"],
        },
    }

    if enable_phase_guided_discovery:
        artifact["results"]["phase_visit_counts"] = result.get("phase_visit_counts", {})
        artifact["results"]["phase_guidance_targets"] = result.get("phase_guidance_targets", [])
    if enable_phase_novelty_discovery:
        artifact["results"]["novel_phase_candidates"] = result.get("novel_phase_candidates", [])
        artifact["results"]["novelty_scores"] = result.get("novelty_scores", [])

    if enable_self_reflection:
        artifact["results"]["hypothesis_list"] = result.get("hypothesis_list", [])
        artifact["results"]["hypothesis_rankings"] = result.get("hypothesis_rankings", [])
        artifact["results"]["reflection_metrics"] = result.get("reflection_metrics", [])
        artifact["phase_diagram"] = result.get("phase_diagram", {"regions": [], "phase_boundaries": []})

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
