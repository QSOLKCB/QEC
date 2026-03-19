from __future__ import annotations

import json

import numpy as np

from qec.analysis.experiment_targets import (
    detect_high_uncertainty_regions,
    generate_experiment_targets,
)
from qec.analysis.phase_diagram_uncertainty import estimate_phase_uncertainty
from qec.discovery.autonomous_scheduler import schedule_autonomous_target
from qec.discovery.discovery_engine import run_structure_discovery
from qec.discovery.experiment_planner import SpectralExperimentPlanner


def _phase_surface() -> dict[str, np.ndarray]:
    return {
        "grid_x": np.asarray([0.8, 1.0, 1.2], dtype=np.float64),
        "grid_y": np.asarray([-0.3, -0.2, -0.1], dtype=np.float64),
        "grid_z": np.asarray(
            [
                [0.1, 0.2, 0.5],
                [0.2, 0.9, 0.4],
                [0.1, 0.3, 0.2],
            ],
            dtype=np.float64,
        ),
    }


def test_uncertainty_map_determinism() -> None:
    p = _phase_surface()
    u1 = estimate_phase_uncertainty(p)
    u2 = estimate_phase_uncertainty(p)
    assert np.allclose(u1["uncertainty_map"], u2["uncertainty_map"])


def test_region_detection_determinism() -> None:
    uncertainty = estimate_phase_uncertainty(_phase_surface())
    r1 = detect_high_uncertainty_regions(uncertainty, threshold=0.02)
    r2 = detect_high_uncertainty_regions(uncertainty, threshold=0.02)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_target_generation_reproducibility() -> None:
    uncertainty = estimate_phase_uncertainty(_phase_surface())
    regions = detect_high_uncertainty_regions(uncertainty, threshold=0.02)
    t1 = generate_experiment_targets(regions, max_targets=4)
    t2 = generate_experiment_targets(regions, max_targets=4)
    assert [x.tolist() for x in t1] == [x.tolist() for x in t2]


def test_scheduler_experiment_planning_determinism() -> None:
    frontier = [
        np.asarray([1.0, -0.2, 0.1, 0.0], dtype=np.float64),
        np.asarray([1.2, -0.1, 0.1, 0.0], dtype=np.float64),
        np.asarray([0.9, -0.3, 0.1, 0.0], dtype=np.float64),
    ]
    phase_uncertainty = np.asarray([0.1, 0.6, 0.6], dtype=np.float64)
    novelty = np.asarray([0.0, 0.2, 0.1], dtype=np.float64)
    s1 = schedule_autonomous_target(
        memory=None,
        frontier_spectra=frontier,
        strategy="experiment_planning",
        uncertainty_weight=0.7,
        exploration_weight=0.3,
        phase_uncertainty=phase_uncertainty,
        candidate_novelty=novelty,
    )
    s2 = schedule_autonomous_target(
        memory=None,
        frontier_spectra=frontier,
        strategy="experiment_planning",
        uncertainty_weight=0.7,
        exploration_weight=0.3,
        phase_uncertainty=phase_uncertainty,
        candidate_novelty=novelty,
    )
    assert np.allclose(s1["target_spectrum"], s2["target_spectrum"])
    assert s1["combined_score"] == s2["combined_score"]


def test_planner_and_engine_reproducibility() -> None:
    planner = SpectralExperimentPlanner(uncertainty_threshold=0.02, max_targets=3)
    p1 = planner.plan_experiments(_phase_surface(), landscape_memory=None)
    p2 = planner.plan_experiments(_phase_surface(), landscape_memory=None)
    assert p1["phase_uncertainty_score"] == p2["phase_uncertainty_score"]
    assert [t.tolist() for t in p1["planned_targets"]] == [t.tolist() for t in p2["planned_targets"]]

    spec = {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }
    r1 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_experiment_planner=True,
        planner_uncertainty_threshold=0.01,
        planner_max_targets=3,
    )
    r2 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_experiment_planner=True,
        planner_uncertainty_threshold=0.01,
        planner_max_targets=3,
    )
    g1 = r1["generation_summaries"][-1]
    g2 = r2["generation_summaries"][-1]
    assert g1["planner_iteration"] == g2["planner_iteration"]
    assert g1["phase_uncertainty_score"] == g2["phase_uncertainty_score"]
    assert g1["planned_experiment_targets"] == g2["planned_experiment_targets"]
