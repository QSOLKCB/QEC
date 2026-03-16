from __future__ import annotations

import numpy as np

from src.qec.analysis.basin_map_export import export_basin_map
from src.qec.analysis.basin_statistics import basin_sizes
from src.qec.analysis.basin_transitions import detect_basin_transitions
from src.qec.analysis.spectral_basins import identify_spectral_basins
from src.qec.analysis.spectral_phase_space import (
    project_basin_centers_2d,
    project_basin_centers_3d,
)
from src.qec.discovery.discovery_engine import run_structure_discovery


def _default_spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_basin_detection() -> None:
    traj = np.asarray(
        [
            [1.0, 0.1],
            [1.01, 0.11],
            [2.0, 0.5],
        ],
        dtype=np.float64,
    )
    assignments, centers = identify_spectral_basins(traj, threshold=0.2)

    assert centers.shape == (2, 2)
    assert assignments.tolist() == [0, 0, 1]




def test_basin_clustering_squared_distance() -> None:
    traj = [
        [1.0, 0.2],
        [1.01, 0.21],
        [2.0, 0.5],
    ]

    assignments, centers = identify_spectral_basins(
        traj,
        threshold=0.2,
    )

    assert len(centers) == 2
    assert assignments.tolist() == [0, 0, 1]


def test_basin_clustering_deterministic() -> None:
    traj = np.random.RandomState(0).rand(20, 3)

    a1, c1 = identify_spectral_basins(traj, 0.3)
    a2, c2 = identify_spectral_basins(traj, 0.3)

    assert a1.tolist() == a2.tolist()
    np.testing.assert_allclose(c1, c2)

def test_transition_detection_and_sizes() -> None:
    assignments = np.asarray([0, 0, 1, 1, 2, 2, 1], dtype=np.int64)
    assert detect_basin_transitions(assignments) == [2, 4, 6]
    assert basin_sizes(assignments) == {0: 2, 1: 3, 2: 2}


def test_basin_assignment_stability_across_runs() -> None:
    traj = np.asarray(
        [
            [0.9, 0.1, 0.2],
            [0.91, 0.09, 0.21],
            [1.8, 0.5, 0.7],
            [1.81, 0.51, 0.71],
        ],
        dtype=np.float64,
    )

    a1, c1 = identify_spectral_basins(traj, threshold=0.15)
    a2, c2 = identify_spectral_basins(traj, threshold=0.15)

    assert a1.tolist() == a2.tolist()
    np.testing.assert_allclose(c1, c2)


def test_basin_export_and_projection_helpers() -> None:
    basins = np.asarray(
        [
            [1.0, 0.2, 0.3, 0.4],
            [2.0, 0.4, 0.1, 0.9],
        ],
        dtype=np.float64,
    )
    assignments = np.asarray([0, 0, 1], dtype=np.int64)
    transitions = [2]

    payload = export_basin_map(
        basins,
        assignments,
        transitions,
        include_phase_space_projections=True,
    )

    assert payload["num_basins"] == 2
    assert payload["assignments"] == [0, 0, 1]
    assert payload["transitions"] == [2]
    assert len(payload["basin_centers_projected_2d"]) == 2
    assert len(payload["basin_centers_projected_3d"]) == 2

    p2 = project_basin_centers_2d(basins)
    p3 = project_basin_centers_3d(basins)
    assert p2.shape == (2, 2)
    assert p3.shape == (2, 3)




def test_large_trajectory_scaling() -> None:
    rng = np.random.RandomState(0)
    traj = rng.rand(2000, 3)

    assignments, centers = identify_spectral_basins(
        traj,
        threshold=0.25,
    )

    assert len(assignments) == 2000
    assert centers.shape[1] == 3

def test_discovery_basin_topology_mapping_determinism() -> None:
    spec = _default_spec()

    result_a = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=19,
        enable_basin_topology_mapping=True,
    )
    result_b = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=19,
        enable_basin_topology_mapping=True,
    )

    assert "spectral_basin_topology" in result_a
    assert "spectral_basin_sizes" in result_a
    assert result_a["spectral_basin_topology"] == result_b["spectral_basin_topology"]
    assert result_a["spectral_basin_sizes"] == result_b["spectral_basin_sizes"]
