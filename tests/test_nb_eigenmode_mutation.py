from __future__ import annotations

import numpy as np

from qec.analysis.non_backtracking_matrix import build_non_backtracking_matrix
from qec.analysis.non_backtracking_spectrum import leading_nb_eigenmode
from qec.discovery.discovery_engine import run_structure_discovery
from qec.discovery.nb_eigenmode_mutation import score_edges_by_eigenmode


def _default_spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_nb_matrix_shape() -> None:
    adj = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    B, edges = build_non_backtracking_matrix(adj)

    assert B.dtype == np.float64
    assert B.shape[0] == len(edges)
    assert B.shape[1] == len(edges)


def test_leading_nb_eigenmode_extraction() -> None:
    adj = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    B, _ = build_non_backtracking_matrix(adj)
    eigval, eigvec = leading_nb_eigenmode(B)

    assert np.isfinite(float(np.abs(eigval)))
    assert eigvec.shape == (B.shape[0],)


def test_score_edges_by_eigenmode_is_deterministic() -> None:
    edges = [(0, 1), (1, 0), (0, 2)]
    eigvec = np.array([1.0 + 0.0j, -2.0 + 0.0j, 0.5 + 0.0j], dtype=np.complex128)

    s1 = score_edges_by_eigenmode(edges, eigvec)
    s2 = score_edges_by_eigenmode(edges, eigvec)

    assert s1 == s2
    assert s1[(1, 0)] == 2.0


def test_nb_eigenmode_mutation_ranking_stability() -> None:
    spec = _default_spec()
    r1 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_nb_eigenmode_mutation=True,
    )
    r2 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_nb_eigenmode_mutation=True,
    )

    assert r1["elite_history"] == r2["elite_history"]


def test_trajectory_records_nb_eigenvalue_column() -> None:
    spec = _default_spec()
    result = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=11,
        enable_spectral_trajectory=True,
        enable_nb_eigenmode_mutation=True,
    )
    trajectory = np.asarray(result["spectral_trajectory"], dtype=np.float64)
    assert trajectory.shape[1] == 5
    assert np.all(np.isfinite(trajectory[:, -1]))
