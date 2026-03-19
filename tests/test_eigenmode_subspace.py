from __future__ import annotations

import numpy as np

from qec.analysis.eigenmode_subspace import cluster_eigenmodes


def test_degenerate_projector_rotation_stability() -> None:
    eigenvalues = np.array([-1.0, -1.0 + 1e-12], dtype=np.float64)
    basis_a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    theta = np.pi / 4.0
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ], dtype=np.float64)
    basis_b = basis_a @ rot

    clusters_a = cluster_eigenmodes(eigenvalues, basis_a, base_epsilon=0.01)
    clusters_b = cluster_eigenmodes(eigenvalues, basis_b, base_epsilon=0.01)

    assert len(clusters_a) == 1
    assert len(clusters_b) == 1
    assert clusters_a[0].indices == (0, 1)
    assert clusters_b[0].indices == (0, 1)
    assert np.allclose(clusters_a[0].projector, clusters_b[0].projector)
