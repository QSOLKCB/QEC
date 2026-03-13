"""Tests for v13.2.0 localization metrics."""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.analysis.localization_metrics import IPR, ParticipationEntropy, compute_edge_energy_map


def test_ipr_uniform_vector():
    n = 16
    v = np.ones(n, dtype=np.float64) / np.sqrt(n)
    assert abs(IPR.compute(v) - (1.0 / n)) < 1e-12


def test_entropy_localized_vs_uniform():
    vu = np.ones(8, dtype=np.float64) / np.sqrt(8.0)
    vl = np.zeros(8, dtype=np.float64)
    vl[0] = 1.0
    assert ParticipationEntropy.compute(vu) > ParticipationEntropy.compute(vl)


def test_edge_energy_map_deterministic_and_sparse_supported():
    A = scipy.sparse.csr_matrix(np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
    ], dtype=np.float64))
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    m1 = compute_edge_energy_map(A, v)
    m2 = compute_edge_energy_map(A, v)
    assert m1 == m2
    assert m1[0][0:2] == (0, 1)
    assert m1[-1][0:2] == (2, 3)
