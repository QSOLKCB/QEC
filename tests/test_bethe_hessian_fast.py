from __future__ import annotations

import numpy as np

from src.qec.analysis.bethe_hessian_fast import BetheHessianBuilder
from src.qec.analysis.spectral_frustration import (
    apply_swap,
    build_bethe_hessian,
    count_negative_modes,
    spectral_frustration_count,
)


def _example_adjacency() -> np.ndarray:
    A = np.zeros((6, 6), dtype=np.float64)
    edges = [(0, 3), (0, 4), (1, 4), (1, 5), (2, 3), (2, 5)]
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


def test_bh_equivalence_fast_vs_rebuild() -> None:
    A = _example_adjacency()
    r = 1.7

    builder = BetheHessianBuilder(A, r)
    fast = builder.build()
    rebuilt = build_bethe_hessian(A, r)

    assert np.allclose(fast, rebuilt, atol=1e-12, rtol=0.0)


def test_swap_update_equivalence() -> None:
    A = _example_adjacency()
    r = 1.7
    swap = (0, 3, 1, 5)

    builder = BetheHessianBuilder(A, r)
    fast = builder.build_after_swap(*swap)

    A_trial = apply_swap(A, *swap)
    rebuilt = build_bethe_hessian(A_trial, r)

    assert np.allclose(fast, rebuilt, atol=1e-12, rtol=0.0)


def test_frustration_equivalence() -> None:
    A = _example_adjacency()
    r = 1.7
    swaps = [(0, 3, 1, 5), (2, 5, 1, 4)]

    result_fast = spectral_frustration_count(A, r, swaps)

    baseline = count_negative_modes(build_bethe_hessian(A, r))
    expected = []
    for swap in sorted(swaps):
        A_trial = apply_swap(A, *swap)
        H_trial = build_bethe_hessian(A_trial, r)
        expected.append({"swap": swap, "negative_modes": count_negative_modes(H_trial)})

    assert result_fast["baseline_negative_modes"] == baseline
    assert result_fast["candidate_negative_modes"] == expected


def test_determinism_repeated_inertia_counts() -> None:
    A = _example_adjacency()
    r = 1.7
    swaps = [(0, 3, 1, 5), (2, 5, 1, 4)]

    runs = [spectral_frustration_count(A, r, swaps) for _ in range(5)]
    assert all(run == runs[0] for run in runs[1:])
