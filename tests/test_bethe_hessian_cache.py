from __future__ import annotations

import numpy as np

from qec.analysis.bethe_hessian_utils import BetheHessianCache, build_bethe_hessian
from qec.analysis.spectral_frustration import SpectralFrustrationAnalyzer, count_negative_modes


def _adjacency() -> np.ndarray:
    A = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
    ], dtype=np.float64)
    return A


def _apply_swap_dense(A: np.ndarray, ci: int, vi: int, cj: int, vj: int) -> np.ndarray:
    B = np.array(A, dtype=np.float64, copy=True)
    for i, j, val in [
        (ci, vi, 0.0),
        (vi, ci, 0.0),
        (cj, vj, 0.0),
        (vj, cj, 0.0),
        (ci, vj, 1.0),
        (vj, ci, 1.0),
        (cj, vi, 1.0),
        (vi, cj, 1.0),
    ]:
        if 0 <= i < B.shape[0] and 0 <= j < B.shape[0] and i != j:
            B[i, j] = val
    return B


def test_cache_matches_direct_build():
    A = _adjacency()
    r = 1.75
    H_ref, _, _ = build_bethe_hessian(A, r)

    cache = BetheHessianCache(A, r)
    H_cached = cache.build()

    np.testing.assert_allclose(H_cached, H_ref, rtol=0.0, atol=1e-12)


def test_swap_update_matches_recomputation():
    A = _adjacency()
    r = 1.4
    swap = (0, 1, 2, 3)

    cache = BetheHessianCache(A, r)
    H_cached = cache.update_for_swap(*swap)

    A_swapped = _apply_swap_dense(A, *swap)
    H_ref, _, _ = build_bethe_hessian(A_swapped, r)

    np.testing.assert_allclose(H_cached, H_ref, rtol=0.0, atol=1e-12)


def test_deterministic_cached_build_and_inertia():
    A = _adjacency()
    r = 1.9

    cache1 = BetheHessianCache(A, r)
    cache2 = BetheHessianCache(A, r)

    H1 = cache1.build()
    H2 = cache2.build()

    np.testing.assert_array_equal(H1, H2)
    assert count_negative_modes(H1) == count_negative_modes(H2)


def test_frustration_equivalence_cached_vs_rebuild():
    A = _adjacency()
    r = 1.6
    swaps = [(0, 1, 2, 3), (1, 2, 0, 3)]

    analyzer = SpectralFrustrationAnalyzer()
    cached = analyzer.evaluate(A, r=r, swaps=swaps, use_cache=True)
    rebuilt = analyzer.evaluate(A, r=r, swaps=swaps, use_cache=False)

    assert cached == rebuilt


def test_cached_evaluations_stable_across_repeated_runs():
    A = _adjacency()
    r = 1.6
    swaps = [(0, 1, 2, 3), (1, 2, 0, 3), (0, 2, 1, 3)]

    analyzer = SpectralFrustrationAnalyzer()
    ref = analyzer.evaluate(A, r=r, swaps=swaps, use_cache=True)

    for _ in range(10):
        cur = analyzer.evaluate(A, r=r, swaps=swaps, use_cache=True)
        assert cur == ref
