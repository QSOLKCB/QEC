"""Spectral frustration counting via Bethe-Hessian inertia."""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse

from src.qec.analysis.bethe_hessian_utils import BetheHessianCache, build_bethe_hessian


Swap = tuple[int, int, int, int]


def count_negative_modes(H: np.ndarray | scipy.sparse.spmatrix) -> int:
    """Count negative inertia modes using deterministic LDL^T."""
    H_dense = H.toarray() if scipy.sparse.issparse(H) else np.asarray(H, dtype=np.float64)
    _, D, _ = scipy.linalg.ldl(H_dense, lower=True, hermitian=True)
    diag = np.diag(D).astype(np.float64, copy=False)
    return int(np.sum(diag < 0.0))


def _apply_swap_to_adjacency(A: np.ndarray | scipy.sparse.spmatrix, ci: int, vi: int, cj: int, vj: int) -> np.ndarray | scipy.sparse.spmatrix:
    cache = BetheHessianCache(A, r=1.0)
    cache.update_for_swap(ci, vi, cj, vj)
    return cache.A


class SpectralFrustrationAnalyzer:
    """Deterministic frustration evaluation with opt-in BH caching."""

    def evaluate(
        self,
        A: np.ndarray | scipy.sparse.spmatrix,
        r: float,
        swaps: list[Swap] | None = None,
        use_cache: bool = False,
    ) -> dict[str, object]:
        if use_cache:
            cache = BetheHessianCache(A, r)
            H = cache.build()
            base = count_negative_modes(H)
            swap_modes: list[int] = []
            for ci, vi, cj, vj in (swaps or []):
                H_next = cache.update_for_swap(ci, vi, cj, vj)
                swap_modes.append(count_negative_modes(H_next))
            return {
                "negative_modes": base,
                "swap_negative_modes": swap_modes,
            }

        A_curr = scipy.sparse.csr_matrix(A, dtype=np.float64) if scipy.sparse.issparse(A) else np.asarray(A, dtype=np.float64)
        H, _, _ = build_bethe_hessian(A_curr, r)
        base = count_negative_modes(H)

        swap_modes = []
        for ci, vi, cj, vj in (swaps or []):
            A_curr = _apply_swap_to_adjacency(A_curr, ci, vi, cj, vj)
            H_next, _, _ = build_bethe_hessian(A_curr, r)
            swap_modes.append(count_negative_modes(H_next))

        return {
            "negative_modes": base,
            "swap_negative_modes": swap_modes,
        }
