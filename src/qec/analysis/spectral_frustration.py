"""Deterministic spectral frustration counting via Bethe-Hessian inertia."""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse

from src.qec.analysis.bethe_hessian_fast import BetheHessianBuilder


def build_bethe_hessian(A: np.ndarray | scipy.sparse.spmatrix, r: float) -> np.ndarray:
    """Construct Bethe-Hessian matrix ``H(r) = (r^2-1)I - rA + D``."""
    A_arr = np.asarray(scipy.sparse.csr_matrix(A, dtype=np.float64).toarray(), dtype=np.float64)
    r_f = float(r)
    I = np.eye(A_arr.shape[0], dtype=np.float64)
    D = np.diag(np.sum(A_arr, axis=1, dtype=np.float64)).astype(np.float64, copy=False)
    return (r_f * r_f - 1.0) * I - r_f * A_arr + D


def apply_swap(A: np.ndarray | scipy.sparse.spmatrix, ci: int, vi: int, cj: int, vj: int) -> np.ndarray:
    """Apply deterministic 2-edge swap on a symmetric adjacency matrix copy."""
    A_trial = np.asarray(scipy.sparse.csr_matrix(A, dtype=np.float64).toarray(), dtype=np.float64).copy()
    ci_i, vi_i, cj_i, vj_i = int(ci), int(vi), int(cj), int(vj)

    A_trial[ci_i, vi_i] = 0.0
    A_trial[vi_i, ci_i] = 0.0
    A_trial[cj_i, vj_i] = 0.0
    A_trial[vj_i, cj_i] = 0.0

    A_trial[ci_i, vj_i] = 1.0
    A_trial[vj_i, ci_i] = 1.0
    A_trial[cj_i, vi_i] = 1.0
    A_trial[vi_i, cj_i] = 1.0
    return A_trial


def count_negative_modes(H: np.ndarray | scipy.sparse.spmatrix) -> int:
    """Count negative modes using deterministic Sylvester inertia (LDL)."""
    H_arr = np.asarray(scipy.sparse.csr_matrix(H, dtype=np.float64).toarray(), dtype=np.float64)
    _, D, _ = scipy.linalg.ldl(H_arr, lower=True, hermitian=True)
    return int(np.sum(np.diag(D) < 0.0))


def spectral_frustration_count(
    A: np.ndarray | scipy.sparse.spmatrix,
    r: float,
    candidate_swaps: list[tuple[int, int, int, int]] | None = None,
) -> dict[str, object]:
    """Evaluate baseline and candidate frustration as negative-mode counts."""
    builder = BetheHessianBuilder(A, r)
    baseline = count_negative_modes(builder.build())

    trials: list[dict[str, object]] = []
    for ci, vi, cj, vj in sorted(candidate_swaps or []):
        H_trial = builder.build_after_swap(ci, vi, cj, vj)
        neg_modes = count_negative_modes(H_trial)
        trials.append({"swap": (ci, vi, cj, vj), "negative_modes": int(neg_modes)})

    return {
        "baseline_negative_modes": int(baseline),
        "candidate_negative_modes": trials,
    }
