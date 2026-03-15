"""Deterministic spectral frustration counting via Bethe-Hessian inertia."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse

from src.qec.analysis.bethe_hessian_utils import BetheHessianCache
from src.qec.analysis.eigenmode_mutation import build_bethe_hessian as build_tanner_bethe_hessian

_ROUND = 12


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


@dataclass(frozen=True)
class SpectralFrustrationConfig:
    trap_threshold: float = 0.1
    precision: int = _ROUND


@dataclass(frozen=True, eq=False)
class SpectralFrustrationResult:
    frustration_score: float
    negative_modes: int
    max_ipr: float
    transport_imbalance: float
    trap_modes: tuple[np.ndarray, ...]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpectralFrustrationResult):
            return False
        if self.frustration_score != other.frustration_score:
            return False
        if self.negative_modes != other.negative_modes:
            return False
        if self.max_ipr != other.max_ipr:
            return False
        if self.transport_imbalance != other.transport_imbalance:
            return False
        if len(self.trap_modes) != len(other.trap_modes):
            return False
        return all(np.array_equal(a, b) for a, b in zip(self.trap_modes, other.trap_modes))


class SpectralFrustrationAnalyzer:
    def __init__(self, config: SpectralFrustrationConfig | None = None, precision: int | None = None) -> None:
        self.config = config if config is not None else SpectralFrustrationConfig()
        self.precision = int(self.config.precision if precision is None else precision)

    def extract_trap_modes(self, H: np.ndarray | scipy.sparse.spmatrix) -> tuple[np.ndarray, ...]:
        B, _ = build_tanner_bethe_hessian(H)
        vals, vecs = np.linalg.eigh(B.toarray().astype(np.float64, copy=False))
        neg_ids = np.where(vals < 0.0)[0]
        if neg_ids.size == 0:
            return tuple()
        modes = [np.asarray(vecs[:, int(i)], dtype=np.float64).copy() for i in neg_ids.tolist()]
        modes.sort(key=lambda v: (-float(np.max(np.abs(v))), -float(np.sum(v * v)), tuple(np.abs(v).tolist())))
        return tuple(modes)

    def detect_trap_nodes(self, H: np.ndarray | scipy.sparse.spmatrix) -> list[int]:
        modes = self.extract_trap_modes(H)
        if not modes:
            return []
        mode = np.abs(np.asarray(modes[0], dtype=np.float64))
        threshold = float(self.config.trap_threshold) * float(mode.max() if mode.size else 0.0)
        nodes = [int(i) for i in np.where(mode >= threshold)[0].tolist()]
        nodes.sort()
        return nodes

    def compute_frustration(self, H: np.ndarray | scipy.sparse.spmatrix) -> SpectralFrustrationResult:
        B, _ = build_tanner_bethe_hessian(H)
        B_arr = np.asarray(B.toarray(), dtype=np.float64)
        eigvals, eigvecs = np.linalg.eigh(B_arr)
        eigvals = np.asarray(eigvals, dtype=np.float64)
        neg_ids = np.where(eigvals < 0.0)[0]
        negative_modes = int(neg_ids.size)
        if negative_modes == 0:
            max_ipr = 0.0
            transport = 0.0
            frustration = 0.0
            trap_modes: tuple[np.ndarray, ...] = tuple()
        else:
            neg_vecs = np.asarray(eigvecs[:, neg_ids], dtype=np.float64)
            iprs = np.sum(neg_vecs ** 4, axis=0, dtype=np.float64)
            max_ipr = float(np.max(iprs)) if iprs.size else 0.0
            frustration = float(np.sum(np.abs(eigvals[neg_ids]), dtype=np.float64))
            transport = float(np.mean(np.abs(neg_vecs), dtype=np.float64))
            trap_modes = tuple(np.asarray(neg_vecs[:, i], dtype=np.float64).copy() for i in range(neg_vecs.shape[1]))
        return SpectralFrustrationResult(
            frustration_score=float(np.round(np.float64(frustration), self.precision)),
            negative_modes=negative_modes,
            max_ipr=float(np.round(np.float64(max_ipr), self.precision)),
            transport_imbalance=float(np.round(np.float64(transport), self.precision)),
            trap_modes=trap_modes,
        )

    def evaluate(
        self,
        A: np.ndarray | scipy.sparse.spmatrix,
        *,
        r: float,
        swaps: list[tuple[int, int, int, int]] | None = None,
        use_cache: bool = True,
    ) -> dict[str, object]:
        trials: list[dict[str, object]] = []
        swap_list = sorted(swaps or [])
        if use_cache:
            baseline = count_negative_modes(BetheHessianCache(A, r).build())
            for ci, vi, cj, vj in swap_list:
                cache = BetheHessianCache(A, r)
                H_trial = cache.update_for_swap(ci, vi, cj, vj)
                trials.append({"swap": (ci, vi, cj, vj), "negative_modes": int(count_negative_modes(H_trial))})
        else:
            H_base = build_bethe_hessian(A, r)
            baseline = count_negative_modes(H_base)
            for ci, vi, cj, vj in swap_list:
                H_trial = build_bethe_hessian(apply_swap(A, ci, vi, cj, vj), r)
                trials.append({"swap": (ci, vi, cj, vj), "negative_modes": int(count_negative_modes(H_trial))})

        return {"baseline_negative_modes": int(baseline), "candidate_negative_modes": trials}


def spectral_frustration_count(
    A: np.ndarray | scipy.sparse.spmatrix,
    r: float,
    candidate_swaps: list[tuple[int, int, int, int]] | None = None,
) -> dict[str, object]:
    """Evaluate baseline and candidate frustration as negative-mode counts."""
    analyzer = SpectralFrustrationAnalyzer()
    return analyzer.evaluate(A, r=r, swaps=candidate_swaps, use_cache=True)
