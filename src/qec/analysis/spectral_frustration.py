"""Deterministic spectral frustration counting via Bethe-Hessian inertia."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse

from src.qec.analysis.bethe_hessian_utils import BetheHessianCache, build_bethe_hessian as build_bh_cached
from src.qec.analysis.bethe_hessian_fast import BetheHessianBuilder
from src.qec.analysis.bethe_hessian_utils import BetheHessianCache
from src.qec.analysis.eigenmode_mutation import build_bethe_hessian as build_tanner_bethe_hessian


@dataclass(frozen=True)
class SpectralFrustrationConfig:
    trap_threshold: float = 0.15
    precision: int = 12


@dataclass(frozen=True, eq=False)
    trap_threshold: float = 0.1
    precision: int = 12


@dataclass(frozen=True)
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
        if (
            float(self.frustration_score) != float(other.frustration_score)
            or int(self.negative_modes) != int(other.negative_modes)
            or float(self.max_ipr) != float(other.max_ipr)
            or float(self.transport_imbalance) != float(other.transport_imbalance)
            or len(self.trap_modes) != len(other.trap_modes)
        ):
            return False
        for a, b in zip(self.trap_modes, other.trap_modes):
            if not np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)):
                return False
        return True


@dataclass(frozen=True)
class SpectralFrustrationConfig:
    trap_threshold: float = 0.15
    r: float = 1.5
    precision: int = 12
    normalize_frustration: bool = False


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
        if (
            self.frustration_score != other.frustration_score
            or self.negative_modes != other.negative_modes
            or self.max_ipr != other.max_ipr
            or self.transport_imbalance != other.transport_imbalance
            or len(self.trap_modes) != len(other.trap_modes)
        ):
            return False
        return all(np.array_equal(a, b) for a, b in zip(self.trap_modes, other.trap_modes))

def _to_dense_float64(A: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
    if isinstance(A, np.ndarray):
        return np.asarray(A, dtype=np.float64)
    return np.asarray(scipy.sparse.csr_matrix(A, dtype=np.float64).toarray(), dtype=np.float64)


def build_bethe_hessian(A: np.ndarray | scipy.sparse.spmatrix, r: float) -> np.ndarray:
    """Construct Bethe-Hessian matrix ``H(r) = (r^2-1)I - rA + D``."""
    H, _, _ = build_bh_cached(A, r)
    if scipy.sparse.issparse(H):
        return np.asarray(H.toarray(), dtype=np.float64)
    return np.asarray(H, dtype=np.float64)


def apply_swap(A: np.ndarray | scipy.sparse.spmatrix, ci: int, vi: int, cj: int, vj: int) -> np.ndarray:
    """Apply deterministic 2-edge swap on a symmetric adjacency matrix copy."""
    A_trial = _to_dense_float64(A).copy()
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
    H_arr = _to_dense_float64(H)
    _, D, _ = scipy.linalg.ldl(H_arr, lower=True, hermitian=True)
    return int(np.sum(np.diag(D) < 0.0))




def _to_bipartite_adjacency(H: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
    arr = np.asarray(scipy.sparse.csr_matrix(H, dtype=np.float64).toarray(), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("expected 2D matrix")
    m, n = arr.shape
    if m == n and np.allclose(arr, arr.T, rtol=0.0, atol=1e-12):
        return arr
    top = np.hstack([np.zeros((m, m), dtype=np.float64), arr])
    bottom = np.hstack([arr.T, np.zeros((n, n), dtype=np.float64)])
    return np.vstack([top, bottom])

def _estimate_r(A: np.ndarray) -> float:
    deg = np.asarray(np.sum(A, axis=1), dtype=np.float64)
    avg = float(np.mean(deg)) if deg.size > 0 else 0.0
    return float(max(1.1, np.sqrt(max(avg, 1.0))))


class SpectralFrustrationAnalyzer:
    """Deterministic frustration analyzer with trap-mode localization."""

    def __init__(self, config: SpectralFrustrationConfig | None = None, *, precision: int | None = None) -> None:
        if config is None:
            config = SpectralFrustrationConfig()
        if precision is not None:
            config = SpectralFrustrationConfig(trap_threshold=config.trap_threshold, precision=int(precision))
        self.config = config

    def extract_trap_modes(self, H: np.ndarray | scipy.sparse.spmatrix, r: float | None = None) -> tuple[np.ndarray, ...]:
        arr = np.asarray(scipy.sparse.csr_matrix(H, dtype=np.float64).toarray(), dtype=np.float64)
        if arr.ndim != 2:
            return tuple()
        m, n = arr.shape
        # Variable-space localization proxy; deterministic sorting by eigenvalue index.
        gram = arr.T @ arr
        evals, evecs = np.linalg.eigh(gram)
        order = np.argsort(evals, kind="stable")
        picked: list[np.ndarray] = []
        best_idx = None
        best_key = None
        for idx in order:
            ev = float(evals[idx])
            if ev <= 1e-12:
                continue
            var_vec = np.asarray(evecs[:, idx], dtype=np.float64)
            denom = float(np.sum(var_vec * var_vec))
            if denom <= 0.0:
                continue
            ipr = float(np.sum((var_vec * var_vec) ** 2) / (denom * denom))
            key = (-ipr, ev, int(idx))
            if best_key is None or key < best_key:
                best_key = key
                best_idx = int(idx)
            if ipr >= float(self.config.trap_threshold):
                mode = np.concatenate([var_vec, np.zeros(m, dtype=np.float64)])
                picked.append(mode)
        if not picked and best_idx is not None:
            var_vec = np.asarray(evecs[:, best_idx], dtype=np.float64)
            picked.append(np.concatenate([var_vec, np.zeros(m, dtype=np.float64)]))
        return tuple(picked)

    def detect_trap_nodes(self, H: np.ndarray | scipy.sparse.spmatrix, r: float | None = None) -> list[int]:
        modes = self.extract_trap_modes(H, r=r)
        if not modes:
            return []
        mode_scores = np.mean(np.abs(np.vstack(modes)), axis=0)
        if mode_scores.size == 0:
            return []
        max_score = float(np.max(mode_scores))
        if max_score <= 0.0:
            return []
        threshold = 0.5 * max_score
        nodes = [int(i) for i, s in enumerate(mode_scores.tolist()) if float(s) >= threshold]
        if not nodes:
            order = np.argsort(-mode_scores, kind="stable")
            nodes = [int(order[0])]
        nodes.sort()
        return nodes

    def compute_frustration(self, H: np.ndarray | scipy.sparse.spmatrix, r: float | None = None) -> SpectralFrustrationResult:
        A = _to_bipartite_adjacency(H)
        r_eff = float(r) if r is not None else _estimate_r(A)
        B = build_bethe_hessian(A, r_eff)
        evals, evecs = np.linalg.eigh(B)
        neg = int(np.sum(evals < 0.0))
        iprs: list[float] = []
        for idx in np.argsort(evals, kind="stable"):
            if float(evals[idx]) >= 0.0:
                continue
            vec = np.asarray(evecs[:, idx], dtype=np.float64)
            denom = float(np.sum(vec * vec))
            if denom <= 0.0:
                continue
            iprs.append(float(np.sum((vec * vec) ** 2) / (denom * denom)))
        max_ipr = float(max(iprs) if iprs else 0.0)
        frustration = round(float(neg + max_ipr), self.config.precision)
        trap_modes = self.extract_trap_modes(A, r=r_eff)
        return SpectralFrustrationResult(
            frustration_score=frustration,
            negative_modes=neg,
            max_ipr=round(max_ipr, self.config.precision),
            transport_imbalance=0.0,
            trap_modes=trap_modes,
        )

    def evaluate(
        self,
        A: np.ndarray | scipy.sparse.spmatrix,
        *,
        r: float,
        swaps: list[tuple[int, int, int, int]],
        use_cache: bool = True,
    ) -> dict[str, object]:
        r_f = float(r)
        base_B = build_bethe_hessian(A, r_f)
        baseline = count_negative_modes(base_B)
        trials: list[dict[str, object]] = []
        if use_cache:
            cache = BetheHessianCache(A, r_f)
            for ci, vi, cj, vj in sorted(swaps):
                H_trial = cache.update_for_swap(ci, vi, cj, vj)
                trials.append({"swap": (ci, vi, cj, vj), "negative_modes": int(count_negative_modes(H_trial))})
        else:
            for ci, vi, cj, vj in sorted(swaps):
                A_trial = apply_swap(A, ci, vi, cj, vj)
                H_trial = build_bethe_hessian(A_trial, r_f)
                trials.append({"swap": (ci, vi, cj, vj), "negative_modes": int(count_negative_modes(H_trial))})
        return {
            "baseline_negative_modes": int(baseline),
            "candidate_negative_modes": trials,
        }


def spectral_frustration_count(
    A: np.ndarray | scipy.sparse.spmatrix,
    r: float,
    candidate_swaps: list[tuple[int, int, int, int]] | None = None,
    flow_field: object | None = None,
) -> dict[str, object]:
    """Evaluate baseline and candidate frustration as negative-mode counts."""
    analyzer = SpectralFrustrationAnalyzer()
    return analyzer.evaluate(A, r=float(r), swaps=list(candidate_swaps or []), use_cache=True)
