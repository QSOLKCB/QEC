"""Deterministic spectral frustration counting via Bethe-Hessian inertia."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse

from src.qec.analysis.bethe_hessian_fast import BetheHessianBuilder
from src.qec.analysis.bethe_hessian_utils import BetheHessianCache
from src.qec.analysis.eigenmode_mutation import build_bethe_hessian as build_tanner_bethe_hessian


@dataclass(frozen=True)
class SpectralFrustrationConfig:
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


class SpectralFrustrationAnalyzer:
    """Backwards-compatible deterministic class API for spectral frustration."""

    def __init__(self, config: SpectralFrustrationConfig | None = None, *args, **kwargs):
        precision = kwargs.get("precision")
        if config is None:
            cfg_precision = int(precision) if precision is not None else 12
            self.config = SpectralFrustrationConfig(precision=cfg_precision)
        else:
            if precision is None:
                self.config = config
            else:
                self.config = SpectralFrustrationConfig(
                    trap_threshold=float(config.trap_threshold),
                    precision=int(precision),
                )

    def compute(self, H, *args, **kwargs):
        return spectral_frustration_count(H, *args, **kwargs)

    def evaluate(
        self,
        A: np.ndarray | scipy.sparse.spmatrix,
        *,
        r: float,
        swaps: list[tuple[int, int, int, int]] | None = None,
        use_cache: bool = True,
    ) -> dict[str, object]:
        sorted_swaps = sorted(swaps or [])
        if not use_cache:
            return spectral_frustration_count(A, r=r, candidate_swaps=sorted_swaps)

        cache = BetheHessianCache(A, float(r))
        baseline = count_negative_modes(cache.build())
        trials: list[dict[str, object]] = []
        for ci, vi, cj, vj in sorted_swaps:
            cache_local = BetheHessianCache(A, float(r))
            neg_modes = count_negative_modes(cache_local.update_for_swap(ci, vi, cj, vj))
            trials.append({"swap": (ci, vi, cj, vj), "negative_modes": int(neg_modes)})
        return {
            "baseline_negative_modes": int(baseline),
            "candidate_negative_modes": trials,
        }

    def extract_trap_modes(self, H: np.ndarray | scipy.sparse.spmatrix) -> tuple[np.ndarray, ...]:
        B, _ = build_tanner_bethe_hessian(H)
        B_arr = np.asarray(B.toarray(), dtype=np.float64)
        if B_arr.size == 0:
            return ()
        vals, vecs = np.linalg.eigh(B_arr)
        order = np.argsort(vals, kind="stable")
        vals = vals[order]
        vecs = vecs[:, order]

        out: list[np.ndarray] = []
        for idx in range(vals.shape[0]):
            if float(vals[idx]) >= 0.0:
                continue
            v = np.asarray(vecs[:, idx], dtype=np.float64)
            norm = float(np.linalg.norm(v))
            if norm <= 1e-15:
                continue
            v = v / norm
            pivot = int(np.argmax(np.abs(v))) if v.size else 0
            if v.size and v[pivot] < 0.0:
                v = -v
            out.append(v)
        return tuple(out)

    def detect_trap_nodes(self, H: np.ndarray | scipy.sparse.spmatrix) -> tuple[int, ...]:
        modes = self.extract_trap_modes(H)
        if not modes:
            return ()
        v = np.abs(np.asarray(modes[0], dtype=np.float64))
        vmax = float(v.max()) if v.size else 0.0
        if vmax <= 1e-15:
            return ()
        threshold = float(self.config.trap_threshold) * vmax
        nodes = tuple(int(i) for i in np.where(v >= threshold)[0].tolist())
        return nodes

    def compute_frustration(self, H: np.ndarray | scipy.sparse.spmatrix) -> SpectralFrustrationResult:
        modes = self.extract_trap_modes(H)
        if not modes:
            return SpectralFrustrationResult(
                frustration_score=0.0,
                negative_modes=0,
                max_ipr=0.0,
                transport_imbalance=0.0,
                trap_modes=(),
            )

        iprs: list[float] = []
        for v in modes:
            vv = np.asarray(v, dtype=np.float64)
            p = np.square(np.abs(vv))
            z = float(np.sum(p))
            if z <= 1e-15:
                iprs.append(0.0)
                continue
            p = p / z
            iprs.append(float(np.sum(p * p)))
        max_ipr = max(iprs) if iprs else 0.0

        negative_modes = int(len(modes))
        frustration_score = round(float(negative_modes * max_ipr), self.config.precision)

        first = np.asarray(modes[0], dtype=np.float64)
        pos_sum = float(np.sum(np.abs(first[first >= 0.0])))
        neg_sum = float(np.sum(np.abs(first[first < 0.0])))
        denom = pos_sum + neg_sum
        transport_imbalance = 0.0 if denom <= 1e-15 else abs(pos_sum - neg_sum) / denom

        return SpectralFrustrationResult(
            frustration_score=frustration_score,
            negative_modes=negative_modes,
            max_ipr=round(float(max_ipr), self.config.precision),
            transport_imbalance=round(float(transport_imbalance), self.config.precision),
            trap_modes=tuple(np.asarray(v, dtype=np.float64) for v in modes),
        )
