"""Deterministic spectral frustration counting via Bethe-Hessian inertia."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse

from src.qec.analysis.bethe_hessian_fast import BetheHessianBuilder
from src.qec.analysis.eigenmode_mutation import build_bethe_hessian as build_tanner_bethe_hessian


@dataclass(frozen=True)
class SpectralFrustrationConfig:
    trap_threshold: float = 0.2
    precision: int = 12


@dataclass(frozen=True)
class SpectralFrustrationResult:
    frustration_score: float
    negative_modes: int
    max_ipr: float
    transport_imbalance: float
    trap_modes: tuple[tuple[float, ...], ...]


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
    """Deterministic helper for spectral frustration diagnostics."""

    def __init__(
        self,
        config: SpectralFrustrationConfig | None = None,
        *,
        precision: int | None = None,
    ) -> None:
        cfg = config or SpectralFrustrationConfig()
        if precision is not None:
            cfg = SpectralFrustrationConfig(
                trap_threshold=float(cfg.trap_threshold),
                precision=int(precision),
            )
        self.config = cfg
        self.precision = int(cfg.precision)

    def extract_trap_modes(self, H: np.ndarray | scipy.sparse.spmatrix) -> tuple[tuple[float, ...], ...]:
        B, _ = build_tanner_bethe_hessian(H)
        vals, vecs = np.linalg.eigh(np.asarray(B.toarray(), dtype=np.float64))
        neg_indices = [int(i) for i, val in enumerate(vals.tolist()) if float(val) < 0.0]
        ordered = sorted(
            (
                np.asarray(vecs[:, idx], dtype=np.float64)
                for idx in neg_indices
            ),
            key=lambda vec: tuple(float(x) for x in np.round(np.abs(vec), self.precision)),
        )
        return tuple(tuple(float(x) for x in vec.tolist()) for vec in ordered)

    def detect_trap_nodes(self, H: np.ndarray | scipy.sparse.spmatrix) -> list[int]:
        modes = self.extract_trap_modes(H)
        if not modes:
            return []
        lead = np.abs(np.asarray(modes[0], dtype=np.float64))
        if lead.size == 0:
            return []
        threshold = float(np.max(lead) * float(self.config.trap_threshold))
        nodes = [int(idx) for idx, value in enumerate(lead.tolist()) if float(value) >= threshold]
        return sorted(set(nodes))

    def compute_frustration(self, H: np.ndarray | scipy.sparse.spmatrix) -> SpectralFrustrationResult:
        B, _ = build_tanner_bethe_hessian(H)
        B_dense = np.asarray(B.toarray(), dtype=np.float64)
        negative_modes = count_negative_modes(B_dense)
        modes = self.extract_trap_modes(H)
        if modes:
            ipr_values = [float(np.sum(np.asarray(mode, dtype=np.float64) ** 4)) for mode in modes]
            max_ipr = float(max(ipr_values))
            lead = np.abs(np.asarray(modes[0], dtype=np.float64))
            transport_imbalance = float(np.max(lead) - float(np.mean(lead)))
        else:
            max_ipr = 0.0
            transport_imbalance = 0.0
        frustration_score = float(negative_modes) + max_ipr + transport_imbalance
        p = self.precision
        return SpectralFrustrationResult(
            frustration_score=round(float(frustration_score), p),
            negative_modes=int(negative_modes),
            max_ipr=round(float(max_ipr), p),
            transport_imbalance=round(float(transport_imbalance), p),
            trap_modes=tuple(tuple(float(x) for x in np.asarray(mode, dtype=np.float64)) for mode in modes),
        )

    def evaluate(
        self,
        A: np.ndarray | scipy.sparse.spmatrix,
        *,
        r: float,
        swaps: list[tuple[int, int, int, int]] | None = None,
        use_cache: bool = True,
    ) -> dict[str, object]:
        _ = use_cache
        return spectral_frustration_count(A, r, candidate_swaps=swaps)
