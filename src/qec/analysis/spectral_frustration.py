"""v15.0.0 — Deterministic spectral frustration analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse

from src.qec.analysis.defect_catalog import _build_bethe_hessian, _extract_negative_modes
from src.qec.analysis.nonbacktracking_flow import NonBacktrackingFlowAnalyzer


_ROUND = 12
_EPS = 1e-9


@dataclass(frozen=True)
class SpectralFrustrationResult:
    frustration_score: float
    negative_modes: int
    max_ipr: float
    transport_imbalance: float
    trap_modes: tuple[np.ndarray, ...]


class SpectralFrustrationAnalyzer:
    """Compute deterministic frustration score from spectral diagnostics."""

    def __init__(self, *, precision: int = _ROUND) -> None:
        self.precision = int(precision)
        self._flow = NonBacktrackingFlowAnalyzer()

    def extract_trap_modes(
        self,
        H: np.ndarray | scipy.sparse.spmatrix,
        *,
        epsilon: float = _EPS,
    ) -> tuple[np.ndarray, ...]:
        H_csr = scipy.sparse.csr_matrix(H, dtype=np.float64)
        HB, _ = _build_bethe_hessian(H_csr)
        vals, vecs = _extract_negative_modes(HB)

        modes: list[np.ndarray] = []
        for i, eig in enumerate(vals):
            if float(eig) >= float(epsilon):
                continue
            vec = np.asarray(vecs[:, i], dtype=np.float64)
            norm = float(np.linalg.norm(vec))
            if norm <= 0.0:
                continue
            vec = vec / norm
            phase_idx = int(np.argmax(np.abs(vec)))
            if float(vec[phase_idx]) < 0.0:
                vec = -vec
            modes.append(np.asarray(np.round(vec, self.precision), dtype=np.float64))

        return tuple(modes)

    def compute_frustration(self, H: np.ndarray | scipy.sparse.spmatrix) -> SpectralFrustrationResult:
        H_arr = np.asarray(H.todense() if scipy.sparse.issparse(H) else H, dtype=np.float64)
        if H_arr.size == 0:
            return SpectralFrustrationResult(0.0, 0, 0.0, 0.0, tuple())

        trap_modes = self.extract_trap_modes(H_arr)
        negative_modes = len(trap_modes)

        max_ipr = 0.0
        for vec in trap_modes:
            abs_vec = np.abs(np.asarray(vec, dtype=np.float64))
            norm_sq = float(np.dot(abs_vec, abs_vec))
            if norm_sq <= 0.0:
                continue
            ipr = float(np.sum(abs_vec ** 4) / (norm_sq * norm_sq))
            if ipr > max_ipr:
                max_ipr = ipr

        flow = self._flow.compute_flow(H_arr)
        directed_flow = np.asarray(flow.get("directed_edge_flow", np.zeros(0, dtype=np.float64)), dtype=np.float64)
        transport_imbalance = 0.0
        if directed_flow.size > 0:
            transport_imbalance = float(np.sum(np.abs(directed_flow)) / directed_flow.size)

        score = float(negative_modes) + float(max_ipr) + float(transport_imbalance)
        return SpectralFrustrationResult(
            frustration_score=round(score, self.precision),
            negative_modes=int(negative_modes),
            max_ipr=round(float(max_ipr), self.precision),
            transport_imbalance=round(float(transport_imbalance), self.precision),
            trap_modes=trap_modes,
        )
