"""v15.0.0 — Spectral Frustration Counting diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from src.qec.analysis.eigenmode_mutation import build_bethe_hessian
from src.qec.analysis.nonbacktracking_flow import NonBacktrackingEigenvectorFlowAnalyzer

_ROUND = 12


@dataclass(frozen=True)
class SpectralFrustrationResult:
    frustration_score: float
    negative_modes: int
    bh_energy: float
    max_ipr: float
    nb_transport_imbalance: float


@dataclass(frozen=True)
class SpectralFrustrationConfig:
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.25
    delta: float = 0.25
    eigenvalue_eps: float = 1e-9
    trap_threshold: float = 0.25
    compute_ipr: bool = True


def count_negative_modes(H_bh: np.ndarray) -> int:
    """Count negative eigenmodes via deterministic LDL inertia."""
    mat = np.asarray(H_bh, dtype=np.float64)
    if mat.size == 0:
        return 0
    _L, D, _perm = la.ldl(mat, lower=True, hermitian=True)
    neg = 0
    idx = 0
    n = D.shape[0]
    while idx < n:
        if idx + 1 < n and abs(float(D[idx + 1, idx])) > 0.0:
            block = np.asarray(D[idx:idx + 2, idx:idx + 2], dtype=np.float64)
            neg += int(np.sum(np.linalg.eigvalsh(block) < 0.0))
            idx += 2
            continue
        if float(D[idx, idx]) < 0.0:
            neg += 1
        idx += 1
    return int(neg)


class SpectralFrustrationAnalyzer:
    """Compute deterministic spectral frustration proxies for Tanner graphs."""

    def __init__(self, config: SpectralFrustrationConfig | None = None) -> None:
        self.config = config or SpectralFrustrationConfig()
        self._nb = NonBacktrackingEigenvectorFlowAnalyzer()

    def compute_frustration(self, H: np.ndarray) -> SpectralFrustrationResult:
        H_csr = sp.csr_matrix(H, dtype=np.float64)
        if H_csr.shape[0] == 0 or H_csr.shape[1] == 0 or H_csr.nnz == 0:
            return SpectralFrustrationResult(
                frustration_score=0.0,
                negative_modes=0,
                bh_energy=0.0,
                max_ipr=0.0,
                nb_transport_imbalance=0.0,
            )

        B_sparse, _r = build_bethe_hessian(H_csr)
        B = np.asarray(B_sparse.toarray(), dtype=np.float64)
        negative_modes = count_negative_modes(B)

        eigvals = np.linalg.eigvalsh(B)
        neg_vals = eigvals[eigvals < 0.0]
        bh_energy = float(np.sum(np.abs(neg_vals))) if neg_vals.size else 0.0

        max_ipr = 0.0
        if self.config.compute_ipr:
            vals_full, vecs_full = np.linalg.eigh(B)
            mask = vals_full < float(self.config.eigenvalue_eps)
            if np.any(mask):
                vecs = vecs_full[:, mask]
                iprs = np.sum(np.abs(vecs) ** 4, axis=0)
                max_ipr = float(np.max(iprs)) if iprs.size else 0.0

        nb_transport_imbalance = self._compute_nb_transport_imbalance(H_csr)

        score = (
            float(self.config.alpha) * float(negative_modes)
            + float(self.config.beta) * float(bh_energy)
            + float(self.config.gamma) * float(max_ipr)
            + float(self.config.delta) * float(nb_transport_imbalance)
        )

        return SpectralFrustrationResult(
            frustration_score=round(float(score), _ROUND),
            negative_modes=int(negative_modes),
            bh_energy=round(float(bh_energy), _ROUND),
            max_ipr=round(float(max_ipr), _ROUND),
            nb_transport_imbalance=round(float(nb_transport_imbalance), _ROUND),
        )

    def detect_trap_nodes(self, H: np.ndarray, threshold: float | None = None) -> tuple[int, ...]:
        H_csr = sp.csr_matrix(H, dtype=np.float64)
        if H_csr.nnz == 0:
            return ()
        B_sparse, _r = build_bethe_hessian(H_csr)
        B = np.asarray(B_sparse.toarray(), dtype=np.float64)
        vals, vecs = np.linalg.eigh(B)
        mask = vals < float(self.config.eigenvalue_eps)
        if not np.any(mask):
            return ()

        tau = float(self.config.trap_threshold if threshold is None else threshold)
        selected = vecs[:, mask]
        support = np.sum(np.abs(selected) ** 2, axis=1)
        nodes = np.flatnonzero(support > tau)
        return tuple(int(x) for x in nodes.tolist())

    def _compute_nb_transport_imbalance(self, H: sp.csr_matrix) -> float:
        flow = self._nb.build_flow_field(H)
        directed_edges = flow.directed_edges
        if len(directed_edges) == 0:
            return 0.0

        pressure = np.asarray(flow.directed_pressure, dtype=np.float64)
        index = {edge: i for i, edge in enumerate(directed_edges)}
        diffs: list[float] = []
        for (u, v), idx in sorted(index.items()):
            if u >= v:
                continue
            rev = (v, u)
            jdx = index.get(rev)
            if jdx is None:
                continue
            diffs.append(abs(float(pressure[idx]) - float(pressure[jdx])))
        if not diffs:
            return 0.0
        return float(np.mean(np.asarray(diffs, dtype=np.float64)))
