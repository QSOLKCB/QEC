"""
v13.2.0 — Eigenvector localization metrics and spectral instability score.

Deterministic utilities for localization analysis of Tanner-graph spectral
modes, including IPR, participation entropy, edge-energy projection,
and a composite spectral-instability score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse


_ROUND = 12
_EPS = 1e-30


@dataclass(frozen=True)
class IPR:
    """Inverse Participation Ratio metric."""

    @staticmethod
    def compute(eigenvector: np.ndarray) -> float:
        v = np.asarray(eigenvector, dtype=np.float64).ravel()
        power2 = np.abs(v) ** 2
        denom = float(np.sum(power2))
        if denom <= _EPS:
            return 0.0
        ipr = float(np.sum(power2 ** 2)) / (denom * denom)
        ipr = min(1.0, max(0.0, ipr))
        return round(ipr, _ROUND)


@dataclass(frozen=True)
class ParticipationEntropy:
    """Participation entropy metric computed from |v_i|^2."""

    @staticmethod
    def compute(eigenvector: np.ndarray) -> float:
        v = np.asarray(eigenvector, dtype=np.float64).ravel()
        power2 = np.abs(v) ** 2
        total = float(np.sum(power2))
        if total <= _EPS:
            return 0.0
        p = power2 / total
        mask = p > 0.0
        entropy = float(-np.sum(p[mask] * np.log(p[mask])))
        return round(entropy, _ROUND)


@dataclass(frozen=True)
class SpectralInstabilityScore:
    """Composite spectral instability diagnostic."""

    @staticmethod
    def compute(lambda_min: float, ipr: float, entropy: float) -> dict[str, float]:
        lam = round(float(lambda_min), _ROUND)
        ipr_f = round(float(ipr), _ROUND)
        ent = round(float(entropy), _ROUND)
        if ent <= _EPS:
            score = 0.0
        else:
            score = abs(lam) * ipr_f / ent
        return {
            "lambda_min": lam,
            "ipr": ipr_f,
            "entropy": ent,
            "spectral_instability_score": round(float(score), _ROUND),
        }


def compute_edge_energy_map(adjacency: np.ndarray | scipy.sparse.spmatrix, eigenvector: np.ndarray) -> list[tuple[int, int, float]]:
    """Project node amplitudes onto graph edges deterministically.

    Returns a stably sorted list of tuples ``(i, j, edge_energy)`` for all
    undirected edges with ``i < j`` and ``A[i, j] != 0``.
    """
    A = scipy.sparse.coo_matrix(adjacency, dtype=np.float64)
    if (A != A.T).nnz != 0:
        A = ((A + A.T) != 0).astype(np.float64).tocoo()
    else:
        A = A.tocoo()

    v = np.asarray(eigenvector, dtype=np.float64).ravel()
    if v.size == 0 or A.nnz == 0:
        return []

    row = A.row.astype(np.int64, copy=False)
    col = A.col.astype(np.int64, copy=False)
    valid = (row < col) & (row >= 0) & (col >= 0) & (row < v.size) & (col < v.size)
    if not np.any(valid):
        return []

    row = row[valid]
    col = col[valid]
    edge_energy = v[row] * v[col]

    order = np.lexsort((col, row))
    row = row[order]
    col = col[order]
    edge_energy = edge_energy[order]

    return [
        (int(i), int(j), round(float(e), _ROUND))
        for i, j, e in zip(row.tolist(), col.tolist(), edge_energy.tolist())
    ]
