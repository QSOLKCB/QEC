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
    if scipy.sparse.issparse(adjacency):
        A = scipy.sparse.coo_matrix(adjacency.astype(np.float64))
        rows = A.row
        cols = A.col
    else:
        A = np.asarray(adjacency, dtype=np.float64)
        rows, cols = np.nonzero(A)

    v = np.asarray(eigenvector, dtype=np.float64).ravel()
    power2 = np.abs(v) ** 2

    edges: list[tuple[int, int, float]] = []
    for i, j in zip(rows, cols):
        ii = int(i)
        jj = int(j)
        if ii >= jj:
            continue
        if ii >= power2.size or jj >= power2.size:
            continue
        edge_energy = float(power2[ii] + power2[jj])
        edges.append((ii, jj, round(edge_energy, _ROUND)))

    edges.sort(key=lambda x: (x[0], x[1]))
    return edges
