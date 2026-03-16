"""v14.0.0 — Deterministic Ihara-Bass projected edge gradient."""

from __future__ import annotations

import numpy as np

from src.qec.analysis.constants import MIN_EIGENVECTOR_NORM


def compute_ipr(v: np.ndarray) -> float:
    vec = np.asarray(v, dtype=np.float64)
    return float(np.sum(vec ** 4))


def compute_severity(eigenvalue: float, ipr: float) -> float:
    return float(abs(float(eigenvalue)) * float(ipr))


def compute_mode_edge_gradient(
    v: np.ndarray,
    adj_list: list[tuple[int, int]],
    r: float,
) -> dict[tuple[int, int], float]:
    vec = np.asarray(v, dtype=np.float64)
    vec_norm = float(np.linalg.norm(vec, ord=2))
    grad: dict[tuple[int, int], float] = {}
    if not np.isfinite(vec_norm) or vec_norm <= MIN_EIGENVECTOR_NORM:
        return grad
    for u, w in sorted(adj_list):
        key = (u, w) if u <= w else (w, u)
        grad[key] = float(-r * vec[u] * vec[w])
    return grad


def compute_ihara_bass_gradient(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    ipr_scores: np.ndarray,
    adj_list: list[tuple[int, int]],
    r: float,
    *,
    dual_operator: bool = False,
    bh_eigenvectors: np.ndarray | None = None,
    w_nb: float = 1.0,
    w_bh: float = 0.5,
) -> dict[tuple[int, int], float]:
    """Aggregate weighted mode-edge gradients over Tanner edges."""
    eigvals = np.asarray(eigenvalues, dtype=np.float64)
    eigvecs = np.asarray(eigenvectors, dtype=np.float64)
    iprs = np.asarray(ipr_scores, dtype=np.float64)

    if eigvecs.ndim == 1:
        eigvecs = eigvecs[:, None]

    sorted_adj = sorted(adj_list)

    grad: dict[tuple[int, int], float] = {}
    for u, w in sorted_adj:
        key = (u, w) if u <= w else (w, u)
        grad[key] = 0.0

    k_modes = min(eigvals.shape[0], eigvecs.shape[1], iprs.shape[0])
    for k in range(k_modes):
        sigma = compute_severity(float(eigvals[k]), float(iprs[k]))
        if sigma == 0.0:
            continue
        v = eigvecs[:, k]
        for u, w in sorted_adj:
            key = (u, w) if u <= w else (w, u)
            grad[key] += float(w_nb) * sigma * float(-r * v[u] * v[w])

    if dual_operator and bh_eigenvectors is not None:
        bh_vecs = np.asarray(bh_eigenvectors, dtype=np.float64)
        if bh_vecs.ndim == 1:
            bh_vecs = bh_vecs[:, None]
        k_bh = bh_vecs.shape[1]
        for k in range(k_bh):
            vb = bh_vecs[:, k]
            for u, w in sorted_adj:
                key = (u, w) if u <= w else (w, u)
                grad[key] += float(w_bh) * float(-vb[u] * vb[w])

    return {key: float(grad[key]) for key in sorted(grad)}
