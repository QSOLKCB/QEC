"""v14.0.0 — Eigenmode diagnostics for deterministic Tanner mutation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def adjacency_list_from_H(H: np.ndarray | sp.spmatrix) -> list[tuple[int, int]]:
    """Return deterministic Tanner adjacency edges as node-id pairs.

    Node ids use checks first [0..m-1], then variables [m..m+n-1].
    """
    H_csr = sp.csr_matrix(H, dtype=np.float64)
    m, n = H_csr.shape
    coo = H_csr.tocoo()
    edges = [(int(ci), int(m + vi)) for ci, vi in zip(coo.row, coo.col)]
    edges.sort()
    return edges


def build_bethe_hessian(
    H: np.ndarray | sp.spmatrix,
    r: float | None = None,
) -> tuple[sp.csr_matrix, float]:
    """Build sparse Bethe-Hessian B(r) = (r^2-1)I - rA + D."""
    H_csr = sp.csr_matrix(H, dtype=np.float64)
    m, n = H_csr.shape

    A_upper = sp.hstack([sp.csr_matrix((m, m), dtype=np.float64), H_csr], format="csr")
    A_lower = sp.hstack([H_csr.transpose().tocsr(), sp.csr_matrix((n, n), dtype=np.float64)], format="csr")
    A = sp.vstack([A_upper, A_lower], format="csr")

    deg = np.asarray(A.sum(axis=1)).reshape(-1).astype(np.float64)
    d_avg = float(deg.mean()) if deg.size else 0.0
    if r is None:
        r = math.sqrt(max(d_avg - 1.0, 0.0))

    size = m + n
    I = sp.identity(size, dtype=np.float64, format="csr")
    D = sp.diags(deg, offsets=0, dtype=np.float64, format="csr")
    B = ((r * r) - 1.0) * I - (r * A) + D
    return B.tocsr(), float(r)


def extract_unstable_modes(
    B: sp.spmatrix,
    num_modes: int = 20,
) -> list[dict[str, Any]]:
    """Extract negative-eigenvalue Bethe-Hessian modes sorted by severity."""
    B_csr = sp.csr_matrix(B, dtype=np.float64)
    n = B_csr.shape[0]
    if n == 0:
        return []

    k = min(max(int(num_modes), 1), max(n - 1, 1))
    if n <= 2:
        vals, vecs = np.linalg.eigh(B_csr.toarray())
    else:
        vals, vecs = eigsh(
            B_csr,
            k=k,
            which="SA",
            v0=np.ones(n, dtype=np.float64),
            tol=0.0,
        )

    vals = np.asarray(vals, dtype=np.float64)
    vecs = np.asarray(vecs, dtype=np.float64)
    if vecs.ndim == 1:
        vecs = vecs[:, None]

    eigen_pairs = sorted(
        (
            (float(vals[idx]), vecs[:, idx].astype(np.float64, copy=False), int(idx))
            for idx in range(vals.shape[0])
        ),
        key=lambda item: item[0],
    )

    modes: list[dict[str, Any]] = []
    for idx, (lam, v, source_idx) in enumerate(eigen_pairs):
        if lam >= 0.0:
            continue
        abs_v = np.abs(v)
        ipr = float(np.sum(v ** 4))
        participation_entropy = float(-np.sum((v ** 2) * np.log((v ** 2) + 1e-300)))
        severity = float(abs(lam) * ipr)
        cutoff = 0.5 * float(abs_v.max()) if abs_v.size else 0.0
        support_nodes = tuple(int(i) for i in np.where(abs_v >= cutoff)[0].tolist())
        modes.append(
            {
                "eigenvalue": lam,
                "eigenvector": v,
                "ipr": ipr,
                "participation_entropy": participation_entropy,
                "severity": severity,
                "support_nodes": support_nodes,
                "mode_index": int(source_idx),
                "eigen_rank": int(idx),
            },
        )

    modes.sort(
        key=lambda mode: (
            -float(mode["severity"]),
            float(mode["eigenvalue"]),
            int(mode["mode_index"]),
        ),
    )
    return modes
