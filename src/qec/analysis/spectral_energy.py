"""Deterministic Bethe-Hessian spectral energy helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from qec.analysis.eigenmode_mutation import build_bethe_hessian


@dataclass(frozen=True)
class SpectralEnergyResult:
    """Container for spectral energy diagnostics."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    energy: float
    r: float


def compute_bethe_hessian_spectrum(
    H: np.ndarray | sp.spmatrix,
    *,
    num_eigenvalues: int = 20,
    r: float | None = None,
) -> SpectralEnergyResult:
    """Compute deterministic low-end Bethe-Hessian spectrum and energy."""
    B, r_eff = build_bethe_hessian(H, r=r)
    n = int(B.shape[0])
    if n == 0:
        return SpectralEnergyResult(
            eigenvalues=np.zeros((0,), dtype=np.float64),
            eigenvectors=np.zeros((0, 0), dtype=np.float64),
            energy=0.0,
            r=float(r_eff),
        )

    k = min(max(int(num_eigenvalues), 1), n)
    if n <= 2 or k == n:
        vals, vecs = np.linalg.eigh(B.toarray())
    else:
        vals, vecs = eigsh(
            B.tocsr(),
            k=k,
            which="SA",
            v0=np.ones(n, dtype=np.float64),
            tol=0.0,
            maxiter=max(5 * n, 100),
        )

    vals = np.asarray(vals, dtype=np.float64)
    vecs = np.asarray(vecs, dtype=np.float64)
    if vecs.ndim == 1:
        vecs = vecs[:, None]

    order = np.lexsort((np.arange(vals.size, dtype=np.int64), vals))
    vals = vals[order]
    vecs = vecs[:, order]
    negative = vals[vals < 0.0]
    energy = float(np.sum(negative * negative))
    return SpectralEnergyResult(
        eigenvalues=vals,
        eigenvectors=vecs,
        energy=energy,
        r=float(r_eff),
    )


def estimate_bethe_hessian_r(H: np.ndarray | sp.spmatrix) -> float:
    """Estimate Bethe-Hessian ``r`` via ``sqrt(<d(d-1)> / <d>)``."""
    H_csr = sp.csr_matrix(H, dtype=np.float64)
    m, n = H_csr.shape
    if m == 0 and n == 0:
        return 1.0

    top = sp.hstack([sp.csr_matrix((m, m), dtype=np.float64), H_csr], format="csr")
    bottom = sp.hstack([H_csr.transpose().tocsr(), sp.csr_matrix((n, n), dtype=np.float64)], format="csr")
    adjacency = sp.vstack([top, bottom], format="csr")
    degrees = np.asarray(adjacency.sum(axis=1), dtype=np.float64).ravel()
    if degrees.size == 0:
        return 1.0
    mean_degree = float(np.mean(degrees))
    if mean_degree <= 0.0:
        return 1.0
    numerator = float(np.mean(degrees * np.maximum(degrees - 1.0, 0.0)))
    ratio = numerator / mean_degree
    if ratio <= 0.0:
        return 1.0
    return float(np.sqrt(ratio))
