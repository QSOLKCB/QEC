"""
v8.1.0 — Spectral Metrics Aggregator.

Aggregates all spectral stability diagnostics into a single dictionary.
Reuses existing NB spectrum computation where possible to avoid
redundant eigenpair calculations.

Layer 3 — Diagnostics.
Does not import or modify the decoder (Layer 1).
Fully deterministic: no randomness, no global state, no input mutation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import eigsh, eigs

from src.qec.diagnostics._spectral_utils import (
    compute_ipr,
    compute_nb_dominant_eigenpair,
)
from src.qec.diagnostics.bethe_hessian_margin import compute_bethe_hessian_margin
from src.qec.diagnostics.cycle_space_density import compute_cycle_space_density
from src.qec.diagnostics.spectral_nb import _TannerGraph, _compute_eeec, _compute_sis

_ROUND = 12


# -----------------------------------------------------------
# Safe eigenvalue wrappers (v11.2.1 Spectral Stability Patch)
# -----------------------------------------------------------


def safe_eigsh(A, k=1, which="LM"):
    """Compute eigenvalues of a symmetric matrix with ARPACK fallback.

    Attempts ``scipy.sparse.linalg.eigsh`` first.  When the matrix is
    too small for the requested *k* or ARPACK raises any exception, the
    function falls back to a dense eigenvalue computation via
    ``np.linalg.eigvalsh``.

    Parameters
    ----------
    A : sparse matrix or ndarray
        Real symmetric matrix.
    k : int, optional
        Number of eigenvalues requested (default 1).
    which : str, optional
        Which eigenvalues to compute (default ``"LM"``).

    Returns
    -------
    np.ndarray
        1-D array of *k* eigenvalues (real), sorted according to *which*.
    """
    n = A.shape[0]

    # Dense fallback for small matrices where ARPACK cannot operate
    if n < k + 2:
        return _dense_eigvalsh_select(A, k, which)

    try:
        vals, _ = eigsh(A, k=k, which=which)
        return np.sort(np.real(vals))[::-1] if which == "LM" else np.sort(np.real(vals))
    except Exception:
        return _dense_eigvalsh_select(A, k, which)


def _dense_eigvalsh_select(A, k, which):
    """Dense symmetric eigenvalue fallback."""
    dense = A.toarray() if scipy.sparse.issparse(A) else np.asarray(A, dtype=float)
    all_vals = np.linalg.eigvalsh(dense)

    if which == "LM":
        # Largest magnitude — sort by absolute value descending
        order = np.argsort(np.abs(all_vals))[::-1]
    elif which == "SM":
        order = np.argsort(np.abs(all_vals))
    elif which == "LA":
        order = np.argsort(all_vals)[::-1]
    elif which == "SA":
        order = np.argsort(all_vals)
    else:
        order = np.argsort(np.abs(all_vals))[::-1]

    k = min(k, len(all_vals))
    return all_vals[order[:k]]


def safe_eigs(A, k=1, which="LM", tol=0):
    """Compute eigenvalues of a general matrix with ARPACK fallback.

    Attempts ``scipy.sparse.linalg.eigs`` first.  Falls back to dense
    ``np.linalg.eigvals`` on failure or when the matrix is too small.

    Parameters
    ----------
    A : LinearOperator, sparse matrix, or ndarray
        Square matrix or operator.
    k : int, optional
        Number of eigenvalues requested (default 1).
    which : str, optional
        Which eigenvalues to compute (default ``"LM"``).
    tol : float, optional
        Tolerance for ARPACK convergence (default 0).

    Returns
    -------
    np.ndarray
        1-D array of *k* eigenvalues (complex).
    np.ndarray
        2-D array of corresponding eigenvectors, shape ``(n, k)``.
    """
    n = A.shape[0]

    if n < k + 2:
        return _dense_eig_select(A, k, which)

    try:
        vals, vecs = eigs(A, k=k, which=which, tol=tol)
        return vals, vecs
    except Exception:
        return _dense_eig_select(A, k, which)


def _dense_eig_select(A, k, which):
    """Dense non-symmetric eigenvalue fallback."""
    if hasattr(A, "toarray"):
        dense = A.toarray()
    elif hasattr(A, "matmul") or hasattr(A, "matvec"):
        # LinearOperator — materialise via identity probes
        n = A.shape[0]
        dense = np.zeros((n, n), dtype=float)
        for i in range(n):
            e = np.zeros(n)
            e[i] = 1.0
            dense[:, i] = A.matvec(e)
    else:
        dense = np.asarray(A, dtype=float)

    all_vals = np.linalg.eigvals(dense)
    all_vecs = np.linalg.eig(dense)[1]

    if which == "LM":
        order = np.argsort(np.abs(all_vals))[::-1]
    elif which == "SM":
        order = np.argsort(np.abs(all_vals))
    elif which == "LR":
        order = np.argsort(np.real(all_vals))[::-1]
    elif which == "SR":
        order = np.argsort(np.real(all_vals))
    else:
        order = np.argsort(np.abs(all_vals))[::-1]

    k = min(k, len(all_vals))
    return all_vals[order[:k]], all_vecs[:, order[:k]]


def compute_spectral_metrics(H: np.ndarray) -> dict[str, Any]:
    """Compute all spectral stability metrics for a parity-check matrix.

    Computes the dominant NB eigenpair once and derives all metrics
    from it, avoiding redundant spectral decompositions.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - ``spectral_radius`` : float
        - ``entropy`` : float
        - ``spectral_gap`` : float
        - ``bethe_margin`` : float
        - ``support_dimension`` : float
        - ``curvature`` : float
        - ``cycle_density`` : float
        - ``sis`` : float
    """
    H_arr = np.asarray(H, dtype=np.float64)
    graph = _TannerGraph(H_arr)

    # ── Single eigenpair computation ───────────────────────────────
    spectral_radius, eigenvector, directed_edges = (
        compute_nb_dominant_eigenpair(graph)
    )

    # Normalize and canonicalize sign
    norm = np.linalg.norm(eigenvector)
    if norm > 0:
        eigenvector = eigenvector / norm
    max_idx = int(np.argmax(np.abs(eigenvector)))
    if eigenvector[max_idx] < 0:
        eigenvector = -eigenvector

    # ── Edge energy distribution ───────────────────────────────────
    edge_energy = np.abs(eigenvector) ** 2
    total_energy = edge_energy.sum()

    # ── Spectral entropy ──────────────────────────────────────────
    if total_energy > 0:
        p = edge_energy / total_energy
        mask = p > 0
        entropy = float(-np.sum(p[mask] * np.log(p[mask])))
    else:
        entropy = 0.0

    # ── Effective support dimension ───────────────────────────────
    support_dimension = float(np.exp(entropy))

    # ── Spectral curvature ────────────────────────────────────────
    nonzero_energy = edge_energy[edge_energy > 0]
    if len(nonzero_energy) >= 2:
        log_energy = np.log(nonzero_energy)
        curvature = float(np.var(log_energy))
    else:
        curvature = 0.0

    # ── IPR, EEEC, SIS ───────────────────────────────────────────
    ipr = compute_ipr(eigenvector)
    eeec = _compute_eeec(edge_energy)
    sis = _compute_sis(spectral_radius, ipr, eeec)

    # ── Spectral gap (top-2 eigenvalues) ──────────────────────────
    from src.qec.diagnostics._spectral_utils import build_nb_operator

    op, _ = build_nb_operator(graph)
    n_edges = len(directed_edges)

    if n_edges >= 3:
        k = min(2, n_edges - 1)
        vals, _ = safe_eigs(op, k=k, which="LM", tol=1e-6)
        magnitudes = np.sort(np.abs(vals))[::-1]
        spectral_gap = float(magnitudes[0] - magnitudes[1]) if len(magnitudes) >= 2 else 0.0
    else:
        spectral_gap = 0.0

    # ── Bethe Hessian margin ──────────────────────────────────────
    bethe_margin = compute_bethe_hessian_margin(H_arr)

    # ── Cycle space density ───────────────────────────────────────
    cycle_density = compute_cycle_space_density(H_arr)

    return {
        "spectral_radius": round(float(spectral_radius), _ROUND),
        "entropy": round(entropy, _ROUND),
        "spectral_gap": round(spectral_gap, _ROUND),
        "bethe_margin": round(bethe_margin, _ROUND),
        "support_dimension": round(support_dimension, _ROUND),
        "curvature": round(curvature, _ROUND),
        "cycle_density": round(cycle_density, _ROUND),
        "sis": round(sis, _ROUND),
    }
