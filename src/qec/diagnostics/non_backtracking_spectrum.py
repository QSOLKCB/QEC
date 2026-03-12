"""
v6.0.0 — Non-Backtracking (Hashimoto) Spectrum Diagnostics.

Computes eigenvalues of the non-backtracking matrix derived from the
Tanner graph of a parity-check matrix.  The non-backtracking matrix
captures the structure of message passing on the graph without
immediate reversal, making it a more faithful spectral proxy for
BP dynamics than the ordinary adjacency spectrum.

Operates purely on the parity-check matrix — does not run BP decoding.
Does not modify decoder internals.  Fully deterministic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def _build_sparse_nb_matrix(
    H: np.ndarray,
) -> tuple[scipy.sparse.csr_matrix, list[tuple[int, int]]]:
    """Build the non-backtracking matrix as a sparse CSR matrix.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix, shape (m, n).

    Returns
    -------
    B : scipy.sparse.csr_matrix
        Sparse NB matrix of shape (2|E|, 2|E|).
    directed_edges : list[tuple[int, int]]
        Directed edge list.
    """
    m, n = H.shape

    # Build directed edge list using nonzero entries.
    rows, cols = np.where(H != 0)
    directed_edges: list[tuple[int, int]] = []
    for ci, vi in zip(rows, cols):
        directed_edges.append((int(vi), n + int(ci)))
        directed_edges.append((n + int(ci), int(vi)))

    num_directed = len(directed_edges)
    if num_directed == 0:
        return scipy.sparse.csr_matrix((0, 0), dtype=np.float64), []

    # Build outgoing adjacency.
    outgoing: dict[int, list[int]] = {}
    for idx, (_u, v) in enumerate(directed_edges):
        outgoing.setdefault(v, []).append(idx)

    # Build sparse NB matrix via COO triplets.
    nb_rows: list[int] = []
    nb_cols: list[int] = []
    for idx_uv, (u, v) in enumerate(directed_edges):
        for idx_vw in outgoing.get(v, []):
            _, w = directed_edges[idx_vw]
            if w != u:
                nb_rows.append(idx_uv)
                nb_cols.append(idx_vw)

    data = np.ones(len(nb_rows), dtype=np.float64)
    B = scipy.sparse.csr_matrix(
        (data, (nb_rows, nb_cols)),
        shape=(num_directed, num_directed),
    )
    return B, directed_edges


def compute_non_backtracking_spectrum(
    parity_check_matrix: np.ndarray,
) -> dict[str, Any]:
    """Compute eigenvalues of the non-backtracking (Hashimoto) matrix.

    The non-backtracking matrix B is defined on directed edges of the
    Tanner graph.  For directed edges (u -> v) and (v -> w), we have
    B_{(u->v), (v->w)} = 1 if w != u (no immediate backtrack).

    Parameters
    ----------
    parity_check_matrix : np.ndarray
        Binary parity-check matrix H with shape (m, n).

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary with keys:
        - ``nb_eigenvalues``: list of eigenvalue magnitudes (sorted
          descending by magnitude, real and imaginary parts as pairs)
        - ``spectral_radius``: float, largest eigenvalue magnitude
        - ``num_eigenvalues``: int, total number of eigenvalues
    """
    H = np.asarray(parity_check_matrix, dtype=np.float64)

    B_sparse, directed_edges = _build_sparse_nb_matrix(H)
    num_directed = len(directed_edges)

    if num_directed == 0:
        return {
            "nb_eigenvalues": [],
            "spectral_radius": 0.0,
            "num_eigenvalues": 0,
        }

    # ── Compute eigenvalues ───────────────────────────────────────
    # Use dense eigensolver for determinism.  ARPACK eigs can produce
    # non-deterministic eigenvalues at the ULP level even with fixed v0.
    # Sparse construction (above) keeps build cost at O(|E|).
    # Dense eigensolver is used up to 512 directed edges; above that,
    # sparse with dense fallback.
    k = min(6, num_directed - 2)
    if k < 1 or num_directed <= 512:
        eigenvalues = np.linalg.eigvals(B_sparse.toarray())
    else:
        try:
            v0 = np.ones(num_directed, dtype=np.float64)
            eigenvalues = scipy.sparse.linalg.eigs(
                B_sparse,
                k=k,
                which="LM",
                v0=v0,
                return_eigenvectors=False,
            )
        except (scipy.sparse.linalg.ArpackNoConvergence, RuntimeError):
            eigenvalues = np.linalg.eigvals(B_sparse.toarray())

    # Compute magnitudes.
    magnitudes = np.abs(eigenvalues)

    # Sort by magnitude descending, with deterministic tie-breaking
    # by real part descending, then imaginary part descending.
    sort_keys = np.lexsort((
        -eigenvalues.imag,
        -eigenvalues.real,
        -magnitudes,
    ))
    eigenvalues = eigenvalues[sort_keys]
    magnitudes = magnitudes[sort_keys]

    spectral_radius = float(magnitudes[0])

    # Convert eigenvalues to JSON-safe format: list of [real, imag] pairs.
    nb_eigenvalues: list[list[float]] = []
    for ev in eigenvalues:
        nb_eigenvalues.append([float(ev.real), float(ev.imag)])

    return {
        "nb_eigenvalues": nb_eigenvalues,
        "spectral_radius": spectral_radius,
        "num_eigenvalues": num_directed,
    }
