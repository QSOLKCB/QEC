"""
Internal spectral utilities for Tanner graph analysis.

These utilities provide safe primitives for computing
non-backtracking spectral diagnostics without constructing
dense matrices.

This module must remain deterministic and memory-safe.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, LinearOperator, eigs


# -----------------------------------------------------------
# Directed edge construction
# -----------------------------------------------------------


def build_directed_edges(graph):
    """
    Build deterministic directed edge list.

    Returns
    -------
    edges : list[(u, v)]
        Directed edges in deterministic order.
    """

    edges = []

    for u in sorted(graph.nodes()):
        for v in sorted(graph.neighbors(u)):
            edges.append((u, v))

    return edges


# -----------------------------------------------------------
# Non-backtracking matvec
# -----------------------------------------------------------


def nb_matvec(graph, directed_edges, x):
    """
    Matrix-vector product for the non-backtracking operator.

    Parameters
    ----------
    graph
        Tanner graph
    directed_edges
        ordered list of directed edges
    x
        vector

    Returns
    -------
    y
        result of NB * x
    """

    edge_index = {e: i for i, e in enumerate(directed_edges)}

    y = np.zeros_like(x)

    for i, (u, v) in enumerate(directed_edges):

        for w in graph.neighbors(v):

            # enforce non-backtracking condition
            if w == u:
                continue

            j = edge_index.get((v, w))

            if j is not None:
                y[j] += x[i]

    return y


# -----------------------------------------------------------
# LinearOperator wrapper
# -----------------------------------------------------------


def build_nb_operator(graph):
    """
    Construct NB LinearOperator without dense matrix.

    Returns
    -------
    LinearOperator
    """

    directed_edges = build_directed_edges(graph)

    n = len(directed_edges)

    def matvec(x):
        return nb_matvec(graph, directed_edges, x)

    op = LinearOperator(
        shape=(n, n),
        matvec=matvec,
        dtype=float,
    )

    return op, directed_edges


# -----------------------------------------------------------
# Dominant eigenpair
# -----------------------------------------------------------


def compute_nb_dominant_eigenpair(graph, tol=1e-6):
    """
    Compute dominant eigenpair of NB operator.

    Returns
    -------
    spectral_radius
    eigenvector
    directed_edges
    """

    op, directed_edges = build_nb_operator(graph)
    n = op.shape[0]
    if n == 0:
        return 0.0, np.zeros(0, dtype=np.float64), directed_edges

    # Use deterministic initial vector for ARPACK reproducibility.
    v0 = np.ones(n, dtype=np.float64)
    try:
        vals, vecs = eigs(
            op,
            k=1,
            which="LR",
            tol=tol,
            v0=v0,
        )
    except (ArpackError, ArpackNoConvergence, ValueError, np.linalg.LinAlgError):
        if n > 512:
            raise

        dense_nb = op.matmat(np.eye(n, dtype=np.float64))
        vals_dense, vecs_dense = np.linalg.eig(dense_nb)
        idx = int(np.argmax(np.real(vals_dense)))
        vals = np.asarray([vals_dense[idx]], dtype=np.complex128)
        vecs = np.asarray(vecs_dense[:, [idx]], dtype=np.complex128)

    spectral_radius = np.real(vals[0])
    eigenvector = np.real(vecs[:, 0])

    # normalize for stability
    norm = np.linalg.norm(eigenvector)
    if norm > 0:
        eigenvector /= norm

    return spectral_radius, eigenvector, directed_edges


# -----------------------------------------------------------
# Inverse Participation Ratio
# -----------------------------------------------------------


def compute_ipr(v):
    """
    Compute inverse participation ratio.

    Measures localization of eigenvector.

    Returns
    -------
    float
    """

    v = np.abs(v)

    return np.sum(v**4) / (np.sum(v**2) ** 2)
