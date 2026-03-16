"""v14.0.0 — Deterministic 2x2 swap search from spectral edge gradients."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def score_2x2_swap(
    G: dict[tuple[int, int], float],
    i1: int,
    j1: int,
    i2: int,
    j2: int,
) -> float:
    """Return ΔL for swap remove(i1,j1),(i2,j2) add(i1,j2),(i2,j1)."""
    return float(
        G.get((i1, j1), 0.0)
        + G.get((i2, j2), 0.0)
        - G.get((i1, j2), 0.0)
        - G.get((i2, j1), 0.0)
    )


def find_best_swap(
    H: np.ndarray | sp.spmatrix,
    G: dict[tuple[int, int], float],
    top_k_edges: int = 50,
    max_candidates: int = 1000,
) -> dict | None:
    H_csr = sp.csr_matrix(H, dtype=np.float64)
    m, n = H_csr.shape

    coo = H_csr.tocoo()
    edges = [(int(ci), int(vi)) for ci, vi in zip(coo.row, coo.col)]
    if len(edges) < 2:
        return None

    ranked = sorted(
        edges,
        key=lambda e: (-abs(float(G.get(e, 0.0))), e[0], e[1]),
    )
    hot = ranked[: max(1, int(top_k_edges))]

    H_lil = H_csr.tolil(copy=False)
    best: tuple[float, tuple[int, int, int, int]] | None = None
    count = 0

    for a_idx in range(len(hot)):
        i1, j1 = hot[a_idx]
        for b_idx in range(a_idx + 1, len(hot)):
            i2, j2 = hot[b_idx]
            if i1 == i2 or j1 == j2:
                continue
            if j1 < 0 or j1 >= n or j2 < 0 or j2 >= n:
                continue
            if H_lil[i1, j2] != 0.0 or H_lil[i2, j1] != 0.0:
                continue

            ordered = (i1, j1, i2, j2) if (i1, j1, i2, j2) <= (i2, j2, i1, j1) else (i2, j2, i1, j1)
            delta = score_2x2_swap(G, ordered[0], ordered[1], ordered[2], ordered[3])
            candidate = (float(delta), ordered)
            if best is None or candidate[0] > best[0] or (
                np.isclose(candidate[0], best[0], atol=0.0, rtol=0.0)
                and candidate[1] < best[1]
            ):
                best = candidate

            count += 1
            if count >= int(max_candidates):
                break
        if count >= int(max_candidates):
            break

    if best is None or best[0] <= 0.0:
        return None

    i1, j1, i2, j2 = best[1]
    return {
        "remove": ((i1, j1), (i2, j2)),
        "add": ((i1, j2), (i2, j1)),
        "delta": float(best[0]),
    }
