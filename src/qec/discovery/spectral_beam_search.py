"""v14.5.0 — Deterministic adaptive spectral beam planning for 2x2 swaps."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from src.qec.analysis.basin_depth import adaptive_beam_width, basin_depth_from_diagnostics
from src.qec.discovery.eigenmode_guided_swap import score_2x2_swap


_DEFAULT_MAX_CANDIDATES = 1000


def _canonical_swap(i1: int, j1: int, i2: int, j2: int) -> tuple[int, int, int, int]:
    return (i1, j1, i2, j2) if (i1, j1, i2, j2) <= (i2, j2, i1, j1) else (i2, j2, i1, j1)


def _extract_edges(H_csr: sp.csr_matrix) -> list[tuple[int, int]]:
    coo = H_csr.tocoo()
    return sorted((int(ci), int(vi)) for ci, vi in zip(coo.row, coo.col))


def _rank_hot_edges(
    edges: list[tuple[int, int]],
    G: dict[tuple[int, int], float],
    limit: int,
) -> list[tuple[int, int]]:
    ranked = sorted(edges, key=lambda e: (-abs(float(G.get(e, 0.0))), e[0], e[1]))
    return ranked[: max(1, int(limit))]


def _enumerate_candidates(
    H_lil: sp.lil_matrix,
    hot: list[tuple[int, int]],
    G: dict[tuple[int, int], float],
    max_candidates: int,
) -> list[tuple[float, tuple[int, int, int, int]]]:
    _, n = H_lil.shape
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
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
            ordered = _canonical_swap(i1, j1, i2, j2)
            delta = score_2x2_swap(G, ordered[0], ordered[1], ordered[2], ordered[3])
            candidates.append((float(delta), ordered))
            count += 1
            if count >= int(max_candidates):
                break
        if count >= int(max_candidates):
            break
    candidates.sort(key=lambda c: (-c[0], c[1][0], c[1][1], c[1][2], c[1][3]))
    return candidates


def _apply_swap_lil(H_lil: sp.lil_matrix, ordered: tuple[int, int, int, int]) -> sp.lil_matrix:
    i1, j1, i2, j2 = ordered
    H_next = H_lil.copy()
    H_next[i1, j1] = 0.0
    H_next[i2, j2] = 0.0
    H_next[i1, j2] = 1.0
    H_next[i2, j1] = 1.0
    return H_next


def find_best_swap_with_adaptive_beam(
    H: np.ndarray | sp.spmatrix,
    G: dict[tuple[int, int], float],
    *,
    top_k_edges: int = 50,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    basin_diagnostics: dict[str, float] | None = None,
    beam_min: int = 3,
    beam_max: int = 10,
    depth_scale: float = 3.0,
    w1: float = 1.0,
    w2: float = 1.0,
    w3: float = 0.5,
    w4: float = 0.5,
) -> tuple[dict[str, Any] | None, dict[str, float | int]]:
    """Run deterministic two-step beam planning with adaptive beam width."""
    H_csr = sp.csr_matrix(H, dtype=np.float64)
    edges = _extract_edges(H_csr)
    if len(edges) < 2:
        return None, {"basin_depth": 0.0, "beam_width": int(beam_min), "planner_depth": 2}

    basin_depth = basin_depth_from_diagnostics(basin_diagnostics, w1=w1, w2=w2, w3=w3, w4=w4)
    beam_width = adaptive_beam_width(basin_depth, beam_min=beam_min, beam_max=beam_max, depth_scale=depth_scale)
    second_step_limit = beam_width

    hot = _rank_hot_edges(edges, G, top_k_edges)
    H_lil = H_csr.tolil(copy=False)
    first_candidates = _enumerate_candidates(H_lil, hot, G, max_candidates=max_candidates)
    if not first_candidates:
        return None, {"basin_depth": basin_depth, "beam_width": beam_width, "planner_depth": 2}

    first_beam = first_candidates[:beam_width]
    best_plan: tuple[float, tuple[int, int, int, int], tuple[int, int, int, int] | None] | None = None

    for first_delta, first_swap in first_beam:
        H_step1 = _apply_swap_lil(H_lil, first_swap)
        edges2 = _extract_edges(H_step1.tocsr())
        hot2 = _rank_hot_edges(edges2, G, top_k_edges)
        second_candidates = _enumerate_candidates(H_step1, hot2, G, max_candidates=max_candidates)
        if second_candidates:
            second_candidates = second_candidates[:second_step_limit]
            second_delta, second_swap = second_candidates[0]
            total = float(first_delta + second_delta)
        else:
            second_swap = None
            total = float(first_delta)

        plan = (total, first_swap, second_swap)
        if best_plan is None:
            best_plan = plan
            continue
        if plan[0] > best_plan[0]:
            best_plan = plan
            continue
        if np.isclose(plan[0], best_plan[0], atol=0.0, rtol=0.0):
            if plan[1] < best_plan[1] or (plan[1] == best_plan[1] and (plan[2] or (10**9,)*4) < (best_plan[2] or (10**9,)*4)):
                best_plan = plan

    if best_plan is None or best_plan[0] <= 0.0:
        return None, {"basin_depth": basin_depth, "beam_width": beam_width, "planner_depth": 2}

    i1, j1, i2, j2 = best_plan[1]
    swap = {
        "remove": ((i1, j1), (i2, j2)),
        "add": ((i1, j2), (i2, j1)),
        "delta": float(score_2x2_swap(G, i1, j1, i2, j2)),
    }
    return swap, {"basin_depth": basin_depth, "beam_width": beam_width, "planner_depth": 2}
