"""v14.0.0 — Eigenmode-guided Tanner-graph spectral descent loop."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from src.qec.analysis.defect_scheduler import DefectScheduler
from src.qec.analysis.eigenmode_mutation import (
    adjacency_list_from_H,
    build_bethe_hessian,
    extract_unstable_modes,
)
from src.qec.analysis.ihara_bass_gradient import compute_ihara_bass_gradient
from src.qec.discovery.eigenmode_guided_swap import find_best_swap


def _project_gradient_to_matrix_edges(
    edge_gradient: dict[tuple[int, int], float],
    m: int,
) -> dict[tuple[int, int], float]:
    projected: dict[tuple[int, int], float] = {}
    for (u, v), score in sorted(edge_gradient.items()):
        if u < m <= v:
            projected[(int(u), int(v - m))] = float(score)
        elif v < m <= u:
            projected[(int(v), int(u - m))] = float(score)
    return {key: projected[key] for key in sorted(projected)}


def _apply_swap(H_csr: sp.csr_matrix, swap: dict) -> sp.csr_matrix:
    H_lil = H_csr.tolil(copy=True)
    for ci, vi in swap["remove"]:
        H_lil[int(ci), int(vi)] = 0.0
    for ci, vi in swap["add"]:
        H_lil[int(ci), int(vi)] = 1.0
    return H_lil.tocsr()


def spectral_descent(
    H: np.ndarray | sp.spmatrix,
    max_iter: int = 200,
    severity_threshold: float = 1e-6,
    scheduler: str = "aggregate",
    *,
    dual_operator: bool = False,
    w_nb: float = 1.0,
    w_bh: float = 0.5,
) -> sp.csr_matrix:
    """Deterministically optimize Tanner topology via spectral descent."""
    H_curr = sp.csr_matrix(H, dtype=np.float64)
    m, _ = H_curr.shape
    mode_scheduler = DefectScheduler(
        strategy=scheduler,
        dual_operator_enabled=bool(dual_operator),
    )

    for _ in range(int(max_iter)):
        B, r = build_bethe_hessian(H_curr)
        modes = extract_unstable_modes(B, num_modes=20)
        if not modes:
            break
        max_severity = float(max(mode["severity"] for mode in modes))
        if max_severity <= float(severity_threshold):
            break

        selected = mode_scheduler.select_modes(modes, bh_modes=modes if dual_operator else None)
        if not selected:
            break

        eigvals = np.asarray([mode["eigenvalue"] for mode in selected], dtype=np.float64)
        eigvecs = np.column_stack([np.asarray(mode["eigenvector"], dtype=np.float64) for mode in selected])
        iprs = np.asarray([mode["ipr"] for mode in selected], dtype=np.float64)

        adj_list = adjacency_list_from_H(H_curr)
        G_edge = compute_ihara_bass_gradient(eigvals, eigvecs, iprs, adj_list, r)
        if dual_operator:
            G_edge = compute_ihara_bass_gradient(
                eigvals,
                eigvecs,
                iprs,
                adj_list,
                r,
                dual_operator=True,
                bh_eigenvectors=eigvecs,
                w_nb=float(w_nb),
                w_bh=float(w_bh),
            )
        G = _project_gradient_to_matrix_edges(G_edge, m)

        swap = find_best_swap(H_curr, G, selected)
        if swap is None:
            break

        H_next = _apply_swap(H_curr, swap)
        B_next, _ = build_bethe_hessian(H_next, r=r)
        modes_next = extract_unstable_modes(B_next, num_modes=20)
        next_severity = float(max((mode["severity"] for mode in modes_next), default=0.0))
        if next_severity > max_severity:
            break
        H_curr = H_next

    return H_curr
