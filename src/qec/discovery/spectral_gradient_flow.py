"""v14.1.0 — Deterministic spectral gradient flow for Tanner-graph discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from qec.analysis.eigenmode_subspace import EigenmodeCluster, cluster_eigenmodes
from qec.analysis.spectral_energy import compute_bethe_hessian_spectrum, estimate_bethe_hessian_r
from qec.fitness.spectral_metrics import compute_girth_spectrum


@dataclass(frozen=True)
class SpectralGradientFlowConfig:
    """Configuration for deterministic spectral gradient flow."""

    num_eigenvalues: int = 20
    max_steps: int = 1000
    energy_tol: float = 1e-8
    swap_budget: int = 500
    top_edge_fraction: float = 0.3
    degen_base_epsilon: float = 0.01
    min_girth: int = 6
    r_update_interval: int = 10


def _sorted_edges(H_csr: sp.csr_matrix) -> list[tuple[int, int]]:
    coo = H_csr.tocoo()
    return sorted((int(ci), int(vi)) for ci, vi in zip(coo.row, coo.col))


def _compute_edge_gradient(
    clusters: list[EigenmodeCluster],
    m: int,
    n: int,
    r: float,
) -> dict[tuple[int, int], float]:
    gradient: dict[tuple[int, int], float] = {}
    for ci in range(m):
        for vi in range(n):
            node_v = m + vi
            value = 0.0
            for cluster in clusters:
                coeff = 2.0 * cluster.mean_eigenvalue * (-2.0 * r) * float(len(cluster.indices))
                value += coeff * float(cluster.projector[ci, node_v])
            gradient[(ci, vi)] = float(value)
    return gradient


def _estimate_swap_delta(
    clusters: list[EigenmodeCluster],
    m: int,
    r: float,
    swap: tuple[int, int, int, int],
) -> float:
    ci, vi, ck, vl = swap
    node_vi = m + vi
    node_vl = m + vl
    delta = 0.0
    for cluster in clusters:
        proj = cluster.projector
        inner = (
            float(proj[ci, node_vl])
            + float(proj[ck, node_vi])
            - float(proj[ci, node_vi])
            - float(proj[ck, node_vl])
        )
        delta += 2.0 * cluster.mean_eigenvalue * (-2.0 * r) * inner * float(len(cluster.indices))
    return float(delta)


def _apply_swap(H_csr: sp.csr_matrix, swap: tuple[int, int, int, int]) -> sp.csr_matrix:
    ci, vi, ck, vl = swap
    out = H_csr.tolil(copy=True)
    out[ci, vi] = 0.0
    out[ck, vl] = 0.0
    out[ci, vl] = 1.0
    out[ck, vi] = 1.0
    return out.tocsr()


def _enumerate_candidate_swaps(
    H_csr: sp.csr_matrix,
    edge_gradient: dict[tuple[int, int], float],
    *,
    top_edge_fraction: float,
    swap_budget: int,
) -> list[tuple[int, int, int, int]]:
    if not (0.0 < float(top_edge_fraction) <= 1.0):
        raise ValueError(
            f"top_edge_fraction must be in the interval (0.0, 1.0], "
            f"got {top_edge_fraction!r}"
        )

    edges = _sorted_edges(H_csr)
    if len(edges) < 2:
        return []
    ranked = sorted(
        edges,
        key=lambda edge: (-abs(float(edge_gradient.get(edge, 0.0))), edge[0], edge[1]),
    )
    top_k = max(2, int(np.ceil(float(top_edge_fraction) * float(len(ranked)))))
    hot = ranked[:top_k]
    H_lil = H_csr.tolil(copy=False)

    candidates: list[tuple[int, int, int, int]] = []
    for idx_a in range(len(hot)):
        ci, vi = hot[idx_a]
        for idx_b in range(idx_a + 1, len(hot)):
            ck, vl = hot[idx_b]
            if ci == ck or vi == vl:
                continue
            if H_lil[ci, vl] != 0.0 or H_lil[ck, vi] != 0.0:
                continue
            swap = (ci, vi, ck, vl) if (ci, vi, ck, vl) <= (ck, vl, ci, vi) else (ck, vl, ci, vi)
            candidates.append(swap)
            if len(candidates) >= int(swap_budget):
                return candidates
    return candidates


def run_spectral_gradient_flow(
    H: np.ndarray | sp.spmatrix,
    *,
    config: SpectralGradientFlowConfig | None = None,
) -> tuple[sp.csr_matrix, list[float]]:
    """Run deterministic energy descent using Bethe-Hessian unstable modes."""
    cfg = config if config is not None else SpectralGradientFlowConfig()
    H_curr = sp.csr_matrix(H, dtype=np.float64)
    m, n = H_curr.shape

    energy_trace: list[float] = []
    r = estimate_bethe_hessian_r(H_curr)

    for step in range(int(cfg.max_steps)):
        if step > 0 and int(cfg.r_update_interval) > 0 and step % int(cfg.r_update_interval) == 0:
            r = estimate_bethe_hessian_r(H_curr)

        spectrum = compute_bethe_hessian_spectrum(
            H_curr,
            num_eigenvalues=int(cfg.num_eigenvalues),
            r=float(r),
        )
        energy = float(spectrum.energy)
        energy_trace.append(energy)
        if energy <= float(cfg.energy_tol):
            break

        negative_mask = spectrum.eigenvalues < 0.0
        if not np.any(negative_mask):
            break

        vals = spectrum.eigenvalues[negative_mask]
        vecs = spectrum.eigenvectors[:, negative_mask]
        clusters = cluster_eigenmodes(vals, vecs, base_epsilon=float(cfg.degen_base_epsilon))
        edge_gradient = _compute_edge_gradient(clusters, m, n, float(r))
        swaps = _enumerate_candidate_swaps(
            H_curr,
            edge_gradient,
            top_edge_fraction=float(cfg.top_edge_fraction),
            swap_budget=int(cfg.swap_budget),
        )
        if not swaps:
            break

        scored = sorted(
            (
                (
                    float(_estimate_swap_delta(clusters, m, float(r), swap)),
                    int(swap[0]),
                    int(swap[1]),
                    int(swap[2]),
                    int(swap[3]),
                ),
                swap,
            )
            for swap in swaps
        )
        # Only evaluate expensive spectrum/girth computations on a small
        # subset of the most promising swaps, as ranked by the cheap
        # _estimate_swap_delta score.
        top_k = min(len(scored), 32)

        best_swap: tuple[int, int, int, int] | None = None
        best_energy = energy
        current_girth = int(compute_girth_spectrum(H_curr.toarray())["girth"])

        for _, swap in scored[:top_k]:
            H_next = _apply_swap(H_curr, swap)
            girth = int(compute_girth_spectrum(H_next.toarray())["girth"])
            if girth < int(cfg.min_girth):
                continue
            next_spectrum = compute_bethe_hessian_spectrum(
                H_next,
                num_eigenvalues=int(cfg.num_eigenvalues),
                r=float(r),
            )
            next_energy = float(next_spectrum.energy)
            if next_energy + float(cfg.energy_tol) < best_energy and girth >= current_girth:
                best_energy = next_energy
                best_swap = swap
                break

        if best_swap is None:
            break

        H_curr = _apply_swap(H_curr, best_swap)

    return H_curr, energy_trace
