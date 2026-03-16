"""Stable public analysis API exports and convenience wrappers."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import scipy.sparse

from .spectral_frustration import (
    spectral_frustration_count,
    SpectralFrustrationResult,
    SpectralFrustrationAnalyzer,
)

_MODULES = [
    "bethe_hessian",
    "localization_metrics",
    "defect_catalog",
    "trapping_set_classifier",
    "subgraph_extractor",
    "nb_perturbation_scorer",
    "ihara_bass_gradient",
    "defect_scheduler",
    "trapping_sets",
    "bp_residuals",
    "absorbing_sets",
    "cycle_topology",
    "residual_clusters",
    "nonbacktracking_flow",
    "constraint_tension",
    "eigenvector_localization",
    "flow_alignment",
    "nb_instability_gradient",
    "nb_eigenmode_flow",
    "basin_switch_detector",
    "basin_diagnostics",
    "bethe_hessian_fast",
    "spectral_frustration",
    "spectral_entropy",
    "trap_memory",
    "spectral_landscape_memory",
    "landscape_metrics",
    "landscape_io",
    "exploration_state",
    "exploration_metrics",
    "spectral_basins",
    "basin_transitions",
    "basin_statistics",
    "basin_map_export",
    "spectral_phase_space",
    "spectral_phase_diagram_3d",
    "spectral_phase_boundaries",
    "spectral_phase_diagram",
    "hypothesis_ranking",
    "hypothesis_generator",
    "discovery_archive_analyzer",
]

for mod in _MODULES:
    package_name = __name__.rsplit(".", 1)[0]
    module = importlib.import_module(f".{mod}", package_name)
    globals().update(
        {k: v for k, v in module.__dict__.items() if not k.startswith("_")}
    )


def compute_nb_spectrum(
    H: np.ndarray | scipy.sparse.spmatrix,
    *,
    power_iterations: int = 50,
    precision: int = 12,
) -> dict[str, Any]:
    """Public wrapper for deterministic NB first-order spectrum computation."""
    scorer = NBPerturbationScorer(power_iterations=power_iterations, precision=precision)
    return scorer.compute_nb_spectrum(H)


def compute_bh_spectrum(
    H: np.ndarray | scipy.sparse.spmatrix,
    *,
    r: float | None = None,
    num_modes: int = 20,
) -> dict[str, Any]:
    """Public wrapper for deterministic Bethe-Hessian matrix and unstable modes."""
    from src.qec.analysis.eigenmode_mutation import build_bethe_hessian, extract_unstable_modes

    B, r_eff = build_bethe_hessian(H, r=r)
    modes = extract_unstable_modes(B, num_modes=num_modes)
    return {
        "bethe_hessian": B,
        "r": float(r_eff),
        "modes": modes,
    }


def enumerate_candidate_swaps(
    H: np.ndarray | scipy.sparse.spmatrix,
    hot_edges: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    """Public deterministic candidate 2x2 swap enumerator used by experiments."""
    H_new = np.asarray(scipy.sparse.csr_matrix(H, dtype=np.float64).toarray(), dtype=np.float64)
    m, n = H_new.shape
    check_neighbors: dict[int, set[int]] = {ci: set() for ci in range(m)}
    var_neighbors: dict[int, set[int]] = {vi: set() for vi in range(n)}
    coords = np.argwhere(H_new != 0)
    for ci, vi in coords:
        cii = int(ci)
        vii = int(vi)
        check_neighbors[cii].add(vii)
        var_neighbors[vii].add(cii)

    swaps: list[tuple[int, int, int, int]] = []
    for ci, vi in hot_edges:
        if H_new[ci, vi] == 0:
            continue
        if len(check_neighbors[ci]) <= 1 or len(var_neighbors[vi]) <= 1:
            continue

        for vj in range(H_new.shape[1]):
            if vj == vi or H_new[ci, vj] != 0:
                continue
            for cj in sorted(var_neighbors[vj]):
                if cj == ci:
                    continue
                if vi in check_neighbors[cj]:
                    continue
                if len(check_neighbors[cj]) <= 1:
                    continue
                swaps.append((ci, vi, cj, vj))
    return swaps


__all__ = [k for k in globals() if not k.startswith("_")]
