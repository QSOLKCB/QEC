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
    "spectral_geometry",
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
    "spectral_theory_dataset",
    "spectral_theory_models",
    "spectral_conjectures",
    "spectral_conjecture_validation",
    "spectral_counterexamples",
    "spectral_theory_memory",
    "spectral_ridges",
]

for mod in _MODULES:
    package_name = __name__.rsplit(".", 1)[0]
    module = importlib.import_module(f".{mod}", package_name)
    globals().update(
        {k: v for k, v in module.__dict__.items() if not k.startswith("_")}
    )


def build_theory_dataset(archive: dict[str, Any]) -> dict[str, Any]:
    """Public wrapper for deterministic spectral-theory dataset extraction."""
    from src.qec.analysis.spectral_theory_dataset import build_theory_dataset as _impl
    return _impl(archive)


def fit_theory_models(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Public wrapper for deterministic spectral-theory model fitting."""
    from src.qec.analysis.spectral_theory_models import fit_theory_models as _impl
    return _impl(dataset)


def generate_conjectures(fitted_models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Public wrapper for deterministic spectral conjecture generation."""
    from src.qec.analysis.spectral_conjectures import generate_conjectures as _impl
    return _impl(fitted_models)


def rank_conjectures(conjectures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Public wrapper for deterministic spectral conjecture ranking."""
    from src.qec.analysis.spectral_conjectures import rank_conjectures as _impl
    return _impl(conjectures)




def validate_conjectures(conjectures: list[dict[str, Any]], dataset: dict[str, Any], tolerance: float = 0.15) -> list[dict[str, Any]]:
    """Public wrapper for deterministic conjecture validation."""
    from src.qec.analysis.spectral_conjecture_validation import validate_conjectures as _impl
    return _impl(conjectures, dataset, tolerance=tolerance)


def extract_counterexamples(
    conjecture: dict[str, Any],
    dataset: dict[str, Any],
    error_threshold: float,
    max_counterexamples: int = 128,
) -> list[dict[str, Any]]:
    """Public wrapper for deterministic counterexample extraction."""
    from src.qec.analysis.spectral_counterexamples import extract_counterexamples as _impl
    return _impl(conjecture, dataset, error_threshold=error_threshold, max_counterexamples=max_counterexamples)


def initialize_theory_memory() -> dict[str, Any]:
    """Public wrapper for deterministic theory memory initialization."""
    from src.qec.analysis.spectral_theory_memory import initialize_theory_memory as _impl
    return _impl()


def update_theory_memory(
    theory_memory: dict[str, Any],
    conjectures: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    counterexamples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Public wrapper for deterministic theory memory updates."""
    from src.qec.analysis.spectral_theory_memory import update_theory_memory as _impl
    return _impl(theory_memory, conjectures, validations, counterexamples)


def summarize_theory_memory(theory_memory: dict[str, Any]) -> list[dict[str, Any]]:
    """Public wrapper for deterministic theory memory summary generation."""
    from src.qec.analysis.spectral_theory_memory import summarize_theory_memory as _impl
    return _impl(theory_memory)




def detect_spectral_ridges(points: list[list[float]] | np.ndarray) -> list[dict[str, Any]]:
    """Public wrapper for deterministic spectral ridge detection."""
    from src.qec.analysis.spectral_ridges import detect_spectral_ridges as _impl
    return _impl(points)


def build_ridge_graph(ridges: list[dict[str, Any]]) -> dict[str, Any]:
    """Public wrapper for deterministic ridge-graph construction."""
    from src.qec.analysis.spectral_ridges import build_ridge_graph as _impl
    return _impl(ridges)


def map_ridges_to_basins(
    ridges: list[dict[str, Any]],
    basins: list[dict[str, Any]],
) -> dict[str, Any]:
    """Public wrapper for deterministic basin/ridge boundary mapping."""
    from src.qec.analysis.spectral_ridges import map_ridges_to_basins as _impl
    return _impl(ridges, basins)
def detect_spectral_basins(points: list[list[float]] | np.ndarray) -> list[dict[str, Any]]:
    """Public wrapper for deterministic spectral basin detection."""
    from src.qec.analysis.spectral_basins import detect_spectral_basins as _impl
    return _impl(points)


def build_basin_transition_graph(
    trajectory: list[list[float]] | np.ndarray,
    basins: list[dict[str, Any]],
) -> dict[str, Any]:
    """Public wrapper for deterministic basin transition graph construction."""
    from src.qec.analysis.spectral_basins import build_basin_transition_graph as _impl
    return _impl(trajectory, basins)


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
