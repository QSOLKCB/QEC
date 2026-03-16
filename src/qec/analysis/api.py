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
    "spectral_phase_map",
    "phase_characterization",
    "theory_synthesis",
    "conjecture_validation",
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


def compute_phase_novelty_score(vector: np.ndarray, known_phase_centroids: list[np.ndarray] | np.ndarray) -> float:
    """Public wrapper for deterministic phase novelty scoring."""
    from src.qec.discovery.phase_novelty_search import compute_phase_novelty_score as _impl
    return _impl(vector, known_phase_centroids)


def select_novel_phase_target(
    candidate_vectors: list[np.ndarray] | np.ndarray,
    known_phase_centroids: list[np.ndarray] | np.ndarray,
) -> dict[str, np.ndarray]:
    """Public wrapper for deterministic novelty-target selection."""
    from src.qec.discovery.phase_novelty_search import select_novel_phase_target as _impl
    return _impl(candidate_vectors, known_phase_centroids)


def propose_phase_novelty_step(current_vector: np.ndarray, novelty_vector: np.ndarray) -> np.ndarray:
    """Public wrapper for deterministic novelty-guided step proposal."""
    from src.qec.discovery.phase_novelty_search import propose_phase_novelty_step as _impl
    return _impl(current_vector, novelty_vector)



def compute_phase_metrics(graph: np.ndarray, spectrum: np.ndarray, decoder_stats: dict[str, Any] | None) -> dict[str, float]:
    """Public wrapper for deterministic phase metric computation."""
    from src.qec.analysis.phase_characterization import compute_phase_metrics as _impl
    return _impl(graph, spectrum, decoder_stats)


def classify_phase(metrics: dict[str, Any]) -> dict[str, str]:
    """Public wrapper for deterministic phase classification."""
    from src.qec.analysis.phase_characterization import classify_phase as _impl
    return _impl(metrics)


def build_phase_profile(phase_id: int, metrics: dict[str, Any], label: dict[str, Any] | str) -> dict[str, Any]:
    """Public wrapper for deterministic phase profile construction."""
    from src.qec.analysis.phase_characterization import build_phase_profile as _impl
    return _impl(phase_id, metrics, label)


def build_phase_dataset(phase_profiles: Any) -> dict[str, Any]:
    """Public wrapper for deterministic phase dataset extraction."""
    from src.qec.analysis.theory_synthesis import build_phase_dataset as _impl
    return _impl(phase_profiles)


def fit_spectral_models(X: Any, y: Any) -> list[dict[str, Any]]:
    """Public wrapper for deterministic spectral model fitting."""
    from src.qec.analysis.theory_synthesis import fit_spectral_models as _impl
    return _impl(X, y)


def generate_spectral_conjectures(models: list[dict[str, Any]], feature_names: list[str]) -> list[dict[str, Any]]:
    """Public wrapper for deterministic spectral conjecture synthesis."""
    from src.qec.analysis.theory_synthesis import generate_spectral_conjectures as _impl
    return _impl(models, feature_names)


def evaluate_conjecture(conjecture: dict[str, Any], phase_profile: dict[str, Any]) -> dict[str, Any]:
    """Public wrapper for deterministic conjecture evaluation against a phase profile."""
    from src.qec.analysis.conjecture_validation import evaluate_conjecture as _impl
    return _impl(conjecture, phase_profile)


def find_conjecture_counterexamples(
    conjecture: dict[str, Any],
    phase_profiles: list[dict[str, Any]],
    error_threshold: float = 0.1,
    max_counterexamples: int = 128,
) -> list[dict[str, Any]]:
    """Public wrapper for deterministic conjecture counterexample detection."""
    from src.qec.analysis.conjecture_validation import find_conjecture_counterexamples as _impl
    return _impl(conjecture, phase_profiles, error_threshold=error_threshold, max_counterexamples=max_counterexamples)


def design_validation_experiment(
    conjecture: dict[str, Any],
    archive: list[dict[str, Any]],
    num_targets: int = 8,
) -> list[dict[str, Any]]:
    """Public wrapper for deterministic validation experiment design."""
    from src.qec.analysis.conjecture_validation import design_validation_experiment as _impl
    return _impl(conjecture, archive, num_targets=num_targets)


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



def run_ternary_decoder(
    parity_matrix: np.ndarray,
    received_vector: np.ndarray,
    *,
    max_iterations: int = 20,
) -> dict[str, Any]:
    """Public wrapper for deterministic ternary message-passing decoder."""
    from src.qec.decoder.ternary.ternary_decoder import run_ternary_decoder as _impl
    return _impl(parity_matrix, received_vector, max_iterations=max_iterations)


def compute_ternary_stability(messages: np.ndarray) -> np.float64:
    """Public wrapper for deterministic ternary stability computation."""
    from src.qec.decoder.ternary.ternary_metrics import compute_ternary_stability as _impl
    return _impl(messages)


def compute_ternary_entropy(messages: np.ndarray) -> np.float64:
    """Public wrapper for deterministic ternary entropy computation."""
    from src.qec.decoder.ternary.ternary_metrics import compute_ternary_entropy as _impl
    return _impl(messages)


def compute_ternary_conflict_density(messages: np.ndarray) -> np.float64:
    """Public wrapper for deterministic ternary conflict density computation."""
    from src.qec.decoder.ternary.ternary_metrics import compute_ternary_conflict_density as _impl
    return _impl(messages)


def detect_zero_regions(messages: np.ndarray) -> dict[str, Any]:
    """Public wrapper for deterministic ternary zero-region detection."""
    from src.qec.decoder.ternary.ternary_trapping import detect_zero_regions as _impl
    return _impl(messages)


def compute_frustration_index(messages: np.ndarray) -> np.float64:
    """Public wrapper for deterministic ternary frustration index computation."""
    from src.qec.decoder.ternary.ternary_trapping import compute_frustration_index as _impl
    return _impl(messages)


def detect_persistent_zero_states(history: list[np.ndarray]) -> list[int]:
    """Public wrapper for deterministic persistent zero-state detection."""
    from src.qec.decoder.ternary.ternary_trapping import detect_persistent_zero_states as _impl
    return _impl(history)


def estimate_trapping_indicator(messages: np.ndarray, parity_matrix: np.ndarray) -> np.float64:
    """Public wrapper for deterministic trapping indicator estimation."""
    from src.qec.decoder.ternary.ternary_trapping import estimate_trapping_indicator as _impl
    return _impl(messages, parity_matrix)


def run_decoder_with_rule(
    parity_matrix: np.ndarray,
    received: np.ndarray,
    rule_name: str,
    *,
    max_iterations: int = 20,
) -> dict[str, Any]:
    """Public wrapper for deterministic ternary decoder with rule variant."""
    from src.qec.decoder.ternary.ternary_rule_evaluator import run_decoder_with_rule as _impl
    return _impl(parity_matrix, received, rule_name, max_iterations=max_iterations)


def evaluate_decoder_rule(
    parity_matrix: np.ndarray,
    received: np.ndarray,
    rule_name: str,
    *,
    max_iterations: int = 20,
) -> dict[str, np.float64]:
    """Public wrapper for deterministic decoder rule evaluation."""
    from src.qec.decoder.ternary.ternary_rule_evaluator import evaluate_decoder_rule as _impl
    return _impl(parity_matrix, received, rule_name, max_iterations=max_iterations)


def list_decoder_rules() -> list[str]:
    """Public wrapper returning sorted list of available decoder rule names."""
    from src.qec.decoder.ternary.ternary_rule_variants import RULE_REGISTRY
    return sorted(RULE_REGISTRY.keys())


def evaluate_graph_decoder_pair(
    parity_matrix: np.ndarray,
    received: np.ndarray,
    rule_name: str,
    *,
    max_iterations: int = 20,
) -> dict[str, Any]:
    """Public wrapper for deterministic graph-decoder pair evaluation."""
    from src.qec.decoder.ternary.ternary_coevolution import evaluate_graph_decoder_pair as _impl
    return _impl(parity_matrix, received, rule_name, max_iterations=max_iterations)


def evaluate_rule_population(
    parity_matrix: np.ndarray,
    received: np.ndarray,
    *,
    max_iterations: int = 20,
) -> list[dict[str, Any]]:
    """Public wrapper for deterministic rule population evaluation."""
    from src.qec.decoder.ternary.ternary_coevolution import evaluate_rule_population as _impl
    return _impl(parity_matrix, received, max_iterations=max_iterations)


def select_best_rule(
    rule_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Public wrapper for deterministic best rule selection."""
    from src.qec.decoder.ternary.ternary_coevolution import select_best_rule as _impl
    return _impl(rule_results)


def construct_phase_map(
    basins: list[dict[str, Any]],
    ridges: list[dict[str, Any]],
    phase_surface: dict[str, Any] | None,
    trajectory: list[list[float]] | np.ndarray,
) -> dict[str, Any]:
    """Public wrapper for deterministic spectral phase-map construction."""
    from src.qec.analysis.spectral_phase_map import construct_phase_map as _impl
    return _impl(basins, ridges, phase_surface, trajectory)


def label_phases(basins: list[dict[str, Any]], ridges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Public wrapper for deterministic phase labeling from basins/ridges."""
    from src.qec.analysis.spectral_phase_map import label_phases as _impl
    return _impl(basins, ridges)


def render_phase_map(phase_map: dict[str, Any], output_path: str) -> dict[str, Any]:
    """Public wrapper for deterministic phase-map rendering."""
    from src.qec.analysis.spectral_phase_map import render_phase_map as _impl
    return _impl(phase_map, output_path)


def select_phase_target(phase_map: dict[str, Any], phase_visit_counts: dict[int, int]) -> dict[str, int]:
    """Public wrapper for deterministic phase target selection."""
    from src.qec.discovery.phase_guided_search import select_phase_target as _impl
    return _impl(phase_map, phase_visit_counts)


def propose_phase_guided_step(
    current_vector: np.ndarray,
    phase_map: dict[str, Any],
    target_phase: dict[str, Any] | int,
) -> np.ndarray:
    """Public wrapper for deterministic phase-guided spectral step proposal."""
    from src.qec.discovery.phase_guided_search import propose_phase_guided_step as _impl
    return _impl(current_vector, phase_map, target_phase)
