from __future__ import annotations

import json

import numpy as np

from qec.analysis.discovery_archive_analyzer import analyze_discovery_archive
from qec.analysis.hypothesis_generator import generate_structural_hypotheses
from qec.analysis.hypothesis_ranking import rank_hypotheses
from qec.analysis.spectral_landscape_memory import SpectralLandscapeMemory
from qec.analysis.spectral_phase_boundaries import detect_phase_boundaries
from qec.discovery.archive import create_archive, update_discovery_archive
from qec.discovery.autonomous_scheduler import (
    compute_combined_score,
    select_best_candidate,
)
from qec.discovery.discovery_engine import run_structure_discovery


def _spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def _build_archive() -> dict[str, object]:
    archive = create_archive(top_k=6)
    candidates = []
    for i in range(6):
        candidates.append(
            {
                "candidate_id": f"c{i}",
                "H": np.zeros((2, 2), dtype=np.float64),
                "objectives": {
                    "spectral_radius": float(np.float64(1.0 + 0.1 * i)),
                    "bethe_margin": float(np.float64(0.2 + 0.05 * i)),
                    "bp_stability": float(np.float64(0.2 + 0.05 * i)),
                    "motif_frequency": float(np.float64(0.1 * i)),
                    "cycle_density": float(np.float64(0.6 - 0.04 * i)),
                    "mean_degree": 2.0,
                    "composite_score": float(np.float64(2.5 - 0.2 * i)),
                    "instability_score": float(np.float64(1.0 / (1.0 + i))),
                    "entropy": float(np.float64(0.2 + 0.01 * i)),
                },
                "metrics": {},
                "generation": 0,
                "novelty": float(np.float64(0.05 * i)),
                "is_feasible": True,
            }
        )
    return update_discovery_archive(archive, candidates)


def test_archive_analysis_is_deterministic() -> None:
    archive = _build_archive()
    a1 = analyze_discovery_archive(archive)
    a2 = analyze_discovery_archive(archive)
    assert json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True)
    assert isinstance(a1["feature_correlations"], dict)


def test_hypothesis_generation_and_ranking_are_reproducible() -> None:
    correlations = analyze_discovery_archive(_build_archive())
    h1 = generate_structural_hypotheses(correlations["feature_correlations"])
    h2 = generate_structural_hypotheses(correlations["feature_correlations"])
    assert json.dumps(h1, sort_keys=True) == json.dumps(h2, sort_keys=True)

    r1 = rank_hypotheses(h1)
    r2 = rank_hypotheses(h2)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    assert all("score" in item for item in r1)
    assert all(isinstance(item["hypothesis_id"], int) for item in h1)


def test_scheduler_hypothesis_guided_selection_is_deterministic() -> None:
    candidates = [
        {"exploration_score": 0.25, "hypothesis_bias": 0.90},
        {"exploration_score": 0.40, "hypothesis_bias": 0.20},
    ]
    i1 = select_best_candidate(candidates, strategy="hypothesis_guided", hypothesis_weight=0.5)
    i2 = select_best_candidate(candidates, strategy="hypothesis_guided", hypothesis_weight=0.5)
    assert i1 == i2 == 0

    score = compute_combined_score(0.8, 0.2, strategy="hypothesis_guided", hypothesis_weight=0.25)
    assert np.isclose(score, 0.65)


def test_phase_boundary_detection_is_deterministic() -> None:
    memory = SpectralLandscapeMemory(dim=4)
    memory.add([0.2, 0.1, 0.0, 0.0], threshold=0.01)
    memory.add([0.8, 0.2, 0.0, 0.0], threshold=0.01)
    memory.add([0.9, 0.4, 0.0, 0.0], threshold=0.01)

    p1 = detect_phase_boundaries(memory)
    p2 = detect_phase_boundaries(memory)
    assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)


def test_reflective_engine_mode_is_reproducible() -> None:
    spec = _spec()
    r1 = run_structure_discovery(
        spec,
        num_generations=3,
        population_size=4,
        base_seed=123,
        enable_self_reflection=True,
        reflection_interval=1,
        hypothesis_weight=0.5,
    )
    r2 = run_structure_discovery(
        spec,
        num_generations=3,
        population_size=4,
        base_seed=123,
        enable_self_reflection=True,
        reflection_interval=1,
        hypothesis_weight=0.5,
    )
    assert json.dumps(r1["generation_summaries"], sort_keys=True) == json.dumps(
        r2["generation_summaries"], sort_keys=True
    )
    assert json.dumps(r1.get("hypothesis_rankings", []), sort_keys=True) == json.dumps(
        r2.get("hypothesis_rankings", []), sort_keys=True
    )
