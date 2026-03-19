from __future__ import annotations

import numpy as np

from qec.analysis.information_gain import (
    information_gain_score,
    rank_candidates_by_information_gain,
)
from qec.analysis.spectral_landscape_memory import SpectralLandscapeMemory
from qec.analysis.spectral_uncertainty import estimate_spectral_uncertainty
from qec.discovery.autonomous_scheduler import schedule_autonomous_target
from qec.discovery.discovery_engine import run_structure_discovery


def test_uncertainty_estimation_correctness() -> None:
    memory = SpectralLandscapeMemory(dim=2)
    memory.add(np.asarray([0.0, 0.0], dtype=np.float64), threshold=0.01)

    near = estimate_spectral_uncertainty(memory, np.asarray([0.0, 0.0], dtype=np.float64))
    far = estimate_spectral_uncertainty(memory, np.asarray([3.0, 4.0], dtype=np.float64))

    assert near == 0.0
    assert np.isclose(far, 25.0 / 26.0)



def test_uncertainty_empty_landscape() -> None:
    memory = SpectralLandscapeMemory()
    spectrum = np.zeros(4, dtype=np.float64)

    u = estimate_spectral_uncertainty(memory, spectrum)

    assert u == 1.0

def test_information_gain_scoring_stability() -> None:
    memory = SpectralLandscapeMemory(dim=2)
    memory.add(np.asarray([1.0, 1.0], dtype=np.float64), threshold=0.01)
    candidate = np.asarray([2.0, 1.0], dtype=np.float64)

    s1 = information_gain_score(memory, candidate, novelty_weight=0.25, uncertainty_weight=0.75)
    s2 = information_gain_score(memory, candidate, novelty_weight=0.25, uncertainty_weight=0.75)
    assert s1 == s2


def test_candidate_ranking_determinism_tiebreak() -> None:
    memory = SpectralLandscapeMemory(dim=2)
    memory.add(np.asarray([0.0, 0.0], dtype=np.float64), threshold=0.01)
    candidates = [
        np.asarray([1.0, 0.0], dtype=np.float64),
        np.asarray([1.0, 0.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
    ]

    ranked = rank_candidates_by_information_gain(candidates, memory)
    indices = [item["candidate_index"] for item in ranked]
    assert indices == [0, 1, 2]


def test_scheduler_reproducibility() -> None:
    memory = SpectralLandscapeMemory(dim=2)
    memory.add(np.asarray([0.0, 0.0], dtype=np.float64), threshold=0.01)
    frontier = [
        np.asarray([0.1, 0.0], dtype=np.float64),
        np.asarray([2.0, 2.0], dtype=np.float64),
    ]

    r1 = schedule_autonomous_target(memory, frontier, strategy="information_gain")
    r2 = schedule_autonomous_target(memory, frontier, strategy="information_gain")

    assert np.allclose(r1["target_spectrum"], r2["target_spectrum"])
    assert r1["information_gain_score"] == r2["information_gain_score"]


def test_engine_information_gain_integration_determinism() -> None:
    spec = {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }
    r1 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_landscape_learning=True,
        enable_information_gain_scheduler=True,
    )
    r2 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_landscape_learning=True,
        enable_information_gain_scheduler=True,
    )
    g1 = r1["generation_summaries"][-1]
    g2 = r2["generation_summaries"][-1]

    assert g1["information_gain_score"] == g2["information_gain_score"]
    assert g1["spectral_uncertainty"] == g2["spectral_uncertainty"]
    assert g1["novelty_score"] == g2["novelty_score"]
    assert g1["selected_target_spectrum"] == g2["selected_target_spectrum"]
