from __future__ import annotations

import json

import numpy as np

from src.qec.analysis.curriculum_regions import classify_region_difficulty
from src.qec.analysis.spectral_landscape_memory import SpectralLandscapeMemory
from src.qec.analysis.spectral_difficulty import estimate_spectral_difficulty
from src.qec.analysis.curriculum_metrics import curriculum_success_rate, curriculum_progress
from src.qec.discovery.spectral_curriculum import SpectralCurriculumController
from src.qec.discovery.autonomous_scheduler import schedule_candidate
from src.qec.discovery.discovery_engine import run_structure_discovery


def _memory_with_regions() -> SpectralLandscapeMemory:
    mem = SpectralLandscapeMemory(dim=4)
    for row in (
        [0.1, 0.2, 0.05, 0.15],
        [0.2, 0.1, 0.10, 0.20],
        [0.8, 0.9, 0.70, 0.85],
        [1.1, 1.0, 1.05, 0.95],
    ):
        mem.add(np.asarray(row, dtype=np.float64), threshold=0.0)
    mem.recent_spectrum = np.asarray([0.15, 0.15, 0.07, 0.17], dtype=np.float64)
    return mem


def _default_spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_difficulty_estimation_stability() -> None:
    mem = _memory_with_regions()
    mem.spectral_uncertainty = np.float64(0.25)
    spectrum = np.asarray([0.2, 0.2, 0.1, 0.2], dtype=np.float64)
    d1 = estimate_spectral_difficulty(mem, spectrum)
    d2 = estimate_spectral_difficulty(mem, spectrum)
    assert np.float64(d1) == np.float64(d2)


def test_region_classification_determinism() -> None:
    mem = _memory_with_regions()
    t1 = classify_region_difficulty(mem)
    t2 = classify_region_difficulty(mem)
    assert t1 == t2
    assert set(t1.keys()) == {
        "tier_0_easy",
        "tier_1_intermediate",
        "tier_2_hard",
        "tier_3_frontier",
    }


def test_curriculum_progression_correctness() -> None:
    ctl = SpectralCurriculumController(current_tier=0)
    metrics = ctl.update_progress(3, 10, 0.2)
    assert metrics["advance"] is True
    advanced = ctl.advance_tier_if_ready(metrics["advance"])
    assert advanced is True
    assert ctl.current_tier == 1
    assert curriculum_success_rate(1, 0) == np.float64(0.0)
    assert curriculum_progress(1, 5, 0.5)["advance"] is False


def test_scheduler_integration_determinism() -> None:
    ctl = SpectralCurriculumController(current_tier=0)
    region_tiers = {
        "tier_0_easy": [0],
        "tier_1_intermediate": [1],
        "tier_2_hard": [2],
        "tier_3_frontier": [3],
    }
    candidates = [
        {"candidate_id": "c0", "region_index": 0, "spectrum": [0.1, 0.2, 0.3, 0.4], "bayesian_uncertainty": 0.2},
        {"candidate_id": "c1", "region_index": 1, "spectrum": [0.2, 0.2, 0.2, 0.2], "bayesian_uncertainty": 0.9},
    ]
    s1 = schedule_candidate(
        candidates,
        strategy="curriculum_exploration",
        curriculum_controller=ctl,
        region_tiers=region_tiers,
        memory=None,
        model=None,
    )
    s2 = schedule_candidate(
        candidates,
        strategy="curriculum_exploration",
        curriculum_controller=ctl,
        region_tiers=region_tiers,
        memory=None,
        model=None,
    )
    assert s1 == s2
    assert s1 is not None
    assert s1["candidate_id"] == "c0"


def test_engine_reproducibility_with_curriculum() -> None:
    spec = _default_spec()
    r1 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=9,
        enable_landscape_learning=True,
        enable_curriculum_learning=True,
        curriculum_success_threshold=0.2,
        curriculum_initial_tier=0,
    )
    r2 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=9,
        enable_landscape_learning=True,
        enable_curriculum_learning=True,
        curriculum_success_threshold=0.2,
        curriculum_initial_tier=0,
    )
    assert json.dumps(r1["generation_summaries"], sort_keys=True) == json.dumps(
        r2["generation_summaries"], sort_keys=True,
    )
    assert "curriculum_tier" in r1
    assert "curriculum_progress" in r1
    assert "curriculum_success_rate" in r1
