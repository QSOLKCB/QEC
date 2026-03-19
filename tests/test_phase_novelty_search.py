from __future__ import annotations

import numpy as np

from qec.discovery.discovery_engine import run_structure_discovery
from qec.discovery.phase_novelty_search import (
    compute_phase_novelty_score,
    detect_new_phase,
    propose_phase_novelty_step,
    select_novel_phase_target,
)


def _default_spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_compute_phase_novelty_score_deterministic() -> None:
    vector = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    centroids = np.asarray([[1.0, 2.0, 2.0], [3.0, 2.0, 3.0]], dtype=np.float64)
    s1 = compute_phase_novelty_score(vector, centroids)
    s2 = compute_phase_novelty_score(vector, centroids)
    assert np.isclose(s1, 1.0)
    assert s1 == s2


def test_select_novel_phase_target_deterministic_tie_break() -> None:
    candidates = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    centroids = np.asarray([[0.0, 0.0]], dtype=np.float64)
    selected = select_novel_phase_target(candidates, centroids)
    assert np.allclose(selected["novelty_vector"], np.asarray([0.0, 2.0], dtype=np.float64))


def test_propose_phase_novelty_step_deterministic() -> None:
    current = np.asarray([0.0, 0.0], dtype=np.float64)
    novelty = np.asarray([3.0, 4.0], dtype=np.float64)
    step = propose_phase_novelty_step(current, novelty)
    assert np.allclose(step, np.asarray([0.15, 0.2], dtype=np.float64))
    assert step.dtype == np.float64


def test_detect_new_phase_threshold() -> None:
    vector = np.asarray([2.0, 2.0], dtype=np.float64)
    centroids = np.asarray([[0.0, 0.0]], dtype=np.float64)
    assert detect_new_phase(vector, centroids, threshold=2.5)["is_new_phase"] is True
    assert detect_new_phase(vector, centroids, threshold=3.0)["is_new_phase"] is False


def test_engine_phase_novelty_discovery_reproducible() -> None:
    spec = _default_spec()
    r1 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=42,
        enable_phase_novelty_discovery=True,
        phase_novelty_interval=1,
        enable_basin_hopping=True,
        basin_detection_interval=1,
    )
    r2 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=42,
        enable_phase_novelty_discovery=True,
        phase_novelty_interval=1,
        enable_basin_hopping=True,
        basin_detection_interval=1,
    )
    assert r1["novelty_scores"] == r2["novelty_scores"]
    assert r1["novel_phase_candidates"] == r2["novel_phase_candidates"]
    summary = r1["generation_summaries"][-1]
    assert "phase_novelty_score" in summary
    assert "novel_phase_detected" in summary
