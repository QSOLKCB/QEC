from __future__ import annotations

import numpy as np

from src.qec.discovery.phase_guided_search import (
    propose_phase_guided_step,
    select_phase_target,
    update_phase_visit_counts,
)
from src.qec.discovery.discovery_engine import run_structure_discovery


def _phase_map() -> dict[str, object]:
    return {
        "phase_regions": [
            {"phase_id": 2, "centroid": [0.8, 0.2, 0.4]},
            {"phase_id": 1, "centroid": [0.2, 0.5, 0.7]},
            {"phase_id": 0, "centroid": [0.1, 0.1, 0.1]},
        ]
    }


def _spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_select_phase_target_deterministic_underexplored() -> None:
    phase_map = _phase_map()
    visit_counts = {0: 3, 1: 0, 2: 1}
    target_1 = select_phase_target(phase_map, visit_counts)
    target_2 = select_phase_target(phase_map, visit_counts)
    assert target_1 == {"target_phase_id": 1}
    assert target_1 == target_2


def test_propose_phase_guided_step_reproducible() -> None:
    current = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    phase_map = _phase_map()
    target = {"target_phase_id": 1}

    step_1 = propose_phase_guided_step(current, phase_map, target)
    step_2 = propose_phase_guided_step(current, phase_map, target)

    expected = np.asarray([0.05, 0.875, 0.175], dtype=np.float64)
    assert step_1.dtype == np.float64
    np.testing.assert_allclose(step_1, expected, atol=0.0)
    np.testing.assert_allclose(step_1, step_2, atol=0.0)


def test_update_phase_visit_counts_stable_ordering() -> None:
    counts = {3: 4, 1: 2}
    updated = update_phase_visit_counts(2, counts)
    updated = update_phase_visit_counts(1, updated)
    assert list(updated.keys()) == [1, 2, 3]
    assert updated == {1: 3, 2: 1, 3: 4}


def test_phase_guided_engine_integration_deterministic() -> None:
    spec = _spec()
    args = dict(
        num_generations=2,
        population_size=4,
        base_seed=42,
        enable_phase_guided_discovery=True,
        phase_guidance_interval=1,
        enable_basin_hopping=True,
        basin_detection_interval=1,
        enable_spectral_ridge_detection=True,
        ridge_detection_interval=1,
        enable_phase_map_reconstruction=True,
        phase_map_interval=1,
    )
    r1 = run_structure_discovery(spec, **args)
    r2 = run_structure_discovery(spec, **args)

    assert r1["phase_visit_counts"] == r2["phase_visit_counts"]
    assert r1["phase_guidance_targets"] == r2["phase_guidance_targets"]
    if r1["phase_guidance_targets"]:
        summary = r1["generation_summaries"][-1]
        assert "current_phase_target" in summary
        assert "phase_guidance_step" in summary
