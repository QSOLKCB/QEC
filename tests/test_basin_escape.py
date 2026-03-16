from __future__ import annotations

import numpy as np

from src.qec.analysis.basin_escape_direction import estimate_escape_direction
from src.qec.analysis.basin_stagnation import detect_basin_stagnation
from src.qec.discovery.basin_escape_mutation import propose_escape_step
from src.qec.discovery.discovery_engine import run_structure_discovery


def _default_spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_stagnation_detection_correctness() -> None:
    assignments = [0, 0, 0, 0, 0, 0]
    assert detect_basin_stagnation(assignments, window=5)
    assert not detect_basin_stagnation([0, 0, 1, 0, 0], window=5)


def test_escape_direction_normalization() -> None:
    current = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
    center = np.asarray([1.0, 2.0, 1.0], dtype=np.float64)
    direction = estimate_escape_direction(current, center)

    np.testing.assert_allclose(np.linalg.norm(direction), 1.0)
    assert direction.dtype == np.float64




def test_boundary_aware_escape() -> None:
    current = np.asarray([1.0, 0.2], dtype=np.float64)
    basin_center = np.asarray([0.9, 0.1], dtype=np.float64)
    other = [
        np.asarray([1.5, 0.5], dtype=np.float64),
        np.asarray([2.0, 0.7], dtype=np.float64),
    ]

    direction = estimate_escape_direction(
        current,
        basin_center,
        other_centers=other,
    )

    assert direction.shape[0] == 2
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0)

def test_escape_step_deterministic_float64() -> None:
    current = np.asarray([1.0, 0.1, 0.2], dtype=np.float64)
    direction = np.asarray([0.0, 1.0, -1.0], dtype=np.float64)

    a = propose_escape_step(current, direction, step=0.3)
    b = propose_escape_step(current, direction, step=0.3)

    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, np.asarray([1.0, 0.4, -0.1], dtype=np.float64))
    assert a.dtype == np.float64


def test_discovery_engine_basin_escape_trigger_and_determinism() -> None:
    spec = _default_spec()
    a = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=17,
        enable_basin_escape=True,
        basin_escape_window=1,
        basin_escape_step=0.3,
    )
    b = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=17,
        enable_basin_escape=True,
        basin_escape_window=1,
        basin_escape_step=0.3,
    )

    assert "basin_escape_events" in a
    assert len(a["basin_escape_events"]) >= 1
    assert a["basin_escape_events"] == b["basin_escape_events"]

    event = a["basin_escape_events"][0]
    assert "escape_success" in event
    assert "escape_norm" in event


def test_discovery_engine_default_output_unchanged_without_escape() -> None:
    result = run_structure_discovery(
        _default_spec(),
        num_generations=1,
        population_size=4,
        base_seed=5,
    )
    assert "basin_escape_events" not in result
