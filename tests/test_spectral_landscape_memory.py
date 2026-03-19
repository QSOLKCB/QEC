from __future__ import annotations

import numpy as np

from qec.analysis.landscape_io import load_landscape, save_landscape
from qec.analysis.landscape_metrics import landscape_coverage, novelty_score
from qec.analysis.spectral_landscape_memory import SpectralLandscapeMemory, cluster_regions
from qec.analysis.spectral_trajectory import SpectralTrajectoryRecorder
from qec.discovery.discovery_engine import run_structure_discovery


def _default_spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_landscape_memory_accumulates_into_centers_and_counts() -> None:
    mem = SpectralLandscapeMemory(max_regions=100)
    mem.add([1.0, 0.2], threshold=0.05)
    mem.add([1.01, 0.21], threshold=0.05)

    assert mem.region_count == 1
    assert mem.counts == [2]
    assert mem.centers().shape == (1, 2)
    assert mem.centers().dtype == np.float64


def test_cluster_regions_is_deterministic() -> None:
    states = np.asarray(
        [[1.0, 0.0], [1.01, 0.02], [3.0, 3.0], [3.01, 3.02]],
        dtype=np.float64,
    )
    r1 = cluster_regions(states, threshold=0.1)
    r2 = cluster_regions(states, threshold=0.1)
    np.testing.assert_allclose(r1, r2)
    assert r1.shape == (2, 2)


def test_capacity_scaling_preserves_centers() -> None:
    mem = SpectralLandscapeMemory(dim=2, initial_capacity=2, max_regions=100)
    mem.add([0.0, 0.0], threshold=0.01)
    mem.add([1.0, 1.0], threshold=0.01)
    before = mem.centers().copy()
    mem.add([2.0, 2.0], threshold=0.01)

    assert mem.capacity >= 4
    assert mem.region_count == 3
    np.testing.assert_allclose(mem.centers()[:2], before)


def test_region_cap_drops_oldest_deterministically() -> None:
    mem = SpectralLandscapeMemory(dim=2, initial_capacity=2, max_regions=2)
    mem.add([0.0, 0.0], threshold=0.01)
    mem.add([1.0, 1.0], threshold=0.01)
    mem.add([2.0, 2.0], threshold=0.01)

    assert mem.region_count == 2
    centers = mem.centers()
    np.testing.assert_allclose(centers[0], np.asarray([1.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(centers[1], np.asarray([2.0, 2.0], dtype=np.float64))


def test_save_load_landscape_roundtrip(tmp_path) -> None:
    mem = SpectralLandscapeMemory(dim=2, max_regions=100)
    mem.add([1.0, 0.2], threshold=0.1)
    mem.add([1.04, 0.24], threshold=0.1)
    mem.add([3.0, 1.5], threshold=0.1)

    path = tmp_path / "landscape.json"
    save_landscape(mem, str(path))
    loaded = load_landscape(str(path))

    np.testing.assert_allclose(loaded.centers(), mem.centers())
    assert loaded.counts == mem.counts
    assert loaded.region_count == mem.region_count
    assert landscape_coverage(loaded) == 2.0


def test_novelty_score_correctness() -> None:
    mem = SpectralLandscapeMemory(dim=2, max_regions=100)
    mem.add([0.0, 0.0], threshold=0.01)
    mem.add([1.0, 0.0], threshold=0.01)

    score = novelty_score(np.asarray([0.0, 1.0], dtype=np.float64), mem)
    expected_min_d2 = 1.0
    expected = expected_min_d2 / (1.0 + expected_min_d2)
    assert np.isclose(score, expected)


def test_discovery_landscape_learning_accumulates_across_runs() -> None:
    spec = _default_spec()
    mem = SpectralLandscapeMemory(max_regions=10000)

    r1 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_spectral_trajectory=True,
        trajectory_recorder=SpectralTrajectoryRecorder(),
        enable_landscape_learning=True,
        landscape_memory=mem,
    )
    n_after_first = mem.region_count

    r2 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_spectral_trajectory=True,
        trajectory_recorder=SpectralTrajectoryRecorder(),
        enable_landscape_learning=True,
        landscape_memory=mem,
    )
    n_after_second = mem.region_count

    assert n_after_first > 0
    assert n_after_second >= n_after_first
    assert "spectral_landscape_regions" in r1
    assert "landscape_coverage" in r1
    assert "spectral_landscape_regions" in r2
    assert "landscape_coverage" in r2


def test_discovery_landscape_learning_deterministic() -> None:
    spec = _default_spec()
    recorder_a = SpectralTrajectoryRecorder()
    recorder_b = SpectralTrajectoryRecorder()

    r1 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=23,
        enable_spectral_trajectory=True,
        trajectory_recorder=recorder_a,
        enable_landscape_learning=True,
        landscape_memory=SpectralLandscapeMemory(),
        novelty_weight=0.25,
    )
    r2 = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=23,
        enable_spectral_trajectory=True,
        trajectory_recorder=recorder_b,
        enable_landscape_learning=True,
        landscape_memory=SpectralLandscapeMemory(),
        novelty_weight=0.25,
    )

    assert r1["elite_history"] == r2["elite_history"]
    assert r1["spectral_landscape_regions"] == r2["spectral_landscape_regions"]
    assert r1["landscape_coverage"] == r2["landscape_coverage"]


def test_default_discovery_result_unchanged_shape() -> None:
    spec = _default_spec()
    result = run_structure_discovery(spec, num_generations=1, population_size=4, base_seed=5)
    assert "spectral_landscape_regions" not in result
    assert "landscape_coverage" not in result
