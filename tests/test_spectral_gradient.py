from __future__ import annotations

import numpy as np

from src.qec.analysis.spectral_gradient import estimate_spectral_gradient
from src.qec.analysis.spectral_trajectory import SpectralTrajectoryRecorder
from src.qec.analysis.trajectory_sonifier import (
    interpolate_spectra,
    sonify_trajectory_smooth,
)
from src.qec.discovery.discovery_engine import run_structure_discovery
from src.qec.discovery.spectral_gradient_mutation import propose_gradient_step


def _default_spec() -> dict[str, int]:
    return {
        "num_variables": 6,
        "num_checks": 3,
        "variable_degree": 2,
        "check_degree": 4,
    }


def test_gradient_estimation_correctness() -> None:
    trajectory = np.asarray(
        [
            [1.0, 0.2],
            [1.1, 0.25],
            [1.2, 0.3],
        ],
        dtype=np.float64,
    )

    gradient = estimate_spectral_gradient(trajectory)
    np.testing.assert_allclose(gradient, np.asarray([0.1, 0.05], dtype=np.float64))


def test_gradient_estimation_is_deterministic() -> None:
    trajectory = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.25, 0.35],
            [0.3, 0.3, 0.4],
        ],
        dtype=np.float64,
    )
    g1 = estimate_spectral_gradient(trajectory)
    g2 = estimate_spectral_gradient(trajectory)
    np.testing.assert_array_equal(g1, g2)


def test_propose_gradient_step_float64() -> None:
    current = np.asarray([1.0, 0.2, 0.1], dtype=np.float64)
    gradient = np.asarray([0.05, -0.1, 0.2], dtype=np.float64)
    target = propose_gradient_step(current, gradient, step=0.2)

    np.testing.assert_allclose(target, np.asarray([1.01, 0.18, 0.14], dtype=np.float64))
    assert target.dtype == np.float64


def test_trajectory_compression_save_every_n_steps() -> None:
    recorder = SpectralTrajectoryRecorder(save_every_n_steps=2)
    recorder.record([1.0, 0.2])
    recorder.record([1.1, 0.25])
    recorder.record([1.2, 0.3])
    recorder.record([1.3, 0.35])

    arr = recorder.as_array()
    assert arr.shape == (2, 2)
    np.testing.assert_allclose(arr[0], np.asarray([1.0, 0.2], dtype=np.float64))
    np.testing.assert_allclose(arr[1], np.asarray([1.2, 0.3], dtype=np.float64))


def test_interpolation_shape_correctness() -> None:
    a = np.asarray([1.0, 0.2, 0.1], dtype=np.float64)
    b = np.asarray([1.2, 0.3, 0.2], dtype=np.float64)
    interp = interpolate_spectra(a, b, steps=5)

    assert interp.shape == (5, 3)
    np.testing.assert_allclose(interp[0], a)
    np.testing.assert_allclose(interp[-1], b)


def test_smooth_sonification_reproducibility() -> None:
    trajectory = [
        [1.0, 0.2, 0.1],
        [1.1, 0.25, 0.15],
        [1.2, 0.3, 0.2],
    ]
    audio1 = sonify_trajectory_smooth(trajectory, interpolation_steps=4)
    audio2 = sonify_trajectory_smooth(trajectory, interpolation_steps=4)

    assert audio1.dtype == np.float64
    np.testing.assert_array_equal(audio1, audio2)


def test_gradient_enabled_discovery_is_deterministic() -> None:
    spec = _default_spec()
    recorder_a = SpectralTrajectoryRecorder()
    recorder_b = SpectralTrajectoryRecorder()

    result_a = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_spectral_trajectory=True,
        trajectory_recorder=recorder_a,
        enable_spectral_gradient=True,
        gradient_step_size=0.1,
    )
    result_b = run_structure_discovery(
        spec,
        num_generations=2,
        population_size=4,
        base_seed=7,
        enable_spectral_trajectory=True,
        trajectory_recorder=recorder_b,
        enable_spectral_gradient=True,
        gradient_step_size=0.1,
    )

    assert result_a["elite_history"] == result_b["elite_history"]
    assert result_a["spectral_trajectory"] == result_b["spectral_trajectory"]
