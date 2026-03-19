"""Tests for v16.5.0 spectral temperature annealing."""

from __future__ import annotations

import numpy as np

from qec.discovery.mutation_nb_gradient import NBGradientMutator


def _matrix() -> np.ndarray:
    return np.array([
        [1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1],
    ], dtype=np.float64)


def test_entropy_temperature_decay_monotonic() -> None:
    mut = NBGradientMutator(
        enabled=True,
        eta_entropy=1.0,
        enable_entropy_annealing=True,
        entropy_temperature_start=1.0,
        entropy_temperature_decay=50.0,
    )

    mut._mutation_step = 0
    t0 = mut._entropy_temperature()
    mut._mutation_step = 10
    t10 = mut._entropy_temperature()
    mut._mutation_step = 100
    t100 = mut._entropy_temperature()

    assert t0 > t10 > t100


def test_entropy_temperature_disabled_mode_is_one() -> None:
    mut = NBGradientMutator(
        enabled=True,
        eta_entropy=1.0,
        enable_entropy_annealing=False,
        entropy_temperature_start=3.0,
        entropy_temperature_decay=7.0,
    )
    mut._mutation_step = 100
    assert mut._entropy_temperature() == 1.0


def test_entropy_temperature_determinism_repeated_runs() -> None:
    H = _matrix()
    kwargs = dict(
        enabled=True,
        avoid_4cycles=False,
        eta_entropy=0.5,
        enable_entropy_annealing=True,
        entropy_temperature_start=1.0,
        entropy_temperature_decay=20.0,
    )

    mut_a = NBGradientMutator(**kwargs)
    _, log_a = mut_a.mutate(H, steps=3)

    mut_b = NBGradientMutator(**kwargs)
    _, log_b = mut_b.mutate(H, steps=3)

    temps_a = [step["entropy_temperature"] for step in log_a]
    temps_b = [step["entropy_temperature"] for step in log_b]
    assert temps_a == temps_b


def test_entropy_annealing_score_stability() -> None:
    H = _matrix()
    kwargs = dict(
        enabled=True,
        avoid_4cycles=False,
        eta_entropy=0.25,
        enable_entropy_annealing=True,
        entropy_temperature_start=1.0,
        entropy_temperature_decay=40.0,
    )

    mut_a = NBGradientMutator(**kwargs)
    _, log_a = mut_a.mutate(H, steps=4)

    mut_b = NBGradientMutator(**kwargs)
    _, log_b = mut_b.mutate(H, steps=4)

    assert log_a == log_b
    if log_a:
        for step in log_a:
            assert "delta_entropy" in step
            assert "entropy_temperature" in step
