from __future__ import annotations

import numpy as np

from src.qec.analysis.spectral_geometry import (
    spectral_distance,
    spectral_diversity,
    spectral_entropy,
)


def test_spectral_distance_euclidean() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([1.0, 3.0, 7.0], dtype=np.float64)
    assert spectral_distance(a, b) == np.sqrt(17.0)


def test_spectral_entropy_zero_sum() -> None:
    assert spectral_entropy([0.0, 0.0, 0.0]) == 0.0


def test_spectral_diversity_mean_step_distance() -> None:
    history = [
        [0.0, 0.0, 0.0],
        [0.0, 3.0, 4.0],
        [0.0, 6.0, 8.0],
    ]
    assert spectral_diversity(history) == 5.0


def test_spectral_geometry_determinism() -> None:
    a = [0.1, -0.2, 0.3]
    b = [0.3, -0.1, 0.0]
    d1 = spectral_distance(a, b)
    d2 = spectral_distance(a, b)
    e1 = spectral_entropy(a)
    e2 = spectral_entropy(a)
    assert d1 == d2
    assert e1 == e2
