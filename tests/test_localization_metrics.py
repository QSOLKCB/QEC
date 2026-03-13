from __future__ import annotations

import numpy as np

from src.qec.analysis.localization_metrics import (
    participation_entropy,
    inverse_participation_ratio,
    effective_support_size,
    extract_support,
)


def test_metrics_basic_values() -> None:
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    assert participation_entropy(v) == 0.0
    assert inverse_participation_ratio(v) == 1.0
    assert effective_support_size(v) == 1.0


def test_extract_support_default_tau() -> None:
    v = np.array([0.01, 0.5, -1.0, 0.08], dtype=np.float64)
    assert extract_support(v) == [1, 2]


def test_metrics_deterministic() -> None:
    v = np.array([0.3, -0.4, 0.5, 0.7], dtype=np.float64)
    r1 = (participation_entropy(v), inverse_participation_ratio(v), effective_support_size(v), extract_support(v))
    r2 = (participation_entropy(v), inverse_participation_ratio(v), effective_support_size(v), extract_support(v))
    assert r1 == r2
