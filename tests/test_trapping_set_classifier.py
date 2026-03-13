from __future__ import annotations

import numpy as np

from src.qec.analysis.trapping_set_classifier import classify_defect


def test_classifier_small_trapping() -> None:
    v = np.zeros(100, dtype=np.float64)
    v[0] = 1.0
    out = classify_defect(v)
    assert out["defect_type"] == "SMALL_TRAPPING_SET"


def test_classifier_benign_mode() -> None:
    v = np.ones(10, dtype=np.float64)
    out = classify_defect(v)
    assert out["defect_type"] == "BENIGN_GLOBAL_MODE"


def test_classifier_determinism() -> None:
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    assert classify_defect(v) == classify_defect(v)
