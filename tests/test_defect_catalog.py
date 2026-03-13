from __future__ import annotations

import numpy as np

from src.qec.analysis.api import detect_spectral_defects


def _sample_H() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def test_defect_catalog_ordering_is_deterministic() -> None:
    H = _sample_H()
    d1 = detect_spectral_defects(H)
    d2 = detect_spectral_defects(H)
    assert d1 == d2
    severities = [d.severity for d in d1]
    assert severities == sorted(severities, reverse=True)


def test_defect_catalog_output_is_stable_and_typed() -> None:
    H = _sample_H()
    defects = detect_spectral_defects(H)
    for defect in defects:
        assert isinstance(defect.eigenvalue, float)
        assert isinstance(defect.severity, float)
        assert isinstance(defect.support_nodes, tuple)
        assert defect.classification in {
            "SMALL_TRAPPING_SET",
            "CYCLE_CLUSTER",
            "PSEUDO_CODEWORD_BASIN",
            "BENIGN_GLOBAL_MODE",
        }
