from __future__ import annotations

import numpy as np

from src.qec.analysis.defect_catalog import detect_spectral_defects


def _H() -> np.ndarray:
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


def test_catalog_sorted_by_severity() -> None:
    defects = detect_spectral_defects(_H(), k_max=8, use_multiresolution_scan=True, scan_steps=2)
    vals = [d.severity for d in defects]
    assert vals == sorted(vals, reverse=True)


def test_catalog_determinism() -> None:
    d1 = detect_spectral_defects(_H(), k_max=8, use_multiresolution_scan=True, scan_steps=2)
    d2 = detect_spectral_defects(_H(), k_max=8, use_multiresolution_scan=True, scan_steps=2)
    assert d1 == d2
