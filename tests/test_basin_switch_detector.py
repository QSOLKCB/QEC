from __future__ import annotations

from src.qec.analysis.basin_switch_detector import detect_basin_switch


def test_detects_metastable_switch() -> None:
    prev_sig = {"spectral_radius": 1.2, "mode_ipr": 0.10}
    new_sig = {"spectral_radius": 1.2000000001, "mode_ipr": 0.20}
    assert detect_basin_switch(prev_sig, new_sig, spectral_radius_eps=1e-6, ipr_delta_threshold=0.05) == "metastable_switch"


def test_detects_stable_transition() -> None:
    prev_sig = {"spectral_radius": 1.2, "mode_ipr": 0.10}
    new_sig = {"spectral_radius": 1.25, "mode_ipr": 0.12}
    assert detect_basin_switch(prev_sig, new_sig) == "stable"
