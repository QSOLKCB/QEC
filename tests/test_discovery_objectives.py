"""
Tests for v9.0.0 discovery objectives.

Verifies:
  - objective computation is deterministic
  - all expected keys are present
  - composite score is finite
  - IPR localization and basin-switch risk are in valid ranges
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.discovery.objectives import (
    compute_discovery_objectives,
    compute_ipr_localization,
    compute_basin_switch_risk,
)


def _small_H():
    """3x4 parity-check matrix with known structure."""
    return np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
    ], dtype=np.float64)


class TestDiscoveryObjectives:
    def test_returns_all_keys(self):
        H = _small_H()
        obj = compute_discovery_objectives(H)
        required = {
            "instability_score", "spectral_radius", "bethe_margin",
            "cycle_density", "entropy", "curvature",
            "ipr_localization", "basin_switch_risk", "composite_score",
        }
        assert required == set(obj.keys())

    def test_deterministic(self):
        H = _small_H()
        o1 = compute_discovery_objectives(H, seed=42)
        o2 = compute_discovery_objectives(H, seed=42)
        for key in o1:
            assert o1[key] == o2[key], f"Non-deterministic: {key}"

    def test_composite_score_finite(self):
        H = _small_H()
        obj = compute_discovery_objectives(H)
        assert np.isfinite(obj["composite_score"])

    def test_spectral_radius_positive(self):
        H = _small_H()
        obj = compute_discovery_objectives(H)
        assert obj["spectral_radius"] > 0


class TestIPRLocalization:
    def test_returns_positive(self):
        H = _small_H()
        ipr = compute_ipr_localization(H)
        assert ipr > 0

    def test_deterministic(self):
        H = _small_H()
        i1 = compute_ipr_localization(H)
        i2 = compute_ipr_localization(H)
        assert i1 == i2

    def test_bounded(self):
        H = _small_H()
        ipr = compute_ipr_localization(H)
        assert 0 < ipr <= 1.0


class TestBasinSwitchRisk:
    def test_returns_bounded(self):
        H = _small_H()
        risk = compute_basin_switch_risk(H, seed=42)
        assert 0.0 <= risk <= 1.0

    def test_deterministic(self):
        H = _small_H()
        r1 = compute_basin_switch_risk(H, seed=42)
        r2 = compute_basin_switch_risk(H, seed=42)
        assert r1 == r2
