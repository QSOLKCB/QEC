"""Tests for v13.2.0 spectral instability score."""

from __future__ import annotations

import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.qec.analysis.localization_metrics import SpectralInstabilityScore


def test_sis_formula_deterministic():
    d1 = SpectralInstabilityScore.compute(lambda_min=-0.2, ipr=0.4, entropy=2.0)
    d2 = SpectralInstabilityScore.compute(lambda_min=-0.2, ipr=0.4, entropy=2.0)
    assert d1 == d2
    assert d1["spectral_instability_score"] == 0.04


def test_sis_zero_entropy_safe():
    d = SpectralInstabilityScore.compute(lambda_min=-0.2, ipr=0.4, entropy=0.0)
    assert d["spectral_instability_score"] == 0.0
