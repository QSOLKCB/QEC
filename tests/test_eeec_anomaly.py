"""
Tests for EEEC anomaly detection (v7.6.1).

Verifies:
  - anomaly detection logic with known thresholds
  - batch scan produces correct counts
  - deterministic output
  - JSON-serializable results
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from qec.experiments.eeec_anomaly_scan import (
    DEFAULT_EEEC_THRESHOLD,
    DEFAULT_IPR_THRESHOLD,
    DEFAULT_RADIUS_THRESHOLD,
    detect_eeec_anomaly,
    run_eeec_anomaly_scan,
)


# ── Fixtures ─────────────────────────────────────────────────────


def _small_H():
    """3x4 parity-check matrix."""
    return np.array([
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
    ], dtype=np.float64)


# ── detect_eeec_anomaly tests ────────────────────────────────────


class TestDetectEeecAnomaly:
    """Tests for single-matrix anomaly detection."""

    def test_returns_required_keys(self):
        H = _small_H()
        result = detect_eeec_anomaly(H)
        required_keys = {
            "eeec_anomaly_detected",
            "spectral_radius",
            "eeec",
            "ipr",
            "radius_threshold",
            "eeec_threshold",
            "ipr_threshold",
        }
        assert required_keys == set(result.keys())

    def test_anomaly_detected_is_bool(self):
        H = _small_H()
        result = detect_eeec_anomaly(H)
        assert isinstance(result["eeec_anomaly_detected"], bool)

    def test_thresholds_stored(self):
        H = _small_H()
        result = detect_eeec_anomaly(H)
        assert result["radius_threshold"] == DEFAULT_RADIUS_THRESHOLD
        assert result["eeec_threshold"] == DEFAULT_EEEC_THRESHOLD
        assert result["ipr_threshold"] == DEFAULT_IPR_THRESHOLD

    def test_custom_thresholds(self):
        H = _small_H()
        result = detect_eeec_anomaly(
            H,
            radius_threshold=100.0,
            eeec_threshold=0.0,
            ipr_threshold=0.0,
        )
        # With very permissive thresholds, anomaly should be detected
        # (radius < 100, eeec > 0, ipr > 0) if graph has nonzero metrics
        assert result["radius_threshold"] == 100.0
        assert result["eeec_threshold"] == 0.0
        assert result["ipr_threshold"] == 0.0

    def test_deterministic(self):
        H = _small_H()
        r1 = detect_eeec_anomaly(H)
        r2 = detect_eeec_anomaly(H)
        assert r1 == r2

    def test_json_serializable(self):
        H = _small_H()
        result = detect_eeec_anomaly(H)
        s = json.dumps(result, sort_keys=True)
        parsed = json.loads(s)
        assert parsed == result

    def test_no_anomaly_high_radius(self):
        """With very low radius threshold, no anomaly should be detected."""
        H = _small_H()
        result = detect_eeec_anomaly(H, radius_threshold=0.0)
        assert result["eeec_anomaly_detected"] is False

    def test_no_anomaly_high_eeec_threshold(self):
        """With very high eeec threshold, no anomaly should be detected."""
        H = _small_H()
        result = detect_eeec_anomaly(H, eeec_threshold=2.0)
        assert result["eeec_anomaly_detected"] is False

    def test_no_anomaly_high_ipr_threshold(self):
        """With very high ipr threshold, no anomaly should be detected."""
        H = _small_H()
        result = detect_eeec_anomaly(H, ipr_threshold=2.0)
        assert result["eeec_anomaly_detected"] is False


# ── run_eeec_anomaly_scan tests ──────────────────────────────────


class TestEeecAnomalyScan:
    """Tests for batch anomaly scanning."""

    def test_returns_required_keys(self):
        matrices = [_small_H()]
        result = run_eeec_anomaly_scan(matrices)
        required_keys = {
            "num_matrices",
            "num_anomalies",
            "anomaly_indices",
            "results",
        }
        assert required_keys == set(result.keys())

    def test_num_matrices(self):
        matrices = [_small_H(), _small_H()]
        result = run_eeec_anomaly_scan(matrices)
        assert result["num_matrices"] == 2

    def test_results_length(self):
        matrices = [_small_H(), _small_H()]
        result = run_eeec_anomaly_scan(matrices)
        assert len(result["results"]) == 2

    def test_empty_input(self):
        result = run_eeec_anomaly_scan([])
        assert result["num_matrices"] == 0
        assert result["num_anomalies"] == 0
        assert result["anomaly_indices"] == []
        assert result["results"] == []

    def test_anomaly_count_consistency(self):
        matrices = [_small_H()]
        result = run_eeec_anomaly_scan(matrices)
        expected_count = sum(
            1 for r in result["results"] if r["eeec_anomaly_detected"]
        )
        assert result["num_anomalies"] == expected_count
        assert len(result["anomaly_indices"]) == expected_count

    def test_deterministic(self):
        matrices = [_small_H()]
        r1 = run_eeec_anomaly_scan(matrices)
        r2 = run_eeec_anomaly_scan(matrices)
        assert r1 == r2
