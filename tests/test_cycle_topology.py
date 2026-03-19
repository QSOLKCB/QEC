"""
Tests for v11.3.0 — Cycle Topology Analyzer.

Covers:
  - deterministic output
  - metrics within valid ranges
  - no input mutation
  - empty matrix handling
"""

import numpy as np
import pytest

from qec.analysis.cycle_topology import CycleTopologyAnalyzer


def _make_small_ldpc():
    """Small (4, 8) LDPC-like matrix."""
    H = np.array([
        [1, 1, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 0],
        [1, 0, 0, 1, 0, 0, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1],
    ], dtype=np.float64)
    return H


def _make_no_cycles():
    """Tree-like matrix with no short cycles."""
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ], dtype=np.float64)
    return H


class TestCycleTopologyAnalyzer:
    def test_deterministic(self):
        H = _make_small_ldpc()
        analyzer = CycleTopologyAnalyzer()
        r1 = analyzer.analyze(H)
        r2 = analyzer.analyze(H)

        assert r1["num_short_cycles"] == r2["num_short_cycles"]
        assert r1["cycle_parity_histogram"] == r2["cycle_parity_histogram"]
        assert r1["mean_degree_twist"] == r2["mean_degree_twist"]
        assert r1["max_degree_twist"] == r2["max_degree_twist"]
        assert r1["twisted_cycle_fraction"] == r2["twisted_cycle_fraction"]

    def test_valid_ranges(self):
        H = _make_small_ldpc()
        analyzer = CycleTopologyAnalyzer()
        result = analyzer.analyze(H)

        assert result["num_short_cycles"] >= 0
        assert result["mean_degree_twist"] >= 0.0
        assert result["max_degree_twist"] >= 0.0
        assert 0.0 <= result["twisted_cycle_fraction"] <= 1.0
        assert np.isfinite(result["mean_degree_twist"])
        assert np.isfinite(result["max_degree_twist"])

    def test_no_input_mutation(self):
        H = _make_small_ldpc()
        H_copy = H.copy()
        analyzer = CycleTopologyAnalyzer()
        analyzer.analyze(H)

        np.testing.assert_array_equal(H, H_copy)

    def test_empty_matrix(self):
        H = np.zeros((0, 0), dtype=np.float64)
        analyzer = CycleTopologyAnalyzer()
        result = analyzer.analyze(H)

        assert result["num_short_cycles"] == 0
        assert result["cycle_parity_histogram"] == {}
        assert result["mean_degree_twist"] == 0.0
        assert result["max_degree_twist"] == 0.0
        assert result["twisted_cycle_fraction"] == 0.0

    def test_no_cycles_matrix(self):
        H = _make_no_cycles()
        analyzer = CycleTopologyAnalyzer()
        result = analyzer.analyze(H)

        assert result["num_short_cycles"] == 0
        assert result["mean_degree_twist"] == 0.0

    def test_result_keys_present(self):
        H = _make_small_ldpc()
        analyzer = CycleTopologyAnalyzer()
        result = analyzer.analyze(H)

        assert "num_short_cycles" in result
        assert "cycle_parity_histogram" in result
        assert "mean_degree_twist" in result
        assert "max_degree_twist" in result
        assert "twisted_cycle_fraction" in result

    def test_histogram_values_nonnegative(self):
        H = _make_small_ldpc()
        analyzer = CycleTopologyAnalyzer()
        result = analyzer.analyze(H)

        for length, count in result["cycle_parity_histogram"].items():
            assert count > 0
            assert isinstance(length, int)
            assert length > 0

    def test_max_twist_geq_mean_twist(self):
        H = _make_small_ldpc()
        analyzer = CycleTopologyAnalyzer()
        result = analyzer.analyze(H)

        assert result["max_degree_twist"] >= result["mean_degree_twist"]

    def test_histogram_sum_equals_num_cycles(self):
        H = _make_small_ldpc()
        analyzer = CycleTopologyAnalyzer()
        result = analyzer.analyze(H)

        total = sum(result["cycle_parity_histogram"].values())
        assert total == result["num_short_cycles"]
