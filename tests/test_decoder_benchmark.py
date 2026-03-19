"""
v9.4.0 — Tests for the Discovery Decoder Benchmark.

Verifies:
- Benchmark table schema is correct.
- Success rate is within [0, 1].
- Iterations are positive.
- Results are deterministic across runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from qec.benchmark.discovery_benchmark import run_decoder_benchmark
from qec.benchmark.benchmark_table import build_benchmark_table


def _make_simple_H() -> np.ndarray:
    """Create a small parity-check matrix for testing."""
    return np.array([
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
    ], dtype=np.float64)


class TestRunDecoderBenchmark:
    """Tests for run_decoder_benchmark."""

    def test_returns_expected_keys(self):
        H = _make_simple_H()
        result = run_decoder_benchmark(H, trials=5, base_seed=42)
        assert "bp_success_rate" in result
        assert "avg_iterations" in result
        assert "trials" in result

    def test_success_rate_in_range(self):
        H = _make_simple_H()
        result = run_decoder_benchmark(H, trials=10, base_seed=42)
        assert 0.0 <= result["bp_success_rate"] <= 1.0

    def test_avg_iterations_positive(self):
        H = _make_simple_H()
        result = run_decoder_benchmark(H, trials=10, base_seed=42)
        assert result["avg_iterations"] >= 0.0

    def test_trials_count_matches(self):
        H = _make_simple_H()
        result = run_decoder_benchmark(H, trials=7, base_seed=42)
        assert result["trials"] == 7

    def test_deterministic(self):
        H = _make_simple_H()
        r1 = run_decoder_benchmark(H, trials=10, base_seed=42)
        r2 = run_decoder_benchmark(H, trials=10, base_seed=42)
        assert r1["bp_success_rate"] == r2["bp_success_rate"]
        assert r1["avg_iterations"] == r2["avg_iterations"]

    def test_different_seeds_may_differ(self):
        H = _make_simple_H()
        r1 = run_decoder_benchmark(H, trials=20, base_seed=0)
        r2 = run_decoder_benchmark(H, trials=20, base_seed=999)
        # Results may or may not differ, but both must be valid
        assert 0.0 <= r1["bp_success_rate"] <= 1.0
        assert 0.0 <= r2["bp_success_rate"] <= 1.0


class TestBuildBenchmarkTable:
    """Tests for build_benchmark_table."""

    def test_table_schema(self):
        H = _make_simple_H()
        graphs = [
            {"graph_id": "test_1", "H": H},
            {"graph_id": "test_2", "H": H},
        ]
        table = build_benchmark_table(graphs, trials=5, base_seed=42)

        assert len(table) == 2
        for entry in table:
            assert "graph_id" in entry
            assert "spectral_radius" in entry
            assert "bp_success_rate" in entry
            assert "avg_iterations" in entry

    def test_success_rate_in_range(self):
        H = _make_simple_H()
        graphs = [{"graph_id": "g1", "H": H}]
        table = build_benchmark_table(graphs, trials=10, base_seed=42)
        for entry in table:
            assert 0.0 <= entry["bp_success_rate"] <= 1.0

    def test_iterations_non_negative(self):
        H = _make_simple_H()
        graphs = [{"graph_id": "g1", "H": H}]
        table = build_benchmark_table(graphs, trials=10, base_seed=42)
        for entry in table:
            assert entry["avg_iterations"] >= 0.0

    def test_spectral_radius_non_negative(self):
        H = _make_simple_H()
        graphs = [{"graph_id": "g1", "H": H}]
        table = build_benchmark_table(graphs, trials=5, base_seed=42)
        for entry in table:
            assert entry["spectral_radius"] >= 0.0

    def test_graph_id_preserved(self):
        H = _make_simple_H()
        graphs = [
            {"graph_id": "alpha", "H": H},
            {"graph_id": "beta", "H": H},
        ]
        table = build_benchmark_table(graphs, trials=5, base_seed=42)
        ids = [e["graph_id"] for e in table]
        assert ids == ["alpha", "beta"]

    def test_deterministic(self):
        H = _make_simple_H()
        graphs = [{"graph_id": "g1", "H": H}]
        t1 = build_benchmark_table(graphs, trials=10, base_seed=42)
        t2 = build_benchmark_table(graphs, trials=10, base_seed=42)
        assert t1 == t2

    def test_empty_graphs(self):
        table = build_benchmark_table([], trials=5, base_seed=42)
        assert table == []
