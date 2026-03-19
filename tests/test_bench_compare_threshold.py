"""
Tests for the comparison suite — threshold crossing-point estimator.

Uses small synthetic FER curves so tests are fast and deterministic.
"""

import pytest

from bench.compare import (
    compute_threshold_table,
    compute_runtime_scaling,
    compute_iteration_histogram,
    aggregate_iteration_summaries,
)


def _make_records(decoder: str, distances: list[int],
                  p_values: list[float],
                  fer_fn) -> list[dict]:
    """Build synthetic benchmark records."""
    records = []
    for d in distances:
        for p in p_values:
            records.append({
                "decoder": decoder,
                "distance": d,
                "p": p,
                "fer": fer_fn(d, p),
                "wer": fer_fn(d, p),
                "mean_iters": 10.0,
                "runtime": None,
            })
    return records


class TestThresholdTable:

    def test_crossing_detected(self):
        """When FER curves cross, threshold should be estimated."""
        # Simple synthetic: FER = p * d for small d, crosses near p=0.1.
        # d=3: FER = 3*p (increases fast)
        # d=5: FER = max(0, 5*p - 0.1) (starts lower, crosses d=3)
        def fer_fn(d, p):
            if d == 3:
                return min(1.0, 3 * p)
            else:
                return min(1.0, max(0.0, 5 * p - 0.1))

        ps = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        records = _make_records("test_dec", [3, 5], ps, fer_fn)
        result = compute_threshold_table(records, "test_dec")

        assert result["decoder"] == "test_dec"
        assert result["method"] == "crossing_point"
        assert result["threshold_estimate_p"] is not None
        assert 0.0 < result["threshold_estimate_p"] < 1.0
        assert len(result["crossings"]) > 0

    def test_no_crossing(self):
        """When FER curves never cross, threshold should be None."""
        # d=3 always has higher FER than d=5.
        def fer_fn(d, p):
            return p / d

        ps = [0.01, 0.05, 0.10]
        records = _make_records("test_dec", [3, 5], ps, fer_fn)
        result = compute_threshold_table(records, "test_dec")

        assert result["threshold_estimate_p"] is None
        assert len(result["crossings"]) == 0

    def test_deterministic(self):
        def fer_fn(d, p):
            if d == 3:
                return 3 * p
            return max(0.0, 5 * p - 0.1)

        ps = [0.01, 0.05, 0.10]
        records = _make_records("test", [3, 5], ps, fer_fn)

        r1 = compute_threshold_table(records, "test")
        r2 = compute_threshold_table(records, "test")
        assert r1 == r2

    def test_single_distance(self):
        """Single distance → no crossings possible."""
        def fer_fn(d, p):
            return p

        ps = [0.01, 0.05]
        records = _make_records("test", [3], ps, fer_fn)
        result = compute_threshold_table(records, "test")
        assert result["threshold_estimate_p"] is None


class TestRuntimeScaling:

    def test_scaling_with_runtime(self):
        records = [
            {"decoder": "d", "distance": 3, "p": 0.01, "fer": 0.1,
             "runtime": {"average_latency_us": 100}},
            {"decoder": "d", "distance": 5, "p": 0.01, "fer": 0.1,
             "runtime": {"average_latency_us": 300}},
            {"decoder": "d", "distance": 7, "p": 0.01, "fer": 0.1,
             "runtime": {"average_latency_us": 700}},
        ]
        result = compute_runtime_scaling(records, "d")
        assert result["decoder"] == "d"
        assert len(result["points"]) == 3
        assert result["slope"] is not None

    def test_no_runtime(self):
        records = [
            {"decoder": "d", "distance": 3, "p": 0.01, "fer": 0.1},
            {"decoder": "d", "distance": 5, "p": 0.01, "fer": 0.1},
        ]
        result = compute_runtime_scaling(records, "d")
        assert result["slope"] is None
        assert len(result["points"]) == 0

    def test_deterministic(self):
        records = [
            {"decoder": "d", "distance": 3, "p": 0.01, "fer": 0.1,
             "runtime": {"average_latency_us": 100}},
            {"decoder": "d", "distance": 5, "p": 0.01, "fer": 0.1,
             "runtime": {"average_latency_us": 300}},
        ]
        r1 = compute_runtime_scaling(records, "d")
        r2 = compute_runtime_scaling(records, "d")
        assert r1 == r2

    def test_zero_latency_point_does_not_block_slope(self):
        """A zero-latency point should be skipped, not prevent slope
        estimation when sufficient positive-latency points remain."""
        records = [
            {"decoder": "d", "distance": 3, "p": 0.01, "fer": 0.1,
             "runtime": {"average_latency_us": 0}},
            {"decoder": "d", "distance": 5, "p": 0.01, "fer": 0.1,
             "runtime": {"average_latency_us": 300}},
            {"decoder": "d", "distance": 7, "p": 0.01, "fer": 0.1,
             "runtime": {"average_latency_us": 700}},
        ]
        result = compute_runtime_scaling(records, "d")
        assert len(result["points"]) == 3
        assert result["slope"] is not None


class TestIterationHistogram:

    def test_basic_histogram(self):
        counts = [5, 5, 10, 10, 10, 15]
        result = compute_iteration_histogram(counts)
        assert result["mean_iters"] == pytest.approx(55.0 / 6, rel=1e-3)
        assert result["histogram"]["iters"] == [5, 10, 15]
        assert result["histogram"]["counts"] == [2, 3, 1]

    def test_empty(self):
        result = compute_iteration_histogram([])
        assert result["mean_iters"] == 0.0
        assert result["histogram"]["iters"] == []
        assert result["histogram"]["counts"] == []

    def test_single_value(self):
        result = compute_iteration_histogram([7, 7, 7])
        assert result["mean_iters"] == 7.0
        assert result["histogram"]["iters"] == [7]
        assert result["histogram"]["counts"] == [3]

    def test_deterministic(self):
        counts = [1, 3, 2, 1, 3]
        r1 = compute_iteration_histogram(counts)
        r2 = compute_iteration_histogram(counts)
        assert r1 == r2


class TestAggregateIterationSummaries:

    def test_extracts_histograms(self):
        records = [
            {"decoder": "d", "distance": 3, "p": 0.01, "mean_iters": 5.0,
             "iter_histogram": {"iters": [5], "counts": [10]}},
            {"decoder": "d", "distance": 3, "p": 0.02, "mean_iters": 6.0},
        ]
        summaries = aggregate_iteration_summaries(records)
        assert len(summaries) == 1
        assert summaries[0]["p"] == 0.01
