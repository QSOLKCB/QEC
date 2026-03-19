"""
Tests for benchmark runner determinism.

Verifies that running the same config twice with runtime_mode="off"
produces byte-identical JSON output.
"""

import json

import pytest

from bench.config import BenchmarkConfig, DecoderSpec
from bench.runner import run_benchmark
from bench.schema import dumps_result, validate_result


def _small_config(**overrides) -> BenchmarkConfig:
    """Build a minimal config for fast tests."""
    defaults = dict(
        seed=42,
        distances=[3],
        p_values=[0.01, 0.02],
        trials=10,
        max_iters=20,
        decoders=[
            DecoderSpec(adapter="bp", params={
                "mode": "min_sum",
                "schedule": "flooding",
            })
        ],
        runtime_mode="off",
        collect_iter_hist=False,
    )
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


class TestRunnerDeterminism:

    def test_identical_json_no_runtime(self):
        """Two runs with same config must produce byte-identical JSON
        (excluding created_utc which contains a timestamp).
        """
        config = _small_config()

        r1 = run_benchmark(config)
        r2 = run_benchmark(config)

        # Mask the timestamp for comparison.
        r1["created_utc"] = "MASKED"
        r2["created_utc"] = "MASKED"

        j1 = dumps_result(r1)
        j2 = dumps_result(r2)
        assert j1 == j2, "Non-deterministic output detected"

    def test_schema_valid(self):
        config = _small_config()
        result = run_benchmark(config)
        # Must not raise.
        validate_result(result)

    def test_result_structure(self):
        config = _small_config()
        result = run_benchmark(config)

        assert result["schema_version"] in ("3.0.0", "3.0.1")
        assert "created_utc" in result
        assert "environment" in result
        assert "config" in result
        assert "results" in result
        assert "summaries" in result
        assert isinstance(result["results"], list)
        assert len(result["results"]) > 0

    def test_result_record_fields(self):
        config = _small_config()
        result = run_benchmark(config)
        rec = result["results"][0]

        assert "decoder" in rec
        assert "distance" in rec
        assert "p" in rec
        assert "fer" in rec
        assert "wer" in rec
        assert "mean_iters" in rec
        assert rec["wer"] == rec["fer"]  # WER := FER

    def test_runtime_none_when_off(self):
        config = _small_config(runtime_mode="off")
        result = run_benchmark(config)
        for rec in result["results"]:
            assert rec["runtime"] is None

    def test_iter_histogram_present_when_enabled(self):
        config = _small_config(collect_iter_hist=True)
        result = run_benchmark(config)
        for rec in result["results"]:
            assert "iter_histogram" in rec
            assert "iters" in rec["iter_histogram"]
            assert "counts" in rec["iter_histogram"]

    def test_iter_histogram_absent_when_disabled(self):
        config = _small_config(collect_iter_hist=False)
        result = run_benchmark(config)
        for rec in result["results"]:
            assert "iter_histogram" not in rec

    def test_summaries_thresholds_present(self):
        config = _small_config()
        result = run_benchmark(config)
        assert "thresholds" in result["summaries"]
        assert isinstance(result["summaries"]["thresholds"], list)

    def test_sweep_ordering_deterministic(self):
        """Results must appear in decoder → distance → p order."""
        config = _small_config(
            distances=[3, 5],
            p_values=[0.01, 0.02],
        )
        result = run_benchmark(config)
        records = result["results"]
        # Should be: (3, 0.01), (3, 0.02), (5, 0.01), (5, 0.02)
        expected_pairs = [(3, 0.01), (3, 0.02), (5, 0.01), (5, 0.02)]
        actual_pairs = [(r["distance"], r["p"]) for r in records]
        assert actual_pairs == expected_pairs

    def test_json_serializable(self):
        config = _small_config()
        result = run_benchmark(config)
        text = dumps_result(result)
        # Must round-trip through JSON.
        parsed = json.loads(text)
        assert parsed["schema_version"] in ("3.0.0", "3.0.1")


class TestRunnerRuntime:

    def test_runtime_on(self):
        config = _small_config(
            runtime_mode="on",
            distances=[3],
            p_values=[0.01],
            trials=5,
        )
        config.runtime.warmup = 2
        config.runtime.runs = 3
        result = run_benchmark(config)
        rec = result["results"][0]
        rt = rec["runtime"]
        assert rt is not None
        assert "average_latency_us" in rt
        assert isinstance(rt["average_latency_us"], int)
        assert rt["average_latency_us"] >= 0
        assert "throughput_mhz" in rt
        assert "confidence_interval_us" in rt


class TestMultipleDecoders:

    def test_two_decoders(self):
        config = _small_config(
            distances=[3],
            p_values=[0.01],
            trials=5,
            decoders=[
                DecoderSpec(adapter="bp", params={
                    "mode": "min_sum", "schedule": "flooding",
                }),
                DecoderSpec(adapter="bp", params={
                    "mode": "sum_product", "schedule": "flooding",
                }),
            ],
        )
        result = run_benchmark(config)
        # Should have 2 records (2 decoders × 1 distance × 1 p).
        assert len(result["results"]) == 2
        names = [r["decoder"] for r in result["results"]]
        assert len(set(names)) == 2  # distinct decoder names
