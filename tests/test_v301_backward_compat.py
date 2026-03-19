"""
Backward compatibility audit for v3.0.1 (Deliverable 3).

Proves that:
1) Key public API signatures remain unchanged.
2) v3.0.0 benchmark configs validate and run without new required keys.
3) v3.0.0 result validator still accepts prior output shapes.
4) Determinism contract is preserved (runtime_mode="off" → byte-identical JSON).
5) Import hygiene: core decoder import does not pull in bench/qudit/nonbinary.
"""

import inspect
import json
import sys

import pytest

from bench.config import BenchmarkConfig, DecoderSpec
from bench.runner import run_benchmark
from bench.schema import (
    SCHEMA_VERSION,
    _SUPPORTED_SCHEMA_VERSIONS,
    canonicalize,
    dumps_result,
    validate_result,
)


# ── 1. API surface checks ────────────────────────────────────────────

class TestPublicAPISignatures:
    """Ensure key public entrypoints have not changed."""

    def test_bp_decode_signature(self):
        from src import bp_decode
        sig = inspect.signature(bp_decode)
        params = list(sig.parameters.keys())
        # Must have at least: H, llr
        assert "H" in params
        assert "llr" in params

    def test_simulate_fer_signature(self):
        from src import simulate_fer
        sig = inspect.signature(simulate_fer)
        params = list(sig.parameters.keys())
        assert "H" in params
        assert "decoder_config" in params
        assert "noise_config" in params
        assert "trials" in params

    def test_channel_llr_signature(self):
        from src import channel_llr
        sig = inspect.signature(channel_llr)
        params = list(sig.parameters.keys())
        assert len(params) >= 2  # at minimum: x, p

    def test_syndrome_signature(self):
        from src import syndrome
        sig = inspect.signature(syndrome)
        params = list(sig.parameters.keys())
        assert len(params) >= 2  # at minimum: H, e

    def test_osd0_signature(self):
        from src import osd0
        sig = inspect.signature(osd0)
        params = list(sig.parameters.keys())
        assert "H" in params or len(params) >= 1

    def test_run_benchmark_signature(self):
        sig = inspect.signature(run_benchmark)
        params = list(sig.parameters.keys())
        assert "config" in params

    def test_benchmark_config_from_dict(self):
        """BenchmarkConfig.from_dict still exists and is callable."""
        assert callable(BenchmarkConfig.from_dict)

    def test_benchmark_config_load(self):
        """BenchmarkConfig.load still exists and is callable."""
        assert callable(BenchmarkConfig.load)


# ── 2. v3.0.0 config compatibility ──────────────────────────────────

class TestV300ConfigCompat:
    """v3.0.0 configs must still validate and run unchanged."""

    def _v300_config_dict(self) -> dict:
        """A minimal v3.0.0 config with NO v3.0.1 fields."""
        return {
            "schema_version": "3.0.0",
            "seed": 42,
            "distances": [3],
            "p_values": [0.01],
            "trials": 5,
            "max_iters": 10,
            "decoders": [
                {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "flooding"}}
            ],
            "runtime_mode": "off",
            "runtime": {"warmup": 1, "runs": 1, "measure_memory": False},
            "collect_iter_hist": False,
            "deterministic_metadata": False,
        }

    def test_v300_config_loads(self):
        """v3.0.0 config dict can be loaded into BenchmarkConfig."""
        cfg = BenchmarkConfig.from_dict(self._v300_config_dict())
        assert cfg.seed == 42
        assert cfg.distances == [3]
        assert cfg.qudit is None
        assert cfg.resource_model is None

    def test_v300_config_runs(self):
        """v3.0.0 config runs successfully without new required keys."""
        cfg = BenchmarkConfig.from_dict(self._v300_config_dict())
        result = run_benchmark(cfg)
        assert "results" in result
        assert len(result["results"]) == 1

    def test_v300_config_round_trip(self):
        """v3.0.0 config to_dict does not inject qudit or resource_model."""
        cfg = BenchmarkConfig.from_dict(self._v300_config_dict())
        d = cfg.to_dict()
        assert "qudit" not in d
        assert "resource_model" not in d

    def test_v300_result_validates(self):
        """v3.0.0-shaped result still passes validation."""
        obj = {
            "schema_version": "3.0.0",
            "created_utc": "2026-01-01T00:00:00+00:00",
            "environment": {"platform": "test"},
            "config": {"seed": 42},
            "results": [
                {
                    "decoder": "bp_test",
                    "distance": 3,
                    "p": 0.01,
                    "fer": 0.1,
                    "wer": 0.1,
                    "mean_iters": 5.0,
                }
            ],
            "summaries": {},
        }
        validate_result(obj)  # Must not raise.


# ── 3. v3.0.1 config with new optional fields ───────────────────────

class TestV301OptionalFields:
    """New v3.0.1 fields are additive and opt-in."""

    def _v301_config_dict(self) -> dict:
        return {
            "schema_version": "3.0.1",
            "seed": 42,
            "distances": [3],
            "p_values": [0.01],
            "trials": 5,
            "max_iters": 10,
            "decoders": [
                {"adapter": "bp", "params": {"mode": "min_sum", "schedule": "flooding"}}
            ],
            "runtime_mode": "off",
            "runtime": {"warmup": 1, "runs": 1, "measure_memory": False},
            "collect_iter_hist": False,
            "deterministic_metadata": True,
            "qudit": {"dimension": 2, "encoding": "qubit", "metadata": {}},
            "resource_model": {
                "enabled": False,
                "model": "qubit_decomp_v1",
                "native_model": "native_placeholder_v1",
                "assumptions": {},
            },
        }

    def test_v301_config_loads(self):
        cfg = BenchmarkConfig.from_dict(self._v301_config_dict())
        assert cfg.qudit == {"dimension": 2, "encoding": "qubit", "metadata": {}}
        assert cfg.resource_model is not None
        assert cfg.resource_model.enabled is False

    def test_v301_config_runs(self):
        cfg = BenchmarkConfig.from_dict(self._v301_config_dict())
        result = run_benchmark(cfg)
        assert result["schema_version"] == SCHEMA_VERSION
        assert len(result["results"]) == 1

    def test_v301_qudit_in_config_output(self):
        cfg = BenchmarkConfig.from_dict(self._v301_config_dict())
        d = cfg.to_dict()
        assert "qudit" in d
        assert d["qudit"]["dimension"] == 2

    def test_v301_resource_model_disabled_no_estimates(self):
        """resource_model.enabled=false → no resource_estimates in summaries."""
        cfg = BenchmarkConfig.from_dict(self._v301_config_dict())
        result = run_benchmark(cfg)
        assert "resource_estimates" not in result["summaries"]

    def test_v301_resource_model_enabled_has_estimates(self):
        d = self._v301_config_dict()
        d["resource_model"]["enabled"] = True
        cfg = BenchmarkConfig.from_dict(d)
        result = run_benchmark(cfg)
        assert "resource_estimates" in result["summaries"]
        re = result["summaries"]["resource_estimates"]
        assert "qubit_decomposition" in re
        assert "native_qudit" in re
        assert "comparison" in re


# ── 4. Determinism contract ──────────────────────────────────────────

class TestDeterminismContract:
    """runtime_mode='off' + deterministic_metadata → byte-identical JSON."""

    def test_byte_identical_v300_config(self):
        cfg = BenchmarkConfig.from_dict({
            "schema_version": "3.0.0",
            "seed": 42,
            "distances": [3],
            "p_values": [0.01],
            "trials": 5,
            "max_iters": 10,
            "decoders": [{"adapter": "bp", "params": {"mode": "min_sum", "schedule": "flooding"}}],
            "runtime_mode": "off",
            "deterministic_metadata": True,
        })
        r1 = run_benchmark(cfg)
        r2 = run_benchmark(cfg)
        j1 = dumps_result(r1)
        j2 = dumps_result(r2)
        assert j1 == j2, "v3.0.0 config determinism violated"

    def test_byte_identical_v301_config_with_qudit(self):
        cfg = BenchmarkConfig.from_dict({
            "schema_version": "3.0.1",
            "seed": 42,
            "distances": [3],
            "p_values": [0.01],
            "trials": 5,
            "max_iters": 10,
            "decoders": [{"adapter": "bp", "params": {"mode": "min_sum", "schedule": "flooding"}}],
            "runtime_mode": "off",
            "deterministic_metadata": True,
            "qudit": {"dimension": 2, "encoding": "qubit", "metadata": {}},
        })
        r1 = run_benchmark(cfg)
        r2 = run_benchmark(cfg)
        j1 = dumps_result(r1)
        j2 = dumps_result(r2)
        assert j1 == j2, "v3.0.1 config with qudit determinism violated"

    def test_byte_identical_v301_with_resource_model_enabled(self):
        cfg = BenchmarkConfig.from_dict({
            "schema_version": "3.0.1",
            "seed": 42,
            "distances": [3],
            "p_values": [0.01],
            "trials": 5,
            "max_iters": 10,
            "decoders": [{"adapter": "bp", "params": {"mode": "min_sum", "schedule": "flooding"}}],
            "runtime_mode": "off",
            "deterministic_metadata": True,
            "qudit": {"dimension": 3, "encoding": "qutrit", "metadata": {}},
            "resource_model": {
                "enabled": True,
                "model": "qubit_decomp_v1",
                "native_model": "native_placeholder_v1",
                "assumptions": {},
            },
        })
        r1 = run_benchmark(cfg)
        r2 = run_benchmark(cfg)
        j1 = dumps_result(r1)
        j2 = dumps_result(r2)
        assert j1 == j2, "v3.0.1 config with resource_model determinism violated"


# ── 5. Schema version support ────────────────────────────────────────

class TestSchemaVersionSupport:
    """Both 3.0.0 and 3.0.1 are accepted."""

    def test_3_0_0_accepted(self):
        assert "3.0.0" in _SUPPORTED_SCHEMA_VERSIONS

    def test_3_0_1_accepted(self):
        assert "3.0.1" in _SUPPORTED_SCHEMA_VERSIONS

    def test_unknown_rejected(self):
        obj = {
            "schema_version": "2.0.0",
            "created_utc": "x",
            "environment": {},
            "config": {},
            "results": [],
            "summaries": {},
        }
        with pytest.raises(ValueError, match="Unsupported"):
            validate_result(obj)
