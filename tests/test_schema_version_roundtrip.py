"""
Regression test: schema_version roundtrip through run_benchmark.

Locks in the invariant that the top-level ``schema_version`` in the
benchmark result must match the ``schema_version`` supplied in the
input config.  A v3.0.0 config must produce a v3.0.0-stamped result;
a v3.0.1 config must produce a v3.0.1-stamped result.
"""

from bench.config import BenchmarkConfig, DecoderSpec
from bench.runner import run_benchmark


def _minimal_config(schema_version: str) -> BenchmarkConfig:
    """Build the smallest valid config for a given schema version."""
    return BenchmarkConfig(
        schema_version=schema_version,
        seed=1,
        distances=[3],
        p_values=[0.01],
        trials=5,
        max_iters=10,
        decoders=[
            DecoderSpec(adapter="bp", params={
                "mode": "min_sum",
                "schedule": "flooding",
            })
        ],
        runtime_mode="off",
        deterministic_metadata=True,
    )


class TestSchemaVersionRoundtrip:

    def test_v300_config_produces_v300_output(self):
        """A v3.0.0 config must emit schema_version '3.0.0' at top level."""
        config = _minimal_config("3.0.0")
        result = run_benchmark(config)

        assert result["schema_version"] == "3.0.0"
        assert result["config"]["schema_version"] == "3.0.0"
        assert result["schema_version"] == result["config"]["schema_version"]

    def test_v301_config_produces_v301_output(self):
        """A v3.0.1 config must emit schema_version '3.0.1' at top level."""
        config = _minimal_config("3.0.1")
        result = run_benchmark(config)

        assert result["schema_version"] == "3.0.1"
        assert result["config"]["schema_version"] == "3.0.1"
        assert result["schema_version"] == result["config"]["schema_version"]

    def test_top_level_and_config_versions_always_match(self):
        """Top-level and config schema_version must agree for every
        supported version."""
        for version in ("3.0.0", "3.0.1"):
            config = _minimal_config(version)
            result = run_benchmark(config)
            assert result["schema_version"] == version, (
                f"Top-level schema_version is {result['schema_version']!r}, "
                f"expected {version!r}"
            )
            assert result["config"]["schema_version"] == version, (
                f"config.schema_version is "
                f"{result['config']['schema_version']!r}, "
                f"expected {version!r}"
            )
