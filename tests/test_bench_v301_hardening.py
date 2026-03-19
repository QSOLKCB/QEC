"""
Tests for v3.0.1 determinism hardening:
- Sub-seed derivation independent of decoder order.
- deterministic_metadata mode for fully byte-identical output.
"""

import json

from bench.config import BenchmarkConfig, DecoderSpec
from bench.runner import run_benchmark, _derive_subseed
from bench.schema import dumps_result


# ── Issue 1: sub-seed independence ──────────────────────────────────

class TestSubseedDerivation:

    def test_derive_subseed_stable(self):
        """Same inputs must always produce the same sub-seed."""
        identity = {"adapter": "bp", "params": {"mode": "min_sum"}}
        s1 = _derive_subseed(42, identity, 5, 0.01)
        s2 = _derive_subseed(42, identity, 5, 0.01)
        assert s1 == s2

    def test_derive_subseed_varies_with_params(self):
        """Different parameters must produce different sub-seeds."""
        id_a = {"adapter": "bp", "params": {"mode": "min_sum"}}
        id_b = {"adapter": "bp", "params": {"mode": "sum_product"}}
        sa = _derive_subseed(42, id_a, 5, 0.01)
        sb = _derive_subseed(42, id_b, 5, 0.01)
        assert sa != sb

    def test_derive_subseed_is_32bit(self):
        identity = {"adapter": "bp", "params": {"mode": "min_sum"}}
        s = _derive_subseed(42, identity, 5, 0.01)
        assert 0 <= s < 2**32

    def test_subseed_independent_of_decoder_order(self):
        """Reordering decoders in config must NOT change per-record
        statistical results (FER, mean_iters, etc.).
        """
        dec_a = DecoderSpec(adapter="bp", params={
            "mode": "min_sum", "schedule": "flooding",
        })
        dec_b = DecoderSpec(adapter="bp", params={
            "mode": "sum_product", "schedule": "flooding",
        })

        config_ab = BenchmarkConfig(
            seed=42,
            distances=[3],
            p_values=[0.01],
            trials=10,
            max_iters=20,
            decoders=[dec_a, dec_b],
            runtime_mode="off",
            deterministic_metadata=True,
        )
        config_ba = BenchmarkConfig(
            seed=42,
            distances=[3],
            p_values=[0.01],
            trials=10,
            max_iters=20,
            decoders=[dec_b, dec_a],
            runtime_mode="off",
            deterministic_metadata=True,
        )

        r_ab = run_benchmark(config_ab)
        r_ba = run_benchmark(config_ba)

        # Build lookup: decoder_name -> record (excluding ordering fields).
        def _by_decoder(result):
            return {
                rec["decoder"]: {
                    "fer": rec["fer"],
                    "wer": rec["wer"],
                    "mean_iters": rec["mean_iters"],
                }
                for rec in result["results"]
            }

        ab_map = _by_decoder(r_ab)
        ba_map = _by_decoder(r_ba)

        assert ab_map == ba_map, (
            "Per-record results changed when decoder order was reversed"
        )


# ── Issue 2: deterministic_metadata mode ────────────────────────────

class TestDeterministicMetadata:

    def test_deterministic_metadata_mode(self):
        """With deterministic_metadata=True, two runs must produce
        fully byte-identical JSON without any masking.
        """
        config = BenchmarkConfig(
            seed=42,
            distances=[3],
            p_values=[0.01],
            trials=10,
            max_iters=20,
            decoders=[
                DecoderSpec(adapter="bp", params={
                    "mode": "min_sum", "schedule": "flooding",
                })
            ],
            runtime_mode="off",
            deterministic_metadata=True,
        )

        r1 = run_benchmark(config)
        r2 = run_benchmark(config)

        j1 = dumps_result(r1)
        j2 = dumps_result(r2)
        assert j1 == j2, "Output not byte-identical with deterministic_metadata=True"

    def test_deterministic_metadata_timestamp(self):
        """deterministic_metadata=True must use the epoch timestamp."""
        config = BenchmarkConfig(
            seed=42,
            distances=[3],
            p_values=[0.01],
            trials=5,
            max_iters=20,
            decoders=[
                DecoderSpec(adapter="bp", params={
                    "mode": "min_sum", "schedule": "flooding",
                })
            ],
            runtime_mode="off",
            deterministic_metadata=True,
        )
        result = run_benchmark(config)
        assert result["created_utc"] == "1970-01-01T00:00:00+00:00"

    def test_default_metadata_not_deterministic(self):
        """Default (deterministic_metadata=False) must NOT use epoch."""
        config = BenchmarkConfig(
            seed=42,
            distances=[3],
            p_values=[0.01],
            trials=5,
            max_iters=20,
            decoders=[
                DecoderSpec(adapter="bp", params={
                    "mode": "min_sum", "schedule": "flooding",
                })
            ],
            runtime_mode="off",
        )
        result = run_benchmark(config)
        assert result["created_utc"] != "1970-01-01T00:00:00+00:00"

    def test_default_timestamp_no_microseconds(self):
        """Default timestamp must not include microseconds."""
        config = BenchmarkConfig(
            seed=42,
            distances=[3],
            p_values=[0.01],
            trials=5,
            max_iters=20,
            decoders=[
                DecoderSpec(adapter="bp", params={
                    "mode": "min_sum", "schedule": "flooding",
                })
            ],
            runtime_mode="off",
        )
        result = run_benchmark(config)
        ts = result["created_utc"]
        # ISO8601 with microseconds has a '.' before the fractional part.
        assert "." not in ts, f"Timestamp contains microseconds: {ts}"
