"""
Tests for v3.1.3 channel abstraction layer.

Covers:
  A) Oracle identity — OracleChannel produces identical output to channel_llr().
  B) Non-degenerate FER — bsc_syndrome produces 0 < FER < 1.
  C) Determinism under BSC — two runs with same config are byte-identical.
  D) LLR structural tests — oracle sign depends on error_vector; BSC is uniform.
  E) Config backward compatibility — oracle default omits channel_model from dict.
"""

import json

import numpy as np
import pytest

from src.qec_qldpc_codes import channel_llr
from src.qec.channel import ChannelModel, OracleChannel, BSCSyndromeChannel
from src.bench.config import BenchmarkConfig, DecoderSpec
from src.bench.runner import run_benchmark
from src.bench.schema import dumps_result, validate_result


# --- Test tuning constants (v68.4 optimization) ---
_BSC_FER_TRIALS = 50
_BSC_FER_MAX_ITERS = 30

# ── Helpers ────────────────────────────────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════════════
# A) Oracle Identity Tests
# ═══════════════════════════════════════════════════════════════════════

class TestOracleIdentity:

    def test_oracle_matches_channel_llr(self):
        """OracleChannel.compute_llr must be numerically identical to
        the existing channel_llr() function for all inputs."""
        rng = np.random.default_rng(99)
        for n in [6, 12, 24]:
            for p in [0.01, 0.05, 0.1, 0.3]:
                e = (rng.random(n) < p).astype(np.uint8)
                expected = channel_llr(e, p)
                oracle = OracleChannel()
                actual = oracle.compute_llr(p=p, n=n, error_vector=e)
                np.testing.assert_array_equal(
                    actual, expected,
                    err_msg=f"Mismatch at n={n}, p={p}",
                )

    def test_oracle_requires_error_vector(self):
        """OracleChannel must raise if error_vector is None."""
        oracle = OracleChannel()
        with pytest.raises(ValueError, match="error_vector"):
            oracle.compute_llr(p=0.05, n=10, error_vector=None)

    def test_oracle_validates_p(self):
        """OracleChannel must raise for p outside (0, 1)."""
        oracle = OracleChannel()
        e = np.zeros(5, dtype=np.uint8)
        for bad_p in [0.0, 1.0, -0.1, 1.5]:
            with pytest.raises(ValueError, match="p must be in"):
                oracle.compute_llr(p=bad_p, n=5, error_vector=e)

    def test_oracle_benchmark_identity(self):
        """Oracle benchmark output must be byte-identical to baseline
        (run without channel_model field)."""
        config_oracle = _small_config(
            deterministic_metadata=True,
            channel_model="oracle",
        )
        config_default = _small_config(
            deterministic_metadata=True,
        )
        r_oracle = run_benchmark(config_oracle)
        r_default = run_benchmark(config_default)

        j_oracle = dumps_result(r_oracle)
        j_default = dumps_result(r_default)
        assert j_oracle == j_default, "Oracle mode must be byte-identical to default"


# ═══════════════════════════════════════════════════════════════════════
# B) Non-Degenerate FER Tests
# ═══════════════════════════════════════════════════════════════════════

class TestBSCSyndromeFER:

    def test_bsc_nonzero_fer(self):
        """BSC syndrome channel must produce 0 < FER < 1 at moderate noise."""
        config = _small_config(
            distances=[5],
            p_values=[0.05],
            trials=_BSC_FER_TRIALS,
            max_iters=_BSC_FER_MAX_ITERS,
            channel_model="bsc_syndrome",
            deterministic_metadata=True,
        )
        result = run_benchmark(config)
        validate_result(result)

        rec = result["results"][0]
        fer = rec["fer"]
        assert fer > 0.0, (
            f"BSC syndrome FER must be non-zero, got {fer}"
        )
        assert fer < 1.0, (
            f"BSC syndrome FER must be less than 1, got {fer}"
        )

    def test_bsc_fer_increases_with_noise(self):
        """FER under BSC should increase with error probability."""
        config = _small_config(
            distances=[5],
            p_values=[0.01, 0.05, 0.10],
            trials=_BSC_FER_TRIALS,
            max_iters=_BSC_FER_MAX_ITERS,
            channel_model="bsc_syndrome",
            deterministic_metadata=True,
        )
        result = run_benchmark(config)
        fers = [r["fer"] for r in result["results"]]
        # FER should be non-decreasing as p increases.
        for i in range(len(fers) - 1):
            assert fers[i] <= fers[i + 1], (
                f"FER not monotone: {fers}"
            )


# ═══════════════════════════════════════════════════════════════════════
# C) Determinism Under BSC
# ═══════════════════════════════════════════════════════════════════════

class TestBSCDeterminism:

    def test_bsc_byte_identical(self):
        """Two BSC benchmark runs with identical config must produce
        byte-identical JSON (modulo timestamp)."""
        config = _small_config(
            channel_model="bsc_syndrome",
            deterministic_metadata=True,
        )
        r1 = run_benchmark(config)
        r2 = run_benchmark(config)

        j1 = dumps_result(r1)
        j2 = dumps_result(r2)
        assert j1 == j2, "BSC mode must be deterministic"

    def test_bsc_schema_valid(self):
        """BSC benchmark output must pass schema validation."""
        config = _small_config(
            channel_model="bsc_syndrome",
            deterministic_metadata=True,
        )
        result = run_benchmark(config)
        validate_result(result)


# ═══════════════════════════════════════════════════════════════════════
# D) LLR Structural Tests
# ═══════════════════════════════════════════════════════════════════════

class TestLLRStructure:

    def test_oracle_sign_depends_on_error(self):
        """Oracle LLR sign must flip where error_vector has 1s."""
        oracle = OracleChannel()
        n = 10
        p = 0.05
        e = np.zeros(n, dtype=np.uint8)
        e[3] = 1
        e[7] = 1

        llr = oracle.compute_llr(p=p, n=n, error_vector=e)

        # Positions without error: positive LLR.
        for i in range(n):
            if e[i] == 0:
                assert llr[i] > 0, f"Position {i} should be positive"
            else:
                assert llr[i] < 0, f"Position {i} should be negative"

    def test_bsc_uniform(self):
        """BSC LLR must be uniform (all elements identical)."""
        bsc = BSCSyndromeChannel()
        n = 20
        p = 0.05
        llr = bsc.compute_llr(p=p, n=n)

        assert llr.shape == (n,)
        assert np.all(llr == llr[0]), "BSC LLR must be uniform"

    def test_bsc_independent_of_error_vector(self):
        """BSC LLR must be identical regardless of error_vector content."""
        bsc = BSCSyndromeChannel()
        n = 20
        p = 0.05

        e1 = np.zeros(n, dtype=np.uint8)
        e2 = np.ones(n, dtype=np.uint8)
        e3 = np.array([1, 0] * (n // 2), dtype=np.uint8)

        llr_none = bsc.compute_llr(p=p, n=n, error_vector=None)
        llr_zeros = bsc.compute_llr(p=p, n=n, error_vector=e1)
        llr_ones = bsc.compute_llr(p=p, n=n, error_vector=e2)
        llr_mixed = bsc.compute_llr(p=p, n=n, error_vector=e3)

        np.testing.assert_array_equal(llr_none, llr_zeros)
        np.testing.assert_array_equal(llr_none, llr_ones)
        np.testing.assert_array_equal(llr_none, llr_mixed)

    def test_shapes_match(self):
        """Oracle and BSC must return same shape for same n."""
        oracle = OracleChannel()
        bsc = BSCSyndromeChannel()
        n = 15
        p = 0.05
        e = np.zeros(n, dtype=np.uint8)

        llr_oracle = oracle.compute_llr(p=p, n=n, error_vector=e)
        llr_bsc = bsc.compute_llr(p=p, n=n)

        assert llr_oracle.shape == (n,)
        assert llr_bsc.shape == (n,)

    def test_bsc_magnitude_matches_oracle_base(self):
        """BSC LLR magnitude should equal oracle base LLR (no sign flip)."""
        oracle = OracleChannel()
        bsc = BSCSyndromeChannel()
        n = 10
        p = 0.05
        e = np.zeros(n, dtype=np.uint8)  # All zeros → oracle = +base_llr

        llr_oracle = oracle.compute_llr(p=p, n=n, error_vector=e)
        llr_bsc = bsc.compute_llr(p=p, n=n)

        # With no errors, oracle LLR should equal BSC LLR element-wise.
        np.testing.assert_array_almost_equal(llr_oracle, llr_bsc)

    def test_bsc_validates_p(self):
        """BSCSyndromeChannel must raise for p outside (0, 1)."""
        bsc = BSCSyndromeChannel()
        for bad_p in [0.0, 1.0, -0.1, 1.5]:
            with pytest.raises(ValueError, match="p must be in"):
                bsc.compute_llr(p=bad_p, n=5)


# ═══════════════════════════════════════════════════════════════════════
# E) Config Backward Compatibility
# ═══════════════════════════════════════════════════════════════════════

class TestConfigBackwardCompat:

    def test_default_channel_model_omitted_from_dict(self):
        """channel_model='oracle' must be omitted from to_dict() for
        backward compatibility with pre-v3.1.3 configs."""
        config = _small_config()
        d = config.to_dict()
        assert "channel_model" not in d

    def test_bsc_channel_model_in_dict(self):
        """Non-default channel_model must appear in to_dict()."""
        config = _small_config(channel_model="bsc_syndrome")
        d = config.to_dict()
        assert d["channel_model"] == "bsc_syndrome"

    def test_config_roundtrip_oracle(self):
        """Oracle config must round-trip through JSON identically."""
        config = _small_config()
        text = config.to_json()
        restored = BenchmarkConfig.from_json(text)
        assert restored.channel_model == "oracle"
        # Serialized form must be identical.
        assert restored.to_json() == text

    def test_config_roundtrip_bsc(self):
        """BSC config must round-trip through JSON."""
        config = _small_config(channel_model="bsc_syndrome")
        text = config.to_json()
        restored = BenchmarkConfig.from_json(text)
        assert restored.channel_model == "bsc_syndrome"
        assert restored.to_json() == text

    def test_invalid_channel_model_rejected(self):
        """Invalid channel_model must raise ValueError."""
        with pytest.raises(ValueError, match="channel_model"):
            _small_config(channel_model="invalid_mode")

    def test_legacy_config_without_channel_model(self):
        """A config dict without channel_model (pre-v3.1.3) must
        load with default oracle behavior."""
        d = {
            "seed": 42,
            "distances": [3],
            "p_values": [0.01],
            "trials": 10,
            "max_iters": 20,
        }
        config = BenchmarkConfig.from_dict(d)
        assert config.channel_model == "oracle"

    def test_oracle_config_json_unchanged(self):
        """Oracle config to_json() must not change from pre-v3.1.3 output."""
        config_a = _small_config()
        json_a = config_a.to_json()
        parsed = json.loads(json_a)
        # channel_model must NOT appear in serialized output.
        assert "channel_model" not in parsed
