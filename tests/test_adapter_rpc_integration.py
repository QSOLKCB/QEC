"""
Tests for v3.8.0 BPAdapter RPC integration (Phase 4).

Covers:
- structural_config disabled → identical outputs to baseline
- structural_config absent → identical outputs to baseline
- structural_config enabled → adapter runs without error
- No change to serialize_identity when structural_config absent
"""

import numpy as np
import pytest

from bench.adapters.bp import BPAdapter
from qec_qldpc_codes import create_code, channel_llr, syndrome
from qec.decoder.rpc import RPCConfig, StructuralConfig


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_code():
    """Small rate-0.50 code for fast tests."""
    return create_code(name="rate_0.50", lifting_size=3, seed=42)


@pytest.fixture
def noisy_data(small_code):
    """Error, syndrome, and LLR for the small code."""
    rng = np.random.default_rng(99)
    n = small_code.H_X.shape[1]
    e = (rng.random(n) < 0.05).astype(np.uint8)
    s = syndrome(small_code.H_X, e)
    llr = channel_llr(e, 0.05)
    return small_code.H_X, e, s, llr


# ──────────────────────────────────────────────────────────────────
# Disabled identity — structural_config absent
# ──────────────────────────────────────────────────────────────────

class TestDisabledIdentityAbsent:

    def test_no_structural_config_identical_to_baseline(self, noisy_data):
        """When structural_config is not passed, output matches baseline."""
        H, e, s, llr = noisy_data

        # Baseline adapter (no structural_config)
        a_base = BPAdapter()
        a_base.initialize(config={
            "H": H,
            "mode": "min_sum",
            "max_iters": 20,
            "schedule": "flooding",
        })

        # Adapter with structural_config absent (should be identical)
        a_test = BPAdapter()
        a_test.initialize(config={
            "H": H,
            "mode": "min_sum",
            "max_iters": 20,
            "schedule": "flooding",
        })

        r_base = a_base.decode(syndrome=s, llr=llr, error_vector=e)
        r_test = a_test.decode(syndrome=s, llr=llr, error_vector=e)

        np.testing.assert_array_equal(r_base["correction"], r_test["correction"])
        assert r_base["iters"] == r_test["iters"]
        assert r_base["success"] == r_test["success"]


# ──────────────────────────────────────────────────────────────────
# Disabled identity — structural_config present but disabled
# ──────────────────────────────────────────────────────────────────

class TestDisabledIdentityPresent:

    def test_structural_disabled_identical_to_baseline(self, noisy_data):
        """When structural_config.rpc.enabled=False, output matches baseline."""
        H, e, s, llr = noisy_data

        # Baseline (no structural_config)
        a_base = BPAdapter()
        a_base.initialize(config={
            "H": H,
            "mode": "min_sum",
            "max_iters": 20,
            "schedule": "flooding",
        })

        # Adapter with structural_config disabled
        scfg = StructuralConfig(rpc=RPCConfig(enabled=False))
        a_test = BPAdapter()
        a_test.initialize(config={
            "H": H,
            "mode": "min_sum",
            "max_iters": 20,
            "schedule": "flooding",
            "structural_config": scfg,
        })

        r_base = a_base.decode(syndrome=s, llr=llr, error_vector=e)
        r_test = a_test.decode(syndrome=s, llr=llr, error_vector=e)

        np.testing.assert_array_equal(r_base["correction"], r_test["correction"])
        assert r_base["iters"] == r_test["iters"]
        assert r_base["success"] == r_test["success"]

    def test_structural_default_identical_to_baseline(self, noisy_data):
        """Default StructuralConfig() has rpc disabled → baseline identity."""
        H, e, s, llr = noisy_data

        a_base = BPAdapter()
        a_base.initialize(config={
            "H": H,
            "mode": "min_sum",
            "max_iters": 20,
            "schedule": "flooding",
        })

        a_test = BPAdapter()
        a_test.initialize(config={
            "H": H,
            "mode": "min_sum",
            "max_iters": 20,
            "schedule": "flooding",
            "structural_config": StructuralConfig(),
        })

        r_base = a_base.decode(syndrome=s, llr=llr, error_vector=e)
        r_test = a_test.decode(syndrome=s, llr=llr, error_vector=e)

        np.testing.assert_array_equal(r_base["correction"], r_test["correction"])
        assert r_base["iters"] == r_test["iters"]


# ──────────────────────────────────────────────────────────────────
# Enabled RPC — adapter runs without error
# ──────────────────────────────────────────────────────────────────

class TestEnabledRPC:

    def test_rpc_enabled_runs(self, noisy_data):
        """Adapter with RPC enabled completes without error."""
        H, e, s, llr = noisy_data

        scfg = StructuralConfig(rpc=RPCConfig(
            enabled=True, max_rows=10, w_min=2, w_max=20,
        ))
        a = BPAdapter()
        a.initialize(config={
            "H": H,
            "mode": "min_sum",
            "max_iters": 20,
            "schedule": "flooding",
            "structural_config": scfg,
        })

        result = a.decode(syndrome=s, llr=llr, error_vector=e)
        assert "success" in result
        assert "correction" in result
        assert "iters" in result
        assert isinstance(result["iters"], int)

    def test_rpc_enabled_determinism(self, noisy_data):
        """Two identical decode calls with RPC enabled produce same output."""
        H, e, s, llr = noisy_data

        scfg = StructuralConfig(rpc=RPCConfig(
            enabled=True, max_rows=10, w_min=2, w_max=20,
        ))

        a1 = BPAdapter()
        a1.initialize(config={
            "H": H,
            "mode": "min_sum",
            "max_iters": 20,
            "schedule": "flooding",
            "structural_config": scfg,
        })
        a2 = BPAdapter()
        a2.initialize(config={
            "H": H.copy(),
            "mode": "min_sum",
            "max_iters": 20,
            "schedule": "flooding",
            "structural_config": scfg,
        })

        r1 = a1.decode(syndrome=s, llr=llr)
        r2 = a2.decode(syndrome=s, llr=llr)

        np.testing.assert_array_equal(r1["correction"], r2["correction"])
        assert r1["iters"] == r2["iters"]


# ──────────────────────────────────────────────────────────────────
# serialize_identity — structural_config not leaked
# ──────────────────────────────────────────────────────────────────

class TestSerializeIdentity:

    def test_structural_config_not_in_identity(self, noisy_data):
        """structural_config must not appear in serialized identity params."""
        H, e, s, llr = noisy_data

        scfg = StructuralConfig(rpc=RPCConfig(enabled=True, max_rows=5))
        a = BPAdapter()
        a.initialize(config={
            "H": H,
            "mode": "min_sum",
            "max_iters": 20,
            "schedule": "flooding",
            "structural_config": scfg,
        })

        identity = a.serialize_identity()
        assert "structural_config" not in identity["params"]

    def test_identity_stable_with_structural_config(self, noisy_data):
        """serialize_identity remains stable across calls."""
        H, e, s, llr = noisy_data

        a = BPAdapter()
        a.initialize(config={
            "H": H,
            "mode": "min_sum",
            "max_iters": 20,
            "schedule": "flooding",
            "structural_config": StructuralConfig(),
        })

        id1 = a.serialize_identity()
        id2 = a.serialize_identity()
        assert id1 == id2
