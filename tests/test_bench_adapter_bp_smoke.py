"""
Smoke tests for the BP decoder adapter.

Verifies that the adapter can be initialized, decode, measure runtime,
and produce a stable identity block.
"""

import numpy as np
import pytest

from bench.adapters.bp import BPAdapter
from qec_qldpc_codes import create_code, channel_llr, syndrome


@pytest.fixture
def small_code():
    """Create a small code for fast tests."""
    code = create_code(name="rate_0.50", lifting_size=3, seed=42)
    return code.H_X


@pytest.fixture
def adapter(small_code):
    """Create and initialize a BP adapter."""
    a = BPAdapter()
    a.initialize(config={
        "H": small_code,
        "mode": "min_sum",
        "max_iters": 20,
        "schedule": "flooding",
    })
    return a


class TestBPAdapterSmoke:

    def test_name(self, adapter):
        assert isinstance(adapter.name, str)
        assert "bp" in adapter.name

    def test_decode(self, adapter, small_code):
        rng = np.random.default_rng(99)
        n = small_code.shape[1]
        p = 0.05
        e = (rng.random(n) < p).astype(np.uint8)
        s = syndrome(small_code, e)
        llr = channel_llr(e, p)

        result = adapter.decode(syndrome=s, llr=llr, error_vector=e)

        assert "success" in result
        assert isinstance(result["success"], bool)
        assert "iters" in result
        assert isinstance(result["iters"], int)
        assert result["iters"] >= 0
        assert "meta" in result

    def test_serialize_identity(self, adapter):
        identity = adapter.serialize_identity()
        assert isinstance(identity, dict)
        assert identity["adapter"] == "bp"
        assert "params" in identity
        # Keys must be sorted.
        assert list(identity["params"].keys()) == sorted(identity["params"].keys())

    def test_identity_stable(self, adapter):
        """serialize_identity must return the same dict every time."""
        id1 = adapter.serialize_identity()
        id2 = adapter.serialize_identity()
        assert id1 == id2

    def test_measure_runtime(self, adapter, small_code):
        rng = np.random.default_rng(42)
        n = small_code.shape[1]
        p = 0.05
        e = (rng.random(n) < p).astype(np.uint8)
        s = syndrome(small_code, e)
        llr = channel_llr(e, p)

        rt = adapter.measure_runtime(workload={
            "llr": llr,
            "syndrome": s,
            "warmup": 2,
            "runs": 5,
            "measure_memory": False,
        })

        assert "average_latency_us" in rt
        assert isinstance(rt["average_latency_us"], int)
        assert rt["average_latency_us"] >= 0
        assert "throughput_mhz" in rt
        assert "confidence_interval_us" in rt
        assert len(rt["confidence_interval_us"]) == 2

    def test_uninitialized_decode_raises(self):
        a = BPAdapter()
        with pytest.raises(RuntimeError, match="not initialized"):
            a.decode(llr=np.array([1.0]))
